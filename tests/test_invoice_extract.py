"""The invoice-extraction example pack, scored end to end with no network.

Mirrors tests/test_example.py. What is worth testing here beyond the score is the
division of labour the pack is built on: the model copies fields, and every piece of
arithmetic and policy is an `assert` node the runtime evaluates itself.
"""

import json
import os
import subprocess
import sys
import unittest

from jig.cli import resolve_model
from jig.errors import AssertFailed, ExprError
from jig.eval import evaluate
from jig.graph import run
from jig.model import FakeModel
from jig.pack import load_pack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "examples", "invoice_extract")

TODAY = "2026-08-24"

# One invoice, reused: each test edits the numbers it cares about rather than
# carrying a second wall of text.
INVOICE = """Ridgeline Tools
Invoice RT-7781
Date: 2026-08-14    Due: 2026-09-14
Replacement drill bits, box of 12 ...... $100.00
Subtotal ...............................  $100.00
Tax ....................................    $8.00
TOTAL DUE .............................. $108.00"""


def load():
    return load_pack(PACK)


def scripted(pack):
    return resolve_model(None, pack)


def keyed(header, amounts, due, review=None, think="some notes"):
    """A FakeModel keyed by prompt substring, so routing is what the test varies."""
    script = {
        "the header step": json.dumps(header),
        "the amounts step": json.dumps(amounts),
        "the due-date step": json.dumps(due),
        "the review reasoning step": think,
        "the totals-mismatch step":
            '{"needs_review": true, "review_reason": "totals_mismatch"}',
        "the tax-out-of-range step":
            '{"needs_review": true, "review_reason": "tax_out_of_range"}',
        "the unexpected-currency step":
            '{"needs_review": true, "review_reason": "unexpected_currency"}',
        "the past-due step":
            '{"needs_review": true, "review_reason": "past_due"}',
    }
    if review is not None:
        script["the review step"] = json.dumps(review)
    return FakeModel(script)


HEADER = {"supplier_name": "Ridgeline Tools", "invoice_number": "RT-7781",
          "currency": "USD"}
AMOUNTS = {"subtotal": 100.0, "tax_amount": 8.0, "total_amount": 108.0}
DUE = {"due_date": "2026-09-14"}
CLEAR = {"needs_review": False, "review_reason": "none"}


class TestThePackScores(unittest.TestCase):
    def test_it_scores_twelve_out_of_twelve(self):
        pack = load()
        report = evaluate(pack, scripted(pack))
        self.assertEqual((report.passed, report.total), (12, 12))
        self.assertTrue(report.passed_all)
        self.assertEqual(report.by_node, {})

    def test_the_cli_scores_it_too(self):
        completed = subprocess.run(
            [sys.executable, "-m", "jig", "eval", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("12/12", completed.stdout)

    def test_the_cli_validates_it(self):
        completed = subprocess.run(
            [sys.executable, "-m", "jig", "validate", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("12 evalset cases", completed.stdout)

    def test_the_evalset_and_the_fake_script_have_not_drifted(self):
        """Every scripted list must hold exactly one answer per case that reaches it."""
        pack = load()
        with open(os.path.join(PACK, "fakes", "script.json")) as handle:
            script = json.load(handle)
        self.assertEqual(len(pack.evalset), 12)
        reviewed = sum(1 for case in pack.evalset
                       if case.expect["review_reason"] in
                       ("none", "missing_identifier", "unusual_document"))
        expected = {
            "the header step": 12,
            "the amounts step": 12,
            "the due-date step": 12,
            "the review reasoning step": reviewed,
            "the review step": reviewed,
        }
        for key, count in expected.items():
            self.assertEqual(len(script[key]), count,
                             "%s has %d responses, expected %d"
                             % (key, len(script[key]), count))
        for key, value in script.items():
            if key not in expected:
                # A forced node answers with a constant, so its script is one string.
                self.assertIsInstance(value, str, key)

    def test_every_case_declares_the_seven_fields_the_workflow_promises(self):
        for case in load().evalset:
            for field in ("supplier_name", "invoice_number", "currency",
                          "total_amount", "tax_amount", "due_date", "needs_review"):
                self.assertIn(field, case.expect, case.name)


class TestThePackShape(unittest.TestCase):
    def setUp(self):
        self.pack = load()

    def test_the_model_copies_and_judges_and_the_runtime_calculates(self):
        by_type = {}
        for node in self.pack.nodes.values():
            by_type.setdefault(node.type, []).append(node.name)
        self.assertEqual(
            sorted(by_type["generate"]),
            ["amounts", "due", "flag_currency", "flag_overdue", "flag_tax",
             "flag_totals", "header", "review"],
        )
        self.assertEqual(
            sorted(by_type["assert"]),
            ["currency_check", "date_check", "inputs_check", "tax_check",
             "totals_check"],
        )
        self.assertEqual(sorted(by_type["end"]),
                         ["accepted", "flagged", "manual_review"])

    def test_every_generate_node_has_a_prompt_and_a_grammar(self):
        for node in self.pack.nodes.values():
            if node.type == "generate":
                self.assertTrue(node.prompt, node.name)
                self.assertTrue(node.grammar, node.name)

    def test_every_grammar_closes_itself_and_names_its_fields(self):
        for node in self.pack.nodes.values():
            if node.type != "generate":
                continue
            self.assertEqual(node.grammar.get("additionalProperties"), False, node.name)
            self.assertEqual(sorted(node.grammar["required"]),
                             sorted(node.grammar["properties"]), node.name)

    def test_every_invoice_check_can_divert(self):
        """An assert with nowhere to go kills the run; each of these has a reason node."""
        for node in self.pack.nodes.values():
            if node.type == "assert" and node.name != "inputs_check":
                self.assertTrue(node.expr, node.name)
                self.assertIn(node.on_fail, self.pack.nodes, node.name)

    def test_the_input_guard_deliberately_has_nowhere_to_divert(self):
        guard = self.pack.nodes["inputs_check"]
        self.assertEqual(self.pack.entry, "inputs_check")
        self.assertIsNone(guard.on_fail)

    def test_the_review_node_is_the_only_two_stage_one(self):
        two_stage = [n.name for n in self.pack.nodes.values() if n.two_stage]
        self.assertEqual(two_stage, ["review"])
        self.assertIn("reasoning step", self.pack.nodes["review"].think_prompt)

    def test_the_review_node_declares_a_deterministic_invariant(self):
        review = self.pack.nodes["review"]
        self.assertIn("needs_review", review.assert_expr)
        self.assertEqual(review.on_fail, "manual_review")

    def test_the_forced_nodes_can_only_return_one_answer(self):
        for name, reason in (("flag_totals", "totals_mismatch"),
                             ("flag_tax", "tax_out_of_range"),
                             ("flag_currency", "unexpected_currency"),
                             ("flag_overdue", "past_due")):
            properties = self.pack.nodes[name].grammar["properties"]
            self.assertEqual(properties["needs_review"]["enum"], [True], name)
            self.assertEqual(properties["review_reason"]["enum"], [reason], name)


class TestRunningOneInvoice(unittest.TestCase):
    def test_a_clean_invoice_is_accepted(self):
        pack = load()
        result = run(pack, keyed(HEADER, AMOUNTS, DUE, CLEAR),
                     {"invoice_text": INVOICE, "today": TODAY})
        self.assertEqual(result.end_node, "accepted")
        self.assertEqual(result.output["supplier_name"], "Ridgeline Tools")
        self.assertEqual(result.output["total_amount"], 108.0)
        self.assertEqual(result.output["needs_review"], False)
        self.assertEqual(result.output["review_reason"], "none")

    def test_the_four_checks_cost_no_model_calls(self):
        """Three extraction calls, one think, one emit — the asserts are free."""
        pack = load()
        model = keyed(HEADER, AMOUNTS, DUE, CLEAR)
        run(pack, model, {"invoice_text": INVOICE, "today": TODAY})
        self.assertEqual(model.call_count, 5)
        self.assertIsNone(model.calls[3].grammar)  # the think stage is unconstrained

    def test_a_flagged_invoice_costs_one_call_fewer_than_a_clean_one(self):
        pack = load()
        model = keyed(HEADER, dict(AMOUNTS, total_amount=118.0), DUE, CLEAR)
        run(pack, model, {"invoice_text": INVOICE, "today": TODAY})
        self.assertEqual(model.call_count, 4)  # no think, no emit: the check decided

    def test_the_scratchpad_never_reaches_the_output(self):
        pack = load()
        model = keyed(HEADER, AMOUNTS, DUE, CLEAR,
                      think="Nothing about this document is unusual.")
        result = run(pack, model, {"invoice_text": INVOICE, "today": TODAY})
        self.assertIn("Nothing about this document", model.calls[4].prompt)
        self.assertNotIn("scratchpad", result.state)
        self.assertNotIn("Nothing about this document", json.dumps(result.state))
        self.assertNotIn("Nothing about this document", json.dumps(result.output))


class TestTheChecksHoldWithoutTheModel(unittest.TestCase):
    """Each check is given a model that would happily wave the invoice through."""

    def walk(self, amounts=None, header=None, due=None, today=TODAY):
        pack = load()
        model = keyed(header or HEADER, amounts or AMOUNTS, due or DUE, CLEAR)
        return run(pack, model, {"invoice_text": INVOICE, "today": today})

    def test_totals_that_do_not_add_up_are_caught_by_the_runtime(self):
        result = self.walk(amounts=dict(AMOUNTS, total_amount=118.0))
        self.assertEqual(result.end_node, "flagged")
        self.assertEqual(result.output["review_reason"], "totals_mismatch")
        self.assertEqual(result.output["needs_review"], True)
        self.assertEqual(result.provenance["review_reason"], "flag_totals")
        self.assertNotIn("review", result.path)
        # A diverted `assert` node is routing, not failure: nothing to report.
        self.assertEqual(result.failures, [])

    def test_rounding_noise_is_not_a_mismatch(self):
        """11.37 + 2.27 is 13.639999999999999 in binary; the tolerance absorbs it."""
        self.assertNotEqual(11.37 + 2.27, 13.64)
        result = self.walk(amounts={"subtotal": 11.37, "tax_amount": 2.27,
                                    "total_amount": 13.64})
        self.assertEqual(result.end_node, "accepted")

    def test_a_tax_line_larger_than_the_band_is_caught(self):
        result = self.walk(amounts={"subtotal": 400.0, "tax_amount": 160.0,
                                    "total_amount": 560.0})
        self.assertEqual(result.output["review_reason"], "tax_out_of_range")

    def test_a_currency_the_grammar_allows_but_policy_does_not(self):
        pack = load()
        self.assertIn("JPY", pack.nodes["header"].grammar["properties"]["currency"]["enum"])
        result = self.walk(header=dict(HEADER, currency="JPY"))
        self.assertEqual(result.output["review_reason"], "unexpected_currency")

    def test_a_due_date_in_the_past_is_caught(self):
        result = self.walk(due={"due_date": "2026-05-01"})
        self.assertEqual(result.output["review_reason"], "past_due")

    def test_a_due_date_of_today_is_not_past_due(self):
        result = self.walk(due={"due_date": TODAY})
        self.assertEqual(result.end_node, "accepted")

    def test_an_invoice_with_no_due_date_is_not_past_due(self):
        result = self.walk(due={"due_date": None})
        self.assertEqual(result.end_node, "accepted")
        self.assertIsNone(result.output["due_date"])

    def test_the_first_failing_check_wins(self):
        """Broken totals AND a bad currency: the earlier check names the reason."""
        result = self.walk(amounts=dict(AMOUNTS, total_amount=118.0),
                           header=dict(HEADER, currency="JPY"))
        self.assertEqual(result.output["review_reason"], "totals_mismatch")


class TestTheInputGuard(unittest.TestCase):
    def test_a_run_without_today_stops_instead_of_answering(self):
        """Without the guard this run reports an ordinary invoice as `past_due`."""
        pack = load()
        model = keyed(HEADER, AMOUNTS, DUE, CLEAR)
        with self.assertRaises(ExprError) as caught:
            run(pack, model, {"invoice_text": INVOICE})
        self.assertIn("today", str(caught.exception))
        self.assertEqual(model.call_count, 0)  # nothing was spent

    def test_a_today_that_is_not_iso_stops_too(self):
        """A date jig would compare lexicographically against ISO due dates."""
        pack = load()
        with self.assertRaises(AssertFailed):
            run(pack, keyed(HEADER, AMOUNTS, DUE, CLEAR),
                {"invoice_text": INVOICE, "today": "24/08/2026"})


class TestTheReviewInvariantHolds(unittest.TestCase):
    def test_a_contradictory_review_is_caught_and_routed_to_a_human(self):
        """The review grammar is satisfied but the node's assert is not."""
        pack = load()
        model = keyed(HEADER, AMOUNTS, DUE,
                      {"needs_review": False, "review_reason": "unusual_document"})
        result = run(pack, model, {"invoice_text": INVOICE, "today": TODAY})
        self.assertEqual(result.end_node, "manual_review")
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].node, "review")
        self.assertEqual(result.failures[0].attempts, 3)
        # Nothing the rejected node said reached state, or the human's copy.
        self.assertNotIn("needs_review", result.state)
        self.assertNotIn("needs_review", result.output)
        self.assertEqual(result.output["supplier_name"], "Ridgeline Tools")

    def test_a_flag_without_a_reason_is_the_same_contradiction(self):
        pack = load()
        model = keyed(HEADER, AMOUNTS, DUE,
                      {"needs_review": True, "review_reason": "none"})
        result = run(pack, model, {"invoice_text": INVOICE, "today": TODAY})
        self.assertEqual(result.end_node, "manual_review")


if __name__ == "__main__":
    unittest.main()
