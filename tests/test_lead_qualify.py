"""The lead_qualify example pack, scored end to end with no network.

Mirrors tests/test_example.py. The two things this pack adds over support_triage
are a conditional edge that skips work (a disqualified lead never reaches
`enrich`) and asserts that encode a policy rather than a single invariant, so
those are what most of these tests are about.
"""

import json
import os
import subprocess
import sys
import unittest

from jig.cli import resolve_model
from jig.eval import evaluate
from jig.graph import run
from jig.model import FakeModel
from jig.pack import load_pack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "examples", "lead_qualify")

CASES = 12
REJECTED = 2  # cases that leave at the gate
QUALIFIED = CASES - REJECTED

ENTERPRISE_FREE_MAIL = {
    "name": "Dan Whitfield",
    "company": "Northwind Logistics",
    "email": "dan.whitfield91@gmail.com",
    "message": "Looking at tools for our 3,000 driver fleet. Budget is signed off.",
    "headcount": 3000,
}
SMALL_AND_SLOW = {
    "name": "Chloe Barnes",
    "company": "Fernwood Design",
    "email": "chloe@fernwooddesign.studio",
    "message": "Just browsing for now, no timeline on our side.",
    "headcount": 22,
}

# A keyed script, so a test about routing does not depend on evalset order.
QUALIFIED_SCRIPT = {
    "screen step": '{"email_domain_kind": "company_domain", "company_named": true}',
    "segment step": '{"headcount_band": "10-49", "segment": "smb"}',
    "gate step": '{"disqualified": false, "disqualify_reason": "none"}',
    "enrich step": '{"industry": "software", "seniority": "ic"}',
    "signals step": '{"budget_signal": "none", "urgency": "low"}',
    "score reasoning step": "a small studio with nothing decided yet",
    "score step": '{"fit_band": "weak", "fit_reason": "no budget and no date"}',
    "route step": '{"next_action": "nurture", "owner": "marketing_nurture"}',
}


def load():
    return load_pack(PACK)


def scripted(pack):
    return resolve_model(None, pack)


def keyed(**overrides):
    return FakeModel(dict(QUALIFIED_SCRIPT, **overrides))


class TestThePackScores(unittest.TestCase):
    def test_it_scores_twelve_out_of_twelve(self):
        pack = load()
        report = evaluate(pack, scripted(pack))
        self.assertEqual((report.passed, report.total), (CASES, CASES))
        self.assertTrue(report.passed_all)
        self.assertEqual(report.by_node, {})

    def test_the_cli_validates_it(self):
        completed = subprocess.run(
            [sys.executable, "-m", "jig", "validate", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("lead_qualify", completed.stdout)

    def test_the_cli_scores_it_too(self):
        completed = subprocess.run(
            [sys.executable, "-m", "jig", "eval", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("%d/%d" % (CASES, CASES), completed.stdout)

    def test_the_evalset_and_the_fake_script_have_not_drifted(self):
        """Each key holds one answer per case that actually reaches that node.

        The branch is what makes this worth asserting: `decline` is scripted for
        the rejected cases only and the four nodes below the gate are scripted
        for the rest, so a case added to one arm and not the other desyncs every
        later case rather than failing its own.
        """
        pack = load()
        with open(os.path.join(PACK, "fakes", "script.json")) as handle:
            script = json.load(handle)
        self.assertEqual(len(pack.evalset), CASES)
        expected = {
            "screen step": CASES,
            "segment step": CASES,
            "gate step": CASES,
            "decline step": REJECTED,
            "enrich step": QUALIFIED,
            "signals step": QUALIFIED,
            "score reasoning step": QUALIFIED,
            "score step": QUALIFIED,
            "route step": QUALIFIED,
        }
        self.assertEqual({k: len(v) for k, v in script.items()}, expected)

    def test_every_scripted_key_stays_unambiguous(self):
        """FakeModel matches keys as prompt substrings, longest match winning.

        One key containing another silently reroutes a node's answers, so the
        keys are checked against each other and against the think notes, which
        are pasted into the emit prompt of the two-stage node.
        """
        with open(os.path.join(PACK, "fakes", "script.json")) as handle:
            script = json.load(handle)
        for key in script:
            for other in script:
                if key != other:
                    self.assertNotIn(key, other)
            for note in script["score reasoning step"]:
                self.assertNotIn(key, note)


class TestTheShape(unittest.TestCase):
    def setUp(self):
        self.pack = load()

    def test_it_is_the_eight_node_workflow_plus_its_endings(self):
        generate_nodes = [n for n in self.pack.nodes.values() if n.type == "generate"]
        self.assertEqual(
            sorted(node.name for node in generate_nodes),
            ["decline", "enrich", "gate", "route", "score", "screen", "segment",
             "signals"],
        )
        ends = sorted(n.name for n in self.pack.nodes.values() if n.type == "end")
        self.assertEqual(ends, ["needs_review", "nurture_queue", "rejected",
                                "sales_queue"])

    def test_every_generate_node_has_a_prompt_and_a_grammar(self):
        for node in self.pack.nodes.values():
            if node.type == "generate":
                self.assertTrue(node.prompt, node.name)
                self.assertTrue(node.grammar, node.name)

    def test_every_grammar_field_is_typed_and_mostly_enumerated(self):
        """A free string is allowed only where prose is the point."""
        prose = {"fit_reason", "reply_note"}
        for node in self.pack.nodes.values():
            if node.type != "generate":
                continue
            properties = node.grammar["properties"]
            self.assertEqual(node.grammar.get("additionalProperties"), False, node.name)
            self.assertEqual(sorted(node.grammar["required"]), sorted(properties))
            for field, schema in properties.items():
                self.assertIn("type", schema, "%s.%s" % (node.name, field))
                if field in prose or schema["type"] == "boolean":
                    continue
                self.assertIn("enum", schema, "%s.%s" % (node.name, field))

    def test_only_the_judgement_node_is_two_stage(self):
        two_stage = [n.name for n in self.pack.nodes.values() if n.two_stage]
        self.assertEqual(two_stage, ["score"])
        self.assertIn("score reasoning step", self.pack.nodes["score"].think_prompt)

    def test_the_deterministic_rules_are_asserted_and_routed_to_a_human(self):
        asserted = sorted(
            n.name for n in self.pack.nodes.values() if n.assert_expr
        )
        self.assertEqual(asserted, ["gate", "route", "segment"])
        for name in asserted:
            self.assertEqual(self.pack.nodes[name].on_fail, "needs_review")

    def test_the_gate_has_a_conditional_edge_that_skips_enrichment(self):
        out = self.pack.edges_from("gate")
        self.assertEqual([edge.target for edge in out], ["decline", "enrich"])
        self.assertEqual(out[0].when, {"disqualified": True})
        self.assertIsNone(out[1].when)  # the default arm is last

    def test_no_node_writes_over_a_form_field(self):
        """Merge-mode commit refuses to overwrite a run input; catch it at load."""
        inputs = {"name", "company", "email", "message", "headcount"}
        for node in self.pack.nodes.values():
            if node.type != "generate":
                continue
            self.assertEqual(inputs & set(node.grammar["properties"]), set(), node.name)


class TestTheGateSkipsWork(unittest.TestCase):
    def test_a_disqualified_lead_never_reaches_enrichment(self):
        pack = load()
        model = FakeModel({
            "screen step": '{"email_domain_kind": "free_mail", "company_named": true}',
            "segment step": '{"headcount_band": "1000+", "segment": "enterprise"}',
            "gate step":
                '{"disqualified": true, "disqualify_reason": "free_mail_enterprise"}',
            "decline step":
                '{"next_action": "reject", "reply_note": "Please write in from work."}',
        })
        result = run(pack, model, ENTERPRISE_FREE_MAIL)
        self.assertEqual(result.path,
                         ["screen", "segment", "gate", "decline", "rejected"])
        self.assertEqual(result.end_node, "rejected")
        self.assertEqual(result.output["next_action"], "reject")
        self.assertNotIn("fit_band", result.state)
        self.assertNotIn("industry", result.state)

    def test_skipping_it_is_what_makes_the_gate_worth_a_call(self):
        """Four generations for a rejected lead against eight for a qualified one."""
        pack = load()
        rejected = FakeModel({
            "screen step": '{"email_domain_kind": "free_mail", "company_named": true}',
            "segment step": '{"headcount_band": "1000+", "segment": "enterprise"}',
            "gate step":
                '{"disqualified": true, "disqualify_reason": "free_mail_enterprise"}',
            "decline step": '{"next_action": "reject", "reply_note": "work email please"}',
        })
        run(pack, rejected, ENTERPRISE_FREE_MAIL)
        qualified = keyed()
        run(pack, qualified, SMALL_AND_SLOW)
        self.assertEqual(rejected.call_count, 4)
        # seven nodes plus the think stage of `score`
        self.assertEqual(qualified.call_count, 8)
        self.assertIsNone(qualified.calls[5].grammar)  # think is unconstrained

    def test_a_free_mailbox_alone_does_not_disqualify(self):
        """Only enterprise is refused; a one-person company on gmail goes through."""
        pack = load()
        model = keyed(**{
            "screen step": '{"email_domain_kind": "free_mail", "company_named": true}',
            "segment step": '{"headcount_band": "1-9", "segment": "solo"}',
        })
        result = run(pack, model, dict(SMALL_AND_SLOW,
                                       email="victor.osei@gmail.com", headcount=1))
        self.assertEqual(result.end_node, "nurture_queue")
        # a qualified lead is by definition not disqualified, so the queue end
        # nodes do not project the flag; state still carries it
        self.assertFalse(result.state["disqualified"])
        self.assertEqual(result.output["email_domain_kind"], "free_mail")


class TestRoutingAndTheAsserts(unittest.TestCase):
    def test_a_strong_enterprise_lead_reaches_the_enterprise_ae(self):
        pack = load()
        model = keyed(**{
            "segment step": '{"headcount_band": "1000+", "segment": "enterprise"}',
            "signals step": '{"budget_signal": "budget_approved", "urgency": "high"}',
            "score step": '{"fit_band": "strong", "fit_reason": "budget is approved"}',
            "route step": '{"next_action": "route_to_sales", "owner": "ae_enterprise"}',
        })
        result = run(pack, model, ENTERPRISE_FREE_MAIL)
        self.assertEqual(result.end_node, "sales_queue")
        self.assertEqual(result.output["owner"], "ae_enterprise")

    def test_a_moderate_lead_in_a_hurry_still_goes_to_sales(self):
        pack = load()
        model = keyed(**{
            "signals step": '{"budget_signal": "none", "urgency": "high"}',
            "score step": '{"fit_band": "moderate", "fit_reason": "deadline is real"}',
            "route step": '{"next_action": "route_to_sales", "owner": "sdr"}',
        })
        result = run(pack, model, SMALL_AND_SLOW)
        self.assertEqual(result.end_node, "sales_queue")
        self.assertEqual(result.output["owner"], "sdr")

    def test_a_segment_that_contradicts_its_headcount_band_is_caught(self):
        pack = load()
        model = keyed(**{
            "segment step": '{"headcount_band": "10-49", "segment": "enterprise"}',
        })
        result = run(pack, model, SMALL_AND_SLOW)
        self.assertEqual(result.end_node, "needs_review")
        self.assertEqual(result.failures[0].node, "segment")
        self.assertEqual(result.failures[0].attempts, 3)
        self.assertNotIn("segment", result.state)

    def test_a_gate_that_ignores_its_own_rule_is_caught(self):
        pack = load()
        model = keyed(**{
            "screen step": '{"email_domain_kind": "free_mail", "company_named": true}',
            "segment step": '{"headcount_band": "1000+", "segment": "enterprise"}',
            "gate step": '{"disqualified": false, "disqualify_reason": "none"}',
        })
        result = run(pack, model, ENTERPRISE_FREE_MAIL)
        self.assertEqual(result.end_node, "needs_review")
        self.assertEqual(result.failures[0].node, "gate")
        self.assertNotIn("disqualified", result.state)

    def test_a_weak_lead_sent_to_sales_is_caught(self):
        """The grammar is satisfied; the routing policy is not."""
        pack = load()
        model = keyed(**{
            "route step": '{"next_action": "route_to_sales", "owner": "sdr"}',
        })
        result = run(pack, model, SMALL_AND_SLOW)
        self.assertEqual(result.end_node, "needs_review")
        self.assertEqual(result.failures[0].node, "route")
        self.assertNotIn("next_action", result.state)

    def test_an_owner_that_does_not_match_the_action_is_caught(self):
        pack = load()
        model = keyed(**{"route step": '{"next_action": "nurture", "owner": "sdr"}'})
        result = run(pack, model, SMALL_AND_SLOW)
        self.assertEqual(result.end_node, "needs_review")
        self.assertEqual(result.failures[0].node, "route")


class TestTheScratchpadStaysPrivate(unittest.TestCase):
    def test_the_reasoning_conditions_the_emit_and_stops_there(self):
        pack = load()
        model = keyed()
        result = run(pack, model, SMALL_AND_SLOW)
        notes = QUALIFIED_SCRIPT["score reasoning step"]
        self.assertIn(notes, model.calls[6].prompt)  # the emit half of `score`
        self.assertNotIn("scratchpad", result.state)
        self.assertNotIn(notes, json.dumps(result.state))
        self.assertNotIn(notes, json.dumps(result.output))


class TestABlankHeadcountStillRuns(unittest.TestCase):
    def test_a_null_headcount_renders_rather_than_failing_the_node(self):
        """The form field is optional; the pack requires the key with a null value."""
        pack = load()
        model = keyed()
        result = run(pack, model, dict(SMALL_AND_SLOW, headcount=None))
        self.assertEqual(result.end_node, "nurture_queue")
        self.assertIn("Form headcount: null", model.calls[1].prompt)
