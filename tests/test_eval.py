"""T8 — scoring a pack against its evalset, and blaming the right node.

The second half of this file covers the **tier split**: which cases the pack answered on
its own, which it handed to a human, and how right the automatic ones were. That last
number is the one a deployment is argued over, and a single pass rate cannot express it.
"""

import contextlib
import io
import os
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from jig import cli
from jig.eval import (
    TIER_AUTO,
    TIER_ESCALATED,
    TIER_FAILED,
    CaseResult,
    Escalation,
    Report,
    evaluate,
)
from jig.graph import Failure, RunResult
from jig.model import FakeModel
from jig.pack import Edge, EvalCase, Node, Pack


def two_node_pack():
    """classify -> extract -> end. Each node writes one field."""
    return Pack(
        path="<memory>",
        name="triage",
        version=1,
        entry="classify",
        model=None,
        nodes={
            "classify": Node(
                name="classify",
                type="generate",
                prompt="classify {ticket}",
                grammar={"type": "object"},
            ),
            "extract": Node(
                name="extract",
                type="generate",
                prompt="extract from {ticket}",
                grammar={"type": "object"},
            ),
            "done": Node(name="done", type="end"),
        },
        edges=[Edge("classify", "extract", None), Edge("extract", "done", None)],
        evalset=[
            EvalCase({"ticket": "t1"}, {"category": "billing", "amount": 1}, "one"),
            EvalCase({"ticket": "t2"}, {"category": "billing", "amount": 2}, "two"),
            EvalCase({"ticket": "t3"}, {"category": "billing", "amount": 3}, "three"),
            EvalCase({"ticket": "t4"}, {"category": "billing", "amount": 4}, "four"),
        ],
    )


def scripted(amounts):
    """One classify + one extract response per case, keyed so order is explicit."""
    return FakeModel(
        {
            "classify": ['{"category": "billing"}'] * len(amounts),
            "extract": ['{"amount": %d}' % amount for amount in amounts],
        }
    )


class TestScoring(unittest.TestCase):
    def test_a_clean_pack_scores_everything(self):
        report = evaluate(two_node_pack(), scripted([1, 2, 3, 4]))
        self.assertEqual((report.passed, report.total), (4, 4))
        self.assertTrue(report.passed_all)

    def test_a_pack_scoring_three_of_four_reports_exactly_that(self):
        report = evaluate(two_node_pack(), scripted([1, 2, 99, 4]))
        self.assertEqual(report.passed, 3)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.total, 4)
        self.assertFalse(report.passed_all)

    def test_the_failure_is_attributed_to_the_node_that_wrote_the_field(self):
        report = evaluate(two_node_pack(), scripted([1, 2, 99, 4]))
        self.assertEqual(report.by_node, {"extract": 1})

    def test_a_failure_in_the_first_node_is_attributed_there(self):
        model = FakeModel(
            {
                "classify": [
                    '{"category": "billing"}',
                    '{"category": "technical"}',
                    '{"category": "billing"}',
                    '{"category": "billing"}',
                ],
                "extract": ['{"amount": %d}' % n for n in (1, 2, 3, 4)],
            }
        )
        report = evaluate(two_node_pack(), model)
        self.assertEqual(report.by_node, {"classify": 1})

    def test_per_case_results_carry_which_case_failed(self):
        report = evaluate(two_node_pack(), scripted([1, 2, 99, 4]))
        self.assertEqual([case.passed for case in report.cases], [True, True, False, True])
        self.assertEqual(report.cases[2].name, "three")

    def test_a_mismatch_records_expected_actual_and_node(self):
        report = evaluate(two_node_pack(), scripted([1, 2, 99, 4]))
        mismatch = report.cases[2].mismatches[0]
        self.assertEqual(mismatch.field, "amount")
        self.assertEqual(mismatch.expected, 3)
        self.assertEqual(mismatch.actual, 99)
        self.assertEqual(mismatch.node, "extract")

    def test_only_the_declared_expect_fields_are_compared(self):
        pack = two_node_pack()
        pack.evalset[0] = EvalCase({"ticket": "t1"}, {"category": "billing"}, "one")
        report = evaluate(pack, scripted([1, 2, 3, 4]))
        self.assertTrue(report.cases[0].passed)

    def test_a_missing_output_field_is_a_mismatch_not_a_crash(self):
        model = FakeModel({"classify": '{"category": "billing"}', "extract": '{"other": 1}'})
        report = evaluate(two_node_pack(), model)
        self.assertEqual(report.passed, 0)
        self.assertIn("missing", report.cases[0].mismatches[0].note)


class TestRunFailures(unittest.TestCase):
    def test_a_node_that_exhausts_its_ladder_fails_the_case_and_is_blamed(self):
        pack = two_node_pack()
        pack.nodes["extract"] = Node(
            name="extract",
            type="generate",
            prompt="extract from {ticket}",
            grammar={"type": "object", "required": ["amount"]},
        )
        model = FakeModel({"classify": '{"category": "billing"}', "extract": '{"nope": 1}'})
        report = evaluate(pack, model)
        self.assertEqual(report.passed, 0)
        self.assertEqual(report.by_node, {"extract": 4})
        self.assertIn("amount", report.cases[0].error)

    def test_an_unexpected_exception_fails_only_its_own_case(self):
        """A backend that falls over mid-suite must not take the suite with it."""
        pack = two_node_pack()
        pack.evalset[:] = [
            EvalCase({"ticket": "t%d" % n}, {"category": "billing", "amount": 1}, "c%d" % n)
            for n in (1, 2, 3)
        ]

        class Grumpy:
            """Dies on the second case's extract call, then behaves again."""

            def __init__(self):
                self.seen = 0

            def generate(self, prompt, grammar=None, max_tokens=512):
                self.seen += 1
                if self.seen == 4:
                    raise RuntimeError("backend fell over")
                return '{"category": "billing", "amount": 1}'

        report = evaluate(pack, Grumpy())
        self.assertEqual([case.passed for case in report.cases], [True, False, True])
        self.assertIn("backend fell over", report.cases[1].error)
        self.assertIn("RuntimeError", report.cases[1].error)
        self.assertEqual(report.cases[1].node, "extract")
        self.assertEqual(report.by_node, {"extract": 1})

    def test_a_crash_in_the_first_node_is_attributed_to_it(self):
        pack = two_node_pack()
        pack.evalset[:] = pack.evalset[:1]

        class Dead:
            def generate(self, prompt, grammar=None, max_tokens=512):
                raise RuntimeError("nope")

        report = evaluate(pack, Dead())
        self.assertEqual(report.cases[0].node, "classify")

    def test_a_diverted_failure_is_counted_against_its_node(self):
        pack = two_node_pack()
        pack.nodes["classify"] = Node(
            name="classify",
            type="generate",
            prompt="classify {ticket}",
            grammar={"type": "object", "required": ["category"]},
            on_fail="done",
        )
        model = FakeModel({"classify": '{"wrong": 1}', "extract": '{"amount": 1}'})
        report = evaluate(pack, model)
        self.assertEqual(report.by_node, {"classify": 4})


class TestReportShape(unittest.TestCase):
    def test_the_report_names_the_pack(self):
        self.assertEqual(evaluate(two_node_pack(), scripted([1, 2, 3, 4])).pack, "triage")

    def test_summary_text_states_the_score(self):
        summary = evaluate(two_node_pack(), scripted([1, 2, 99, 4])).summary()
        self.assertIn("3/4", summary)
        self.assertIn("three", summary)
        self.assertIn("extract", summary)

    def test_summary_of_a_clean_run_says_so(self):
        self.assertIn("4/4", evaluate(two_node_pack(), scripted([1, 2, 3, 4])).summary())

    def test_an_empty_evalset_is_an_error_not_a_silent_pass(self):
        pack = two_node_pack()
        pack.evalset.clear()
        with self.assertRaises(ValueError):
            evaluate(pack, scripted([]))

    def test_a_model_factory_is_called_once_per_case(self):
        made = []

        def factory():
            model = FakeModel({"classify": '{"category": "billing"}', "extract": '{"amount": 1}'})
            made.append(model)
            return model

        pack = two_node_pack()
        pack.evalset[:] = pack.evalset[:2]
        pack.evalset[1] = EvalCase({"ticket": "t2"}, {"category": "billing", "amount": 1}, "two")
        report = evaluate(pack, factory)
        self.assertEqual(len(made), 2)
        self.assertEqual(report.passed, 2)

    def test_each_case_gets_its_own_run_id(self):
        report = evaluate(two_node_pack(), scripted([1, 2, 3, 4]))
        self.assertEqual(len({case.run_id for case in report.cases}), 4)

    def test_cases_can_be_filtered_to_a_subset(self):
        pack = two_node_pack()
        report = evaluate(pack, scripted([1, 2]), cases=pack.evalset[:2])
        self.assertEqual(report.total, 2)


class TestAttributionOrder(unittest.TestCase):
    """Which node gets blamed must come from the run, not from the evalset's key order.

    `expect` is a JSON object the pack author typed; its key order is an accident of
    editing. `run_result.path` is the order the walker actually visited nodes in, and
    the earliest failing node is the one worth fixing first — a wrong `category` from
    `classify` is usually why `extract` went on to produce a wrong `amount`.
    """

    def failing_both_nodes(self):
        return FakeModel(
            {"classify": '{"category": "technical"}', "extract": '{"amount": 99}'}
        )

    def test_the_earliest_node_is_blamed_whatever_order_expect_lists(self):
        pack = two_node_pack()
        pack.evalset[:] = [
            EvalCase({"ticket": "t1"}, {"amount": 1, "category": "billing"}, "one")
        ]
        report = evaluate(pack, self.failing_both_nodes())
        self.assertEqual(report.cases[0].node, "classify")
        self.assertEqual(report.by_node, {"classify": 1})

    def test_the_same_case_written_the_other_way_round_blames_the_same_node(self):
        """The point of the fix: reordering the JSON keys changes nothing."""
        pack = two_node_pack()
        pack.evalset[:] = [
            EvalCase({"ticket": "t1"}, {"category": "billing", "amount": 1}, "one")
        ]
        report = evaluate(pack, self.failing_both_nodes())
        self.assertEqual(report.cases[0].node, "classify")

    def test_a_field_no_node_wrote_does_not_swallow_the_blame(self):
        """An expectation on an input field has no provenance; it must not hide the node."""
        pack = two_node_pack()
        pack.evalset[:] = [
            EvalCase({"ticket": "t1"}, {"ticket": "other", "category": "technical"}, "one")
        ]
        model = FakeModel({"classify": '{"category": "billing"}', "extract": '{"amount": 1}'})
        report = evaluate(pack, model)
        self.assertEqual(report.cases[0].node, "classify")

    def test_the_summary_names_the_earliest_node(self):
        pack = two_node_pack()
        pack.evalset[:] = [
            EvalCase({"ticket": "t1"}, {"amount": 1, "category": "billing"}, "one")
        ]
        summary = evaluate(pack, self.failing_both_nodes()).summary()
        self.assertIn("FAIL one [classify]", summary)


# ---------------------------------------------------------------- the tier split


def gated_pack(cases, on_fail="human"):
    """classify -> done, with a declared way to hand the case to a person.

    `retries=0` so a rejected generation burns the ladder on its first sample, which is
    what makes an escalation happen on a scripted model with one line per case.
    """
    return Pack(
        path="<memory>",
        name="triage",
        version=1,
        entry="classify",
        model=None,
        nodes={
            "classify": Node(
                name="classify",
                type="generate",
                prompt="classify {ticket}",
                grammar={"type": "object", "required": ["category"]},
                retries=0,
                on_fail=on_fail,
            ),
            "done": Node(name="done", type="end"),
            "human": Node(name="human", type="end"),
        },
        edges=[Edge("classify", "done", None)],
        evalset=list(cases),
    )


MIXED_CASES = [
    EvalCase({"ticket": "t1"}, {"category": "billing"}, "answered"),
    EvalCase({"ticket": "t2"}, {"category": "billing"}, "answered wrongly"),
    EvalCase({"ticket": "t3"}, {}, "handed over", end="human", rescued=True),
    EvalCase({"ticket": "t4"}, {"category": "billing"}, "answered too"),
]

MIXED_SCRIPT = [
    '{"category": "billing"}',
    '{"category": "technical"}',   # ran unsupervised and got it wrong
    "not json at all",             # burns the ladder, takes on_fail
    '{"category": "billing"}',
]


def mixed_report(on_fail="human"):
    pack = gated_pack(MIXED_CASES, on_fail=on_fail)
    return evaluate(pack, FakeModel({"classify": list(MIXED_SCRIPT)}))


class TestTiers(unittest.TestCase):
    """Every case lands in exactly one of auto / escalated / failed."""

    def test_a_clean_run_is_auto(self):
        report = evaluate(two_node_pack(), scripted([1, 2, 3, 4]))
        self.assertEqual([case.tier for case in report.cases], [TIER_AUTO] * 4)
        self.assertEqual(report.tier_counts[TIER_AUTO], 4)

    def test_a_wrong_answer_with_nothing_diverted_is_still_auto(self):
        """The pack answered by itself. It answered badly — that is the accuracy, not the tier."""
        report = evaluate(two_node_pack(), scripted([1, 2, 99, 4]))
        self.assertEqual([case.tier for case in report.cases], [TIER_AUTO] * 4)
        self.assertFalse(report.cases[2].passed)

    def test_a_diverted_node_escalates_its_case(self):
        report = mixed_report()
        self.assertEqual(report.cases[2].tier, TIER_ESCALATED)
        self.assertEqual(report.tier_counts[TIER_ESCALATED], 1)

    def test_an_escalated_case_names_the_node_that_handed_it_over(self):
        report = mixed_report()
        self.assertEqual(report.cases[2].escalated_by, "classify")
        self.assertEqual(report.escalated_by, {"classify": 1})

    def test_an_escalation_records_which_signal_caused_it(self):
        escalation = mixed_report().cases[2].escalations[0]
        self.assertEqual(escalation.kind, "failed")
        self.assertIn("classify", escalation.node)

    def test_a_run_with_nowhere_to_divert_is_the_failed_tier(self):
        """No `on_fail`: nothing routed it, so it is not an escalation — it is a hole."""
        pack = gated_pack(MIXED_CASES[2:3], on_fail=None)
        report = evaluate(pack, FakeModel({"classify": ["not json at all"]}))
        self.assertEqual(report.cases[0].tier, TIER_FAILED)
        self.assertEqual(report.failed_by, {"classify": 1})
        self.assertEqual(report.tier_counts, {"auto": 0, "escalated": 0, "failed": 1})

    def test_a_backend_that_falls_over_is_the_failed_tier_not_an_escalation(self):
        class Dead:
            def generate(self, prompt, grammar=None, max_tokens=512):
                raise RuntimeError("nope")

        pack = two_node_pack()
        pack.evalset[:] = pack.evalset[:1]
        report = evaluate(pack, Dead())
        self.assertEqual(report.cases[0].tier, TIER_FAILED)
        self.assertEqual(report.failed_by, {"classify": 1})

    def test_the_three_tiers_always_account_for_every_case(self):
        report = mixed_report()
        self.assertEqual(sum(report.tier_counts.values()), report.total)
        self.assertEqual(list(report.tier_counts), ["auto", "escalated", "failed"])


class TestAutoAccuracyIsNotThePassRate(unittest.TestCase):
    """The whole reason the split exists.

    The mixed evalset passes 3 of 4 cases — but one of those passes is a case the pack
    *handed to a human*, and one of the three it answered itself is wrong. "75% passed"
    and "of the 75% it automated, 67% were right" are different claims about deployment,
    and only the second one is a promise the pack can keep unattended.
    """

    def setUp(self):
        self.report = mixed_report()

    def test_the_overall_pass_rate_is_still_reported_unchanged(self):
        self.assertEqual((self.report.passed, self.report.total), (3, 4))

    def test_the_automation_rate_counts_only_unassisted_runs(self):
        self.assertEqual(self.report.automation_rate, 0.75)
        self.assertEqual(self.report.escalation_rate, 0.25)
        self.assertEqual(self.report.failure_rate, 0.0)

    def test_accuracy_is_scored_inside_the_auto_bucket_alone(self):
        self.assertEqual(len(self.report.auto_cases), 3)
        self.assertEqual(self.report.auto_passed, 2)
        self.assertAlmostEqual(self.report.auto_accuracy, 2.0 / 3.0)

    def test_the_two_numbers_are_not_the_same_number(self):
        pass_rate = self.report.passed / float(self.report.total)
        self.assertNotAlmostEqual(pass_rate, self.report.auto_accuracy)

    def test_a_passing_escalated_case_does_not_flatter_the_auto_bucket(self):
        """`rescued: true` passing means the pack routed as promised, not that it answered."""
        rescued = self.report.cases[2]
        self.assertTrue(rescued.passed)
        self.assertNotIn(rescued, self.report.auto_cases)

    def test_accuracy_is_none_not_zero_when_nothing_was_automated(self):
        pack = gated_pack(MIXED_CASES[2:3])
        report = evaluate(pack, FakeModel({"classify": ["not json at all"]}))
        self.assertIsNone(report.auto_accuracy)
        self.assertEqual(report.automation_rate, 0.0)


class TestTheSentenceAClientIsShown(unittest.TestCase):
    """97% handled automatically, 3% to a human, and the automatic 97% is 99.0% correct.

    Built from `CaseResult`s rather than run, so the arithmetic is checked at a scale no
    fixture evalset reaches — and so a report rebuilt outside `evaluate` reports the same
    numbers as one it produced.
    """

    def setUp(self):
        cases = [
            CaseResult(name="auto %d" % n, passed=True, input={}, expected={},
                       tier=TIER_AUTO)
            for n in range(96)
        ]
        cases.append(CaseResult(name="auto wrong", passed=False, input={}, expected={},
                                tier=TIER_AUTO))
        cases.extend(
            CaseResult(name="handed over %d" % n, passed=True, input={}, expected={},
                       tier=TIER_ESCALATED,
                       escalations=[Escalation(node="confirm", kind="unsure")])
            for n in range(3)
        )
        self.report = Report(pack="orders", cases=cases)

    def test_the_three_numbers(self):
        self.assertEqual(self.report.automation_rate, 0.97)
        self.assertEqual(self.report.escalation_rate, 0.03)
        self.assertAlmostEqual(self.report.auto_accuracy, 96 / 97.0)

    def test_the_escalations_name_where_to_put_the_next_assert(self):
        self.assertEqual(self.report.escalated_by, {"confirm": 3})

    def test_the_summary_says_all_three(self):
        rendered = self.report.tier_summary()
        self.assertIn("97 auto, 3 escalated, 0 failed", rendered)
        self.assertIn("97.0%", rendered)
        self.assertIn("99.0%", rendered)      # 96/97, rounded for the terminal
        self.assertIn("(96/97 correct)", rendered)
        self.assertIn("confirm=3", rendered)


class TestTierSummaryText(unittest.TestCase):
    def test_it_names_the_pack_and_every_tier(self):
        rendered = mixed_report().tier_summary()
        self.assertIn("triage: 4 cases", rendered)
        self.assertIn("3 auto, 1 escalated, 0 failed", rendered)
        self.assertIn("accuracy 66.7% (2/3 correct)", rendered)
        self.assertIn("escalated   25.0%   at classify=1", rendered)

    def test_an_empty_auto_bucket_says_so_rather_than_printing_a_number(self):
        pack = gated_pack(MIXED_CASES[2:3])
        rendered = evaluate(pack, FakeModel({"classify": ["not json"]})).tier_summary()
        self.assertIn("n/a", rendered)
        self.assertNotIn("accuracy 0.0%", rendered)

    def test_a_report_with_no_cases_does_not_divide_by_zero(self):
        self.assertEqual(Report(pack="empty").tier_summary(), "empty: no cases")

    def test_nodes_are_listed_worst_first(self):
        report = Report(pack="p", cases=[
            CaseResult(name="a", passed=False, input={}, expected={},
                       tier=TIER_ESCALATED,
                       escalations=[Escalation(node="alpha", kind="unsure")]),
            CaseResult(name="b", passed=False, input={}, expected={},
                       tier=TIER_ESCALATED,
                       escalations=[Escalation(node="omega", kind="unsure")]),
            CaseResult(name="c", passed=False, input={}, expected={},
                       tier=TIER_ESCALATED,
                       escalations=[Escalation(node="omega", kind="unsure")]),
        ])
        self.assertIn("at omega=2, alpha=1", report.tier_summary())

    def test_the_default_summary_is_untouched_by_any_of_this(self):
        """Existing scripts and the README's transcripts parse `summary()`."""
        summary = mixed_report().summary()
        self.assertEqual(summary.splitlines()[0], "triage: 3/4 cases passed")
        for word in ("auto", "escalated", "accuracy", "automation"):
            self.assertNotIn(word, summary)


class TestTheUnsureSignal(unittest.TestCase):
    """The contract between the confidence gate and the report.

    The gate itself lives in the walker. What eval promises is: whatever `RunResult`
    publishes as `unsure`, a case carrying one is not counted as automated, and the node
    that hesitated is named. The walker is stood in for here so that promise is pinned
    down independently of it — including for a runtime that has no gate at all, whose
    `RunResult` has no such field and must score exactly as it always did.
    """

    def _report(self, unsure, failures=(), path=("classify", "extract", "done")):
        result = RunResult(
            run_id="r1",
            output={"category": "billing"},
            state={"category": "billing"},
            path=list(path),
            steps=len(path),
            provenance={"category": "classify"},
            end_node=path[-1],
            failures=list(failures),
        )
        result.unsure = unsure
        pack = two_node_pack()
        pack.evalset[:] = [EvalCase({"ticket": "t1"}, {"category": "billing"}, "one")]
        with mock.patch("jig.eval.run", return_value=result):
            return evaluate(pack, FakeModel(["unused"]))

    def test_an_unsure_node_takes_the_case_out_of_the_auto_bucket(self):
        report = self._report([Escalation(node="extract", kind="unsure")])
        self.assertEqual(report.cases[0].tier, TIER_ESCALATED)
        self.assertIsNone(report.auto_accuracy)
        self.assertEqual(report.escalated_by, {"extract": 1})

    def test_an_unsure_case_can_still_be_correct(self):
        """Correct and unsupervised are different claims; only the second one is the tier."""
        report = self._report([Escalation(node="extract", kind="unsure")])
        self.assertTrue(report.cases[0].passed)
        self.assertEqual(report.passed, 1)

    def test_a_record_can_be_a_mapping(self):
        report = self._report([{"node": "extract", "reason": "two samples disagreed"}])
        self.assertEqual(report.escalated_by, {"extract": 1})
        self.assertEqual(report.cases[0].escalations[0].reason, "two samples disagreed")

    def test_a_record_can_be_a_bare_node_name(self):
        report = self._report(["extract"])
        self.assertEqual(report.escalated_by, {"extract": 1})

    def test_a_record_naming_no_node_still_counts_as_an_escalation(self):
        """Losing the attribution beats quietly counting the case as automated."""
        report = self._report([{"reason": "low agreement"}])
        self.assertEqual(report.cases[0].tier, TIER_ESCALATED)
        self.assertEqual(report.escalated_by, {"<unknown>": 1})

    def test_the_earliest_node_in_the_walk_is_the_one_blamed(self):
        report = self._report(
            [Escalation(node="extract", kind="unsure")],
            failures=[Failure(node="classify", reason="ladder", attempts=3)],
        )
        self.assertEqual(report.cases[0].escalated_by, "classify")
        self.assertEqual(
            [record.kind for record in report.cases[0].escalations],
            ["failed", "unsure"],
        )

    def test_a_runtime_with_no_gate_at_all_is_unaffected(self):
        result = RunResult(
            run_id="r1", output={"category": "billing"}, state={"category": "billing"},
            path=["classify", "done"], steps=2,
            provenance={"category": "classify"}, end_node="done",
        )
        self.assertFalse(hasattr(result, "unsure"))
        pack = two_node_pack()
        pack.evalset[:] = [EvalCase({"ticket": "t1"}, {"category": "billing"}, "one")]
        with mock.patch("jig.eval.run", return_value=result):
            report = evaluate(pack, FakeModel(["unused"]))
        self.assertEqual(report.cases[0].tier, TIER_AUTO)
        self.assertEqual(report.auto_accuracy, 1.0)


# ------------------------------------------------------------------ the CLI surface


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI_PACK = os.path.join(ROOT, "tests", "fixtures", "cli_pack")


def jig_main(*argv):
    """Run the CLI in-process and return (exit code, stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(list(argv))
    return code, out.getvalue()


class TestEvalTiersFlag(unittest.TestCase):
    def test_the_default_output_is_exactly_what_it_was(self):
        """Scripts grep this and the README's transcripts are executed as tests."""
        code, out = jig_main("eval", CLI_PACK)
        self.assertEqual(code, 0)
        self.assertEqual(out, "cli_demo: 2/2 cases passed\n")

    def test_the_flag_adds_the_breakdown_underneath(self):
        code, out = jig_main("eval", CLI_PACK, "--tiers")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "cli_demo: 2/2 cases passed")
        self.assertIn("2 cases — 2 auto, 0 escalated, 0 failed", lines[1])
        self.assertIn("accuracy 100.0% (2/2 correct)", out)

    def test_the_exit_code_still_comes_from_the_cases_not_the_tiers(self):
        """A pack can automate 100% of its cases and still be wrong about them."""
        code, out = jig_main("eval", CLI_PACK, "--model", "fake:fakes/wrong.json",
                             "--tiers")
        self.assertEqual(code, 1)
        self.assertIn("1/2", out)
        self.assertIn("2 auto, 0 escalated, 0 failed", out)
        self.assertIn("accuracy 50.0% (1/2 correct)", out)

    def test_json_carries_the_split_without_being_asked(self):
        import json

        code, out = jig_main("eval", CLI_PACK, "--json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["tiers"]["counts"],
                         {"auto": 2, "escalated": 0, "failed": 0})
        self.assertEqual(report["tiers"]["automation_rate"], 1.0)
        self.assertEqual(report["tiers"]["auto_accuracy"], 1.0)
        self.assertEqual(report["cases"][0]["tier"], "auto")
        self.assertEqual(report["cases"][0]["escalations"], [])

    def test_the_json_keys_that_were_always_there_still_are(self):
        import json

        code, out = jig_main("eval", CLI_PACK, "--model", "fake:fakes/wrong.json",
                             "--json")
        self.assertEqual(code, 1)
        report = json.loads(out)
        self.assertEqual(report["by_node"], {"classify": 1})
        self.assertEqual((report["passed"], report["failed"], report["total"]), (1, 1, 2))

    def test_auto_accuracy_is_null_in_json_rather_than_a_number_nobody_measured(self):
        report = Report(pack="p", cases=[
            CaseResult(name="a", passed=True, input={}, expected={},
                       tier=TIER_ESCALATED,
                       escalations=[Escalation(node="n", kind="unsure")]),
        ])
        self.assertIsNone(cli._report_json(report)["tiers"]["auto_accuracy"])


class TestToolsOption(unittest.TestCase):
    """`--tools` is how a host hands a pack the actions it is allowed to take."""

    MODULE = textwrap.dedent(
        """
        from jig.tools import ToolRegistry

        registry = ToolRegistry()

        @registry.register("ping", writes=["pong"])
        def ping():
            return {"pong": 1}

        def make():
            return registry

        not_a_registry = 42
        """
    )

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "hosttools.py")
        with open(self.path, "w") as handle:
            handle.write(self.MODULE)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.directory, ignore_errors=True)

    def test_a_file_path_yields_the_registry(self):
        registry = cli._load_registry(self.path)
        self.assertEqual(registry.names, ["ping"])

    def test_loading_a_file_does_not_put_its_directory_on_sys_path(self):
        before = list(sys.path)
        cli._load_registry(self.path)
        self.assertEqual(sys.path, before)

    def test_an_attribute_can_be_named_explicitly(self):
        self.assertEqual(cli._load_registry("%s:registry" % self.path).names, ["ping"])

    def test_a_factory_is_called(self):
        """`def registry(): ...` is the natural shape when tools close over a handle."""
        self.assertEqual(cli._load_registry("%s:make" % self.path).names, ["ping"])

    def test_a_module_name_is_importable_too(self):
        sys.path.insert(0, self.directory)
        try:
            self.assertEqual(cli._load_registry("hosttools").names, ["ping"])
        finally:
            sys.path.remove(self.directory)
            sys.modules.pop("hosttools", None)

    def test_a_missing_file_says_which_file(self):
        with self.assertRaises(ValueError) as caught:
            cli._load_registry(os.path.join(self.directory, "ghost.py"))
        self.assertIn("ghost.py", str(caught.exception))

    def test_a_missing_module_points_at_the_fix(self):
        with self.assertRaises(ValueError) as caught:
            cli._load_registry("definitely_not_installed_anywhere")
        self.assertIn("definitely_not_installed_anywhere", str(caught.exception))
        self.assertIn("PYTHONPATH", str(caught.exception))

    def test_a_module_with_no_registry_lists_the_names_it_looked_for(self):
        empty = os.path.join(self.directory, "empty.py")
        with open(empty, "w") as handle:
            handle.write("x = 1\n")
        with self.assertRaises(ValueError) as caught:
            cli._load_registry(empty)
        self.assertIn("registry", str(caught.exception))
        self.assertIn("REGISTRY", str(caught.exception))

    def test_something_that_is_not_a_registry_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            cli._load_registry("%s:not_a_registry" % self.path)
        self.assertIn("int", str(caught.exception))

    def test_no_flag_means_no_registry_and_no_import(self):
        self.assertIsNone(cli._tool_registry(_Args(tools=None), _takes_tools))

    def test_a_runtime_that_cannot_run_tools_says_so_instead_of_a_keyword_error(self):
        with self.assertRaises(ValueError) as caught:
            cli._tool_registry(_Args(tools=self.path), _takes_nothing)
        message = str(caught.exception)
        self.assertIn("cannot run tools", message)
        self.assertIn("_takes_nothing", message)

    def test_a_runtime_that_can_run_tools_gets_the_registry(self):
        registry = cli._tool_registry(_Args(tools=self.path), _takes_tools)
        self.assertEqual(registry.names, ["ping"])


class _Args:
    """The one attribute `_tool_registry` reads, without building a parser."""

    def __init__(self, tools=None):
        self.tools = tools


def _takes_tools(pack, model, tools=None):
    pass


def _takes_nothing(pack, model):
    pass


class TestEvaluateForwardsTools(unittest.TestCase):
    """A pack with no tools must reach the walker exactly as it always did."""

    def _run_with(self, **kwargs):
        seen = {}

        def fake_run(pack, model, inputs, **called):
            seen.update(called)
            return RunResult(
                run_id="r1", output={"category": "billing"},
                state={"category": "billing"}, path=["classify", "done"], steps=2,
                provenance={"category": "classify"}, end_node="done",
            )

        pack = two_node_pack()
        pack.evalset[:] = [EvalCase({"ticket": "t1"}, {"category": "billing"}, "one")]
        with mock.patch("jig.eval.run", fake_run):
            evaluate(pack, FakeModel(["unused"]), **kwargs)
        return seen

    def test_no_tools_means_no_tools_keyword(self):
        self.assertNotIn("tools", self._run_with())

    def test_a_registry_is_handed_to_the_walker(self):
        sentinel = object()
        self.assertIs(self._run_with(tools=sentinel)["tools"], sentinel)
