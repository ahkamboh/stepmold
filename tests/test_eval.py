"""T8 — scoring a pack against its evalset, and blaming the right node."""

import unittest

from jig.eval import evaluate
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
