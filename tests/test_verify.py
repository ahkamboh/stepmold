"""T6 — nothing enters state until it has been verified, and the retry ladder."""

import unittest

from jig.errors import MissingVariable, NodeFailed
from jig.graph import run
from jig.model import FakeModel
from jig.pack import Edge, Node, Pack
from jig.verify import Rejected, extract_json, run_node, verify

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": ["billing", "technical"]}},
    "required": ["category"],
    "additionalProperties": False,
}

GOOD = '{"category": "billing"}'
BAD_ENUM = '{"category": "refund"}'
BAD_JSON = "sorry, I cannot do that"


def node(name="classify", **kwargs):
    options = {
        "type": "generate",
        "prompt": "Classify: {ticket}",
        "grammar": SCHEMA,
    }
    options.update(kwargs)
    return Node(name=name, **options)


def pack_of(*nodes):
    ends = {"done": Node(name="done", type="end"), "give_up": Node(name="give_up", type="end")}
    table = {item.name: item for item in nodes}
    table.update(ends)
    edges = [Edge(nodes[0].name, "done", None)]
    return Pack(
        path="<memory>",
        name="t",
        version=1,
        entry=nodes[0].name,
        model=None,
        nodes=table,
        edges=edges,
    )


class TestVerify(unittest.TestCase):
    def test_a_good_output_comes_back_parsed(self):
        self.assertEqual(verify(node(), GOOD, {}), {"category": "billing"})

    def test_a_schema_violation_is_rejected_with_the_field_name(self):
        with self.assertRaises(Rejected) as caught:
            verify(node(), BAD_ENUM, {})
        self.assertIn("category", str(caught.exception))

    def test_unparseable_output_is_rejected(self):
        with self.assertRaises(Rejected) as caught:
            verify(node(), BAD_JSON, {})
        self.assertIn("JSON", str(caught.exception))

    def test_a_non_object_output_is_rejected(self):
        with self.assertRaises(Rejected):
            verify(node(), '["billing"]', {})

    def test_a_node_assert_runs_against_the_candidate(self):
        checked = node(assert_expr='category == "billing"')
        self.assertEqual(verify(checked, GOOD, {}), {"category": "billing"})
        with self.assertRaises(Rejected) as caught:
            verify(checked, '{"category": "technical"}', {})
        self.assertIn("assert", str(caught.exception))

    def test_a_node_assert_can_see_existing_state(self):
        checked = node(assert_expr="category == expected")
        self.assertEqual(
            verify(checked, GOOD, {"expected": "billing"}), {"category": "billing"}
        )

    def test_a_node_assert_sees_the_candidate_under_its_output_key(self):
        checked = node(output="c", assert_expr='c.category == "billing"')
        self.assertEqual(verify(checked, GOOD, {}), {"category": "billing"})


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_bare_fence(self):
        self.assertEqual(extract_json('```\n{"a": 1}\n```'), {"a": 1})

    def test_json_wrapped_in_prose(self):
        self.assertEqual(extract_json('Sure! {"a": 1} hope that helps'), {"a": 1})

    def test_braces_inside_strings_do_not_confuse_the_scan(self):
        self.assertEqual(extract_json('x {"a": "} not the end"} y'), {"a": "} not the end"})

    def test_nothing_json_shaped_raises(self):
        with self.assertRaises(Rejected):
            extract_json("no json here")


class TestRetryLadder(unittest.TestCase):
    def test_passes_first_try_with_one_call(self):
        model = FakeModel([GOOD])
        self.assertEqual(run_node(node(), {"ticket": "t"}, model), {"category": "billing"})
        self.assertEqual(model.call_count, 1)

    def test_recovers_on_the_second_attempt(self):
        model = FakeModel([BAD_ENUM, GOOD])
        self.assertEqual(run_node(node(), {"ticket": "t"}, model), {"category": "billing"})
        self.assertEqual(model.call_count, 2)

    def test_the_first_resample_differs_from_the_request_that_was_rejected(self):
        """A byte-identical re-sample is a wasted rung against a greedy backend."""
        model = FakeModel([BAD_ENUM, GOOD])
        run_node(node(), {"ticket": "t"}, model)
        self.assertNotEqual(model.calls[1].prompt, model.calls[0].prompt)
        self.assertIn("rejected", model.calls[1].prompt)

    def test_the_first_resample_still_does_not_quote_the_rejected_value(self):
        model = FakeModel([BAD_ENUM, GOOD])
        run_node(node(), {"ticket": "t"}, model)
        self.assertNotIn("refund", model.calls[1].prompt)

    def test_a_prompt_that_cannot_be_rendered_spends_no_generations(self):
        model = FakeModel([GOOD])
        with self.assertRaises(MissingVariable):
            run_node(node(prompt="Classify: {nobody_wrote_this}"), {"ticket": "t"}, model)
        self.assertEqual(model.call_count, 0)

    def test_the_second_resample_appends_the_rejection(self):
        model = FakeModel([BAD_ENUM, BAD_ENUM, GOOD])
        run_node(node(), {"ticket": "t"}, model)
        self.assertIn("rejected", model.calls[2].prompt)
        self.assertIn("category", model.calls[2].prompt)

    def test_exhausting_the_ladder_raises_node_failed(self):
        model = FakeModel([BAD_ENUM, BAD_ENUM, BAD_ENUM])
        with self.assertRaises(NodeFailed) as caught:
            run_node(node(), {"ticket": "t"}, model)
        self.assertEqual(caught.exception.node, "classify")
        self.assertEqual(caught.exception.attempts, 3)
        self.assertIn("category", str(caught.exception))

    def test_retries_zero_means_a_single_attempt(self):
        model = FakeModel([BAD_ENUM, GOOD])
        with self.assertRaises(NodeFailed):
            run_node(node(retries=0), {"ticket": "t"}, model)
        self.assertEqual(model.call_count, 1)

    def test_a_two_stage_node_thinks_once_and_re_emits(self):
        model = FakeModel(["notes", BAD_ENUM, GOOD])
        run_node(node(two_stage=True), {"ticket": "t"}, model)
        self.assertEqual(model.call_count, 3)
        self.assertIsNone(model.calls[0].grammar)
        self.assertIsNotNone(model.calls[1].grammar)
        self.assertIn("notes", model.calls[2].prompt)


class TestLadderInsideAWalk(unittest.TestCase):
    def test_a_recovered_node_leaves_no_trace_of_the_rejected_output(self):
        model = FakeModel([BAD_ENUM, GOOD])
        result = run(pack_of(node()), model, {"ticket": "t"})
        self.assertEqual(result.state, {"ticket": "t", "category": "billing"})
        self.assertNotIn("refund", repr(result.state))
        self.assertNotIn("refund", repr(result.output))

    def test_a_rejected_output_never_reaches_a_later_prompt(self):
        first = node("classify")
        second = node("extract", prompt="Given {category}, extract from {ticket}")
        pack = Pack(
            path="<memory>",
            name="t",
            version=1,
            entry="classify",
            model=None,
            nodes={
                "classify": first,
                "extract": second,
                "done": Node(name="done", type="end"),
            },
            edges=[Edge("classify", "extract", None), Edge("extract", "done", None)],
        )
        model = FakeModel([BAD_ENUM, GOOD, '{"category": "technical"}'])
        run(pack, model, {"ticket": "t"})
        self.assertNotIn("refund", model.calls[2].prompt)

    def test_exhausting_the_ladder_follows_the_on_fail_node(self):
        model = FakeModel([BAD_ENUM, BAD_ENUM, BAD_ENUM])
        result = run(pack_of(node(on_fail="give_up")), model, {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "give_up"])
        self.assertEqual(result.state, {"ticket": "t"})

    def test_a_diverted_failure_is_recorded_on_the_result(self):
        model = FakeModel([BAD_ENUM, BAD_ENUM, BAD_ENUM])
        result = run(pack_of(node(on_fail="give_up")), model, {"ticket": "t"})
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].node, "classify")
        self.assertEqual(result.failures[0].attempts, 3)
        self.assertIn("category", result.failures[0].reason)

    def test_without_on_fail_the_run_raises(self):
        model = FakeModel([BAD_ENUM, BAD_ENUM, BAD_ENUM])
        with self.assertRaises(NodeFailed):
            run(pack_of(node()), model, {"ticket": "t"})

    def test_a_successful_run_records_no_failures(self):
        result = run(pack_of(node()), FakeModel([GOOD]), {"ticket": "t"})
        self.assertEqual(result.failures, [])
