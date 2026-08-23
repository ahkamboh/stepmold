"""T4 — walking a graph from its entry node with a scripted model."""

import unittest

from jig.errors import (
    AssertFailed,
    DanglingEdge,
    DeadEnd,
    MaxStepsExceeded,
    MissingVariable,
    NodeFailed,
)
from jig.model import FakeModel
from jig.pack import Edge, Node, Pack
from jig.graph import run


def build(nodes, edges, entry, max_steps=100):
    """A Pack assembled in memory, so walker tests need no files on disk."""
    return Pack(
        path="<memory>",
        name="test",
        version=1,
        entry=entry,
        model=None,
        nodes={node.name: node for node in nodes},
        edges=[Edge(*edge) if isinstance(edge, tuple) else edge for edge in edges],
        max_steps=max_steps,
    )


def generate(name, prompt="go", schema=None, **kwargs):
    return Node(
        name=name,
        type="generate",
        prompt=prompt,
        grammar=schema if schema is not None else {"type": "object"},
        **kwargs
    )


class TestLinearPath(unittest.TestCase):
    def setUp(self):
        self.pack = build(
            nodes=[
                generate("classify", "Classify: {ticket}"),
                generate("extract", "Category is {category}. Extract from {ticket}."),
                Node(name="done", type="end"),
            ],
            edges=[("classify", "extract"), ("extract", "done")],
            entry="classify",
        )

    def test_walks_every_node_in_order(self):
        model = FakeModel(['{"category": "billing"}', '{"amount": 42}'])
        result = run(self.pack, model, {"ticket": "charged twice"})
        self.assertEqual(result.path, ["classify", "extract", "done"])

    def test_state_accumulates_across_nodes(self):
        model = FakeModel(['{"category": "billing"}', '{"amount": 42}'])
        result = run(self.pack, model, {"ticket": "charged twice"})
        self.assertEqual(
            result.state,
            {"ticket": "charged twice", "category": "billing", "amount": 42},
        )

    def test_prompts_render_from_state_written_by_earlier_nodes(self):
        model = FakeModel(['{"category": "billing"}', '{"amount": 42}'])
        run(self.pack, model, {"ticket": "charged twice"})
        self.assertEqual(model.calls[0].prompt, "Classify: charged twice")
        self.assertEqual(
            model.calls[1].prompt, "Category is billing. Extract from charged twice."
        )

    def test_the_node_grammar_is_handed_to_the_model(self):
        model = FakeModel(['{"category": "billing"}', '{"amount": 42}'])
        run(self.pack, model, {"ticket": "t"})
        self.assertEqual(model.calls[0].grammar["kind"], "json_schema")
        self.assertEqual(model.calls[0].grammar["schema"], {"type": "object"})

    def test_provenance_records_which_node_wrote_each_key(self):
        model = FakeModel(['{"category": "billing"}', '{"amount": 42}'])
        result = run(self.pack, model, {"ticket": "t"})
        self.assertEqual(result.provenance["category"], "classify")
        self.assertEqual(result.provenance["amount"], "extract")
        self.assertNotIn("ticket", result.provenance)

    def test_a_prompt_needing_a_variable_nobody_wrote_fails_loudly(self):
        model = FakeModel(['{"category": "billing"}'])
        with self.assertRaises(MissingVariable):
            run(self.pack, model, {})


class TestOutputPlacement(unittest.TestCase):
    def test_output_key_nests_the_emitted_object(self):
        pack = build(
            nodes=[
                generate("classify", output="classification"),
                Node(name="done", type="end"),
            ],
            edges=[("classify", "done")],
            entry="classify",
        )
        result = run(pack, FakeModel(['{"category": "billing"}']))
        self.assertEqual(result.state["classification"], {"category": "billing"})

    def test_end_node_output_projects_only_the_named_keys(self):
        pack = build(
            nodes=[
                generate("classify"),
                Node(name="done", type="end", output=["category"]),
            ],
            edges=[("classify", "done")],
            entry="classify",
        )
        result = run(pack, FakeModel(['{"category": "billing", "scratch": 1}']))
        self.assertEqual(result.output, {"category": "billing"})
        self.assertIn("scratch", result.state)

    def test_without_a_projection_the_output_is_the_whole_state(self):
        pack = build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )
        result = run(pack, FakeModel(['{"category": "billing"}']), {"ticket": "t"})
        self.assertEqual(result.output, {"ticket": "t", "category": "billing"})

    def test_end_node_records_itself(self):
        pack = build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )
        result = run(pack, FakeModel(['{"a": 1}']))
        self.assertEqual(result.end_node, "done")

    def test_a_non_object_generation_is_rejected_after_the_ladder(self):
        pack = build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )
        model = FakeModel(["not json at all"] * 3)
        with self.assertRaises(NodeFailed):
            run(pack, model)
        self.assertEqual(model.call_count, 3)


class TestConditionalEdges(unittest.TestCase):
    def _pack(self):
        return build(
            nodes=[
                generate("classify"),
                Node(name="refund", type="end", output=["route"]),
                Node(name="support", type="end", output=["route"]),
            ],
            edges=[
                Edge("classify", "refund", {"category": "billing"}),
                Edge("classify", "support", None),
            ],
            entry="classify",
        )

    def test_a_matching_condition_takes_its_edge(self):
        result = run(self._pack(), FakeModel(['{"category": "billing", "route": "r"}']))
        self.assertEqual(result.path[-1], "refund")

    def test_a_non_matching_condition_falls_through_to_the_default_edge(self):
        result = run(
            self._pack(), FakeModel(['{"category": "technical", "route": "s"}'])
        )
        self.assertEqual(result.path[-1], "support")

    def test_conditions_may_test_several_keys_and_dotted_paths(self):
        pack = build(
            nodes=[
                generate("classify", output="c"),
                Node(name="hit", type="end"),
                Node(name="miss", type="end"),
            ],
            edges=[
                Edge("classify", "hit", {"c.category": "billing", "c.urgent": True}),
                Edge("classify", "miss", None),
            ],
            entry="classify",
        )
        hit = run(pack, FakeModel(['{"category": "billing", "urgent": true}']))
        self.assertEqual(hit.path[-1], "hit")
        miss = run(pack, FakeModel(['{"category": "billing", "urgent": false}']))
        self.assertEqual(miss.path[-1], "miss")

    def test_no_matching_edge_is_a_dead_end(self):
        pack = build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[Edge("classify", "done", {"category": "refund"})],
            entry="classify",
        )
        with self.assertRaises(DeadEnd) as caught:
            run(pack, FakeModel(['{"category": "billing"}']))
        self.assertIn("classify", str(caught.exception))

    def test_a_condition_on_a_key_nobody_wrote_simply_does_not_match(self):
        pack = build(
            nodes=[
                generate("classify"),
                Node(name="a", type="end"),
                Node(name="b", type="end"),
            ],
            edges=[Edge("classify", "a", {"nothing": 1}), Edge("classify", "b", None)],
            entry="classify",
        )
        self.assertEqual(run(pack, FakeModel(['{"x": 1}'])).path[-1], "b")


class TestAssertNodes(unittest.TestCase):
    def _pack(self, expr, on_fail=None):
        return build(
            nodes=[
                generate("classify"),
                Node(name="check", type="assert", expr=expr, on_fail=on_fail),
                Node(name="done", type="end"),
                Node(name="give_up", type="end"),
            ],
            edges=[("classify", "check"), ("check", "done")],
            entry="classify",
        )

    def test_a_true_assert_carries_on(self):
        pack = self._pack('category == "billing"')
        result = run(pack, FakeModel(['{"category": "billing"}']))
        self.assertEqual(result.path, ["classify", "check", "done"])

    def test_an_assert_node_calls_no_model(self):
        model = FakeModel(['{"category": "billing"}'])
        run(self._pack("true"), model)
        self.assertEqual(model.call_count, 1)

    def test_a_false_assert_without_on_fail_raises(self):
        pack = self._pack('category == "refund"')
        with self.assertRaises(AssertFailed) as caught:
            run(pack, FakeModel(['{"category": "billing"}']))
        self.assertIn("check", str(caught.exception))

    def test_a_false_assert_with_on_fail_jumps_there(self):
        pack = self._pack('category == "refund"', on_fail="give_up")
        result = run(pack, FakeModel(['{"category": "billing"}']))
        self.assertEqual(result.path, ["classify", "check", "give_up"])


class TestLoops(unittest.TestCase):
    def _pack(self, max_steps=100):
        return build(
            nodes=[generate("tick"), Node(name="done", type="end", output=["count"])],
            edges=[Edge("tick", "done", {"finished": True}), Edge("tick", "tick", None)],
            entry="tick",
            max_steps=max_steps,
        )

    def test_a_loop_that_meets_its_condition_terminates(self):
        model = FakeModel(
            [
                '{"count": 1, "finished": false}',
                '{"count": 2, "finished": false}',
                '{"count": 3, "finished": true}',
            ]
        )
        result = run(self._pack(), model)
        self.assertEqual(result.path, ["tick", "tick", "tick", "done"])
        self.assertEqual(result.output, {"count": 3})

    def test_an_unbroken_loop_hits_the_step_guard(self):
        model = FakeModel({"go": '{"count": 1, "finished": false}'})
        with self.assertRaises(MaxStepsExceeded) as caught:
            run(self._pack(max_steps=5), model)
        self.assertIn("5", str(caught.exception))
        self.assertIn("tick", str(caught.exception))

    def test_the_step_guard_can_be_overridden_per_run(self):
        model = FakeModel({"go": '{"count": 1, "finished": false}'})
        with self.assertRaises(MaxStepsExceeded):
            run(self._pack(max_steps=1000), model, max_steps=3)
        self.assertEqual(model.call_count, 3)


class TestBrokenGraphs(unittest.TestCase):
    def test_an_edge_to_a_node_that_does_not_exist(self):
        pack = build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[Edge("classify", "ghost", None)],
            entry="classify",
        )
        with self.assertRaises(DanglingEdge) as caught:
            run(pack, FakeModel(['{"a": 1}']))
        self.assertIn("ghost", str(caught.exception))

    def test_an_entry_node_that_does_not_exist(self):
        pack = build(
            nodes=[Node(name="done", type="end")], edges=[], entry="ghost"
        )
        with self.assertRaises(DanglingEdge) as caught:
            run(pack, FakeModel(['{"a": 1}']))
        self.assertIn("ghost", str(caught.exception))

    def test_an_on_fail_pointing_nowhere(self):
        pack = build(
            nodes=[
                generate("classify"),
                Node(name="check", type="assert", expr="false", on_fail="ghost"),
                Node(name="done", type="end"),
            ],
            edges=[("classify", "check"), ("check", "done")],
            entry="classify",
        )
        with self.assertRaises(DanglingEdge):
            run(pack, FakeModel(['{"a": 1}']))


class TestRunMetadata(unittest.TestCase):
    def _pack(self):
        return build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )

    def test_a_run_id_is_generated_when_not_supplied(self):
        result = run(self._pack(), FakeModel(['{"a": 1}']))
        self.assertTrue(result.run_id)

    def test_a_supplied_run_id_is_kept(self):
        result = run(self._pack(), FakeModel(['{"a": 1}']), run_id="fixed")
        self.assertEqual(result.run_id, "fixed")

    def test_two_runs_get_different_ids(self):
        first = run(self._pack(), FakeModel(['{"a": 1}']))
        second = run(self._pack(), FakeModel(['{"a": 1}']))
        self.assertNotEqual(first.run_id, second.run_id)

    def test_steps_counts_executed_nodes(self):
        result = run(self._pack(), FakeModel(['{"a": 1}']))
        self.assertEqual(result.steps, 2)

    def test_inputs_are_not_mutated(self):
        inputs = {"ticket": "t"}
        run(self._pack(), FakeModel(['{"a": 1}']), inputs)
        self.assertEqual(inputs, {"ticket": "t"})
