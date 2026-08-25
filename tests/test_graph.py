"""T4 — walking a graph from its entry node with a scripted model.

Later sections cover what the walker gained after T4: `tool` nodes, the exactly-once
record that stops a side effect happening twice across a crash, and where an `Unsure`
generation is routed.
"""

import dataclasses
import unittest
from typing import Optional
from unittest import mock

from jig.errors import (
    AssertFailed,
    DanglingEdge,
    DeadEnd,
    ExprError,
    MaxStepsExceeded,
    MissingVariable,
    NodeFailed,
)
from jig.model import FakeModel
from jig.pack import Edge, Node, Pack
from jig.state import Store, resume
from jig.tools import (
    ToolContract,
    ToolFailed,
    ToolNotRegistered,
    ToolRegistry,
)
from jig.graph import (
    StateCollision,
    ToolReplayMismatch,
    ToolsNotAvailable,
    Unsure,
    commit,
    replay,
    run,
)


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


class TestUnrenderablePrompts(unittest.TestCase):
    """A prompt that cannot be rendered is a node failure like any other."""

    def _pack(self, on_fail=None):
        return build(
            nodes=[
                generate("classify", "Classify: {nobody_wrote_this}", on_fail=on_fail),
                Node(name="done", type="end"),
                Node(name="give_up", type="end"),
            ],
            edges=[("classify", "done")],
            entry="classify",
        )

    def test_an_unrenderable_prompt_takes_the_on_fail_edge(self):
        model = FakeModel(['{"a": 1}'])
        result = run(self._pack(on_fail="give_up"), model, {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "give_up"])

    def test_no_generation_is_spent_on_a_prompt_that_never_rendered(self):
        model = FakeModel(['{"a": 1}'])
        run(self._pack(on_fail="give_up"), model, {"ticket": "t"})
        self.assertEqual(model.call_count, 0)

    def test_the_diversion_is_recorded_with_zero_attempts(self):
        result = run(self._pack(on_fail="give_up"), FakeModel(['{"a": 1}']), {})
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].node, "classify")
        self.assertEqual(result.failures[0].attempts, 0)
        self.assertIn("nobody_wrote_this", result.failures[0].reason)

    def test_without_on_fail_the_missing_variable_still_escapes(self):
        with self.assertRaises(MissingVariable):
            run(self._pack(), FakeModel(['{"a": 1}']), {})


class TestUnevaluableAsserts(unittest.TestCase):
    """graph and verify must route an unevaluable expression the same way."""

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

    def test_an_unevaluable_assert_takes_the_on_fail_edge(self):
        pack = self._pack('nobody_wrote_this == "x"', on_fail="give_up")
        result = run(pack, FakeModel(['{"category": "billing"}']))
        self.assertEqual(result.path, ["classify", "check", "give_up"])

    def test_an_unevaluable_assert_without_on_fail_names_what_is_missing(self):
        pack = self._pack('nobody_wrote_this == "x"')
        with self.assertRaises(ExprError) as caught:
            run(pack, FakeModel(['{"category": "billing"}']))
        self.assertIn("nobody_wrote_this", str(caught.exception))

    def test_a_verify_time_assert_routes_the_same_way(self):
        """The same expression on a generate node already diverts — stay consistent."""
        pack = build(
            nodes=[
                generate("classify", assert_expr='nobody_wrote_this == "x"',
                         on_fail="give_up"),
                Node(name="done", type="end"),
                Node(name="give_up", type="end"),
            ],
            edges=[("classify", "done")],
            entry="classify",
        )
        result = run(pack, FakeModel(['{"a": 1}'] * 3))
        self.assertEqual(result.path, ["classify", "give_up"])


class TestCheckpointOnDeadEnd(unittest.TestCase):
    """A committed node must leave a record even when routing fails after it."""

    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.pack = build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[Edge("classify", "done", {"category": "refund"})],
            entry="classify",
        )

    def test_the_committed_node_is_checkpointed_before_the_dead_end_escapes(self):
        with self.assertRaises(DeadEnd):
            run(self.pack, FakeModel(['{"category": "billing"}']), {"t": 1},
                run_id="r", store=self.store)
        history = self.store.history("r")
        self.assertEqual([c.node for c in history], ["classify"])
        self.assertEqual(history[0].state["category"], "billing")

    def test_that_checkpoint_does_not_claim_the_run_finished(self):
        with self.assertRaises(DeadEnd):
            run(self.pack, FakeModel(['{"category": "billing"}']), {"t": 1},
                run_id="r", store=self.store)
        self.assertFalse(self.store.latest("r").finished)


class TestCommitCollisions(unittest.TestCase):
    """Merge-mode commit must not let one node quietly rewrite someone else's value."""

    def _two_nodes(self, second_schema, on_fail=None):
        return build(
            nodes=[
                generate("classify"),
                generate("extract", schema=second_schema),
                Node(name="done", type="end"),
            ],
            edges=[("classify", "extract"), ("extract", "done")],
            entry="classify",
        )

    def test_a_node_may_not_overwrite_a_run_input(self):
        pack = build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )
        with self.assertRaises(StateCollision) as caught:
            run(pack, FakeModel(['{"ticket": "rewritten"}']), {"ticket": "original"})
        self.assertIn("ticket", str(caught.exception))
        self.assertIn("classify", str(caught.exception))

    def test_a_later_node_restating_a_field_is_recorded_rather_than_refused(self):
        """Node over node is the author's own doing, and provenance names the winner."""
        store = Store(":memory:")
        self.addCleanup(store.close)
        pack = self._two_nodes({"type": "object"})
        result = run(pack, FakeModel(['{"category": "billing"}', '{"category": "tech"}']),
                     run_id="r", store=store)
        self.assertEqual(result.state["category"], "tech")
        self.assertEqual(result.provenance["category"], "extract")
        # The earlier value is not lost: it is still in the earlier node's checkpoint.
        self.assertEqual(store.history("r")[0].state["category"], "billing")

    def test_an_output_key_may_not_land_on_a_run_input_either(self):
        pack = build(
            nodes=[generate("classify", output="ticket"), Node(name="done", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )
        with self.assertRaises(StateCollision):
            run(pack, FakeModel(['{"a": 1}']), {"ticket": "original"})

    def test_a_node_may_overwrite_the_value_it_wrote_itself(self):
        pack = build(
            nodes=[generate("tick"), Node(name="done", type="end", output=["count"])],
            edges=[Edge("tick", "done", {"finished": True}), Edge("tick", "tick", None)],
            entry="tick",
        )
        model = FakeModel(
            ['{"count": 1, "finished": false}', '{"count": 2, "finished": true}']
        )
        self.assertEqual(run(pack, model).output, {"count": 2})

    def test_a_refused_commit_writes_nothing_at_all(self):
        state = {"ticket": "original"}
        provenance = {}
        with self.assertRaises(StateCollision):
            commit(generate("classify"), {"category": "billing", "ticket": "new"},
                   state, provenance)
        self.assertEqual(state, {"ticket": "original"})
        self.assertEqual(provenance, {})


ENUM = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": ["billing", "technical"]}},
    "required": ["category"],
    "additionalProperties": False,
}


class TestAttemptsAreOnTheResult(unittest.TestCase):
    """What a finished run says about how close it came (README problem 4).

    `failures` only ever holds the nodes that ran out of ladder. A node that was
    rejected twice and got there on the third rung used to leave no trace at all.
    """

    def _pack(self, **kwargs):
        return build(
            nodes=[generate("classify", schema=ENUM, **kwargs),
                   Node(name="done", type="end"), Node(name="bail", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )

    def test_every_generate_node_reports_what_it_spent(self):
        model = FakeModel(['{"category": "nope"}', '{"category": "billing"}'])
        result = run(self._pack(retries=2), model, {"ticket": "t"})
        self.assertEqual(result.attempts, {"classify": 2})
        self.assertEqual(result.failures, [])

    def test_a_clean_run_reports_one_attempt_per_node(self):
        result = run(self._pack(), FakeModel(['{"category": "billing"}']), {"ticket": "t"})
        self.assertEqual(result.attempts, {"classify": 1})

    def test_a_diverted_node_reports_the_whole_ladder_it_spent(self):
        model = FakeModel(['{"category": "nope"}'] * 3)
        result = run(self._pack(retries=2, on_fail="bail"), model, {"ticket": "t"})
        self.assertEqual(result.attempts, {"classify": 3})
        self.assertEqual(result.failures[0].attempts, 3)

    def test_an_unrenderable_prompt_spends_nothing_and_says_so(self):
        pack = build(
            nodes=[generate("classify", prompt="Classify: {nobody_wrote_this}",
                            on_fail="bail"),
                   Node(name="done", type="end"), Node(name="bail", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )
        result = run(pack, FakeModel(['{"a": 1}']), {"ticket": "t"})
        self.assertEqual(result.attempts, {})

    def test_a_node_visited_twice_counts_both_visits(self):
        """A loop's counter node is one node with two visits, and cost is what is asked."""
        pack = build(
            nodes=[generate("tick"), Node(name="done", type="end", output=["count"])],
            edges=[Edge("tick", "done", {"finished": True}), Edge("tick", "tick", None)],
            entry="tick",
        )
        model = FakeModel(
            ['{"count": 1, "finished": false}', '{"count": 2, "finished": true}']
        )
        self.assertEqual(run(pack, model).attempts, {"tick": 2})

    def test_a_store_that_can_record_them_is_offered_them(self):
        seen = []

        class RecordingStore:
            def save(self, run_id, step, node, next_node, state, path=None,
                     provenance=None, failures=None, output=None, pack=None,
                     pack_version=None, attempts=None):
                seen.append(attempts)

        model = FakeModel(['{"category": "nope"}', '{"category": "billing"}'])
        run(self._pack(retries=2), model, {"ticket": "t"}, run_id="r",
            store=RecordingStore())
        self.assertEqual(seen[-1], {"classify": 2})

    def test_a_store_that_predates_the_field_is_not_broken_by_it(self):
        """`store` is documented as anything with a `save(...)`, so the counts are
        offered rather than imposed."""
        saved = []

        class OlderStore:
            def save(self, run_id, step, node, next_node, state, path=None,
                     provenance=None, failures=None, output=None, pack=None,
                     pack_version=None):
                saved.append(node)

        run(self._pack(), FakeModel(['{"category": "billing"}']), {"ticket": "t"},
            run_id="r", store=OlderStore())
        self.assertEqual(saved, ["classify", "done"])

    def test_a_real_store_still_checkpoints_every_node(self):
        store = Store(":memory:")
        self.addCleanup(store.close)
        model = FakeModel(['{"category": "nope"}', '{"category": "billing"}'])
        result = run(self._pack(retries=2), model, {"ticket": "t"}, run_id="r",
                     store=store)
        self.assertEqual(len(store.history("r")), 2)
        self.assertEqual(result.attempts, {"classify": 2})

    def test_replay_restores_the_counts_a_checkpoint_carries(self):
        store = Store(":memory:")
        self.addCleanup(store.close)
        run(self._pack(), FakeModel(['{"category": "billing"}']), {"ticket": "t"},
            run_id="r", store=store)
        checkpoint = store.latest("r")
        # A store that has nowhere to keep them replays a run with none, rather than
        # inventing numbers it was never given.
        self.assertEqual(replay(checkpoint).attempts,
                         dict(getattr(checkpoint, "attempts", None) or {}))


# --------------------------------------------------------------------------- tools


@dataclasses.dataclass(frozen=True)
class _Node(Node):
    """A node carrying the keys the walker reads and `jig.pack` may not parse yet.

    `jig/pack.py` owns turning `tool:` and `on_unsure:` in graph.yaml into fields on
    `Node`; this file owns what the walker does when it meets them. The two land
    independently, so these tests supply the fields themselves rather than waiting: the
    day `Node` carries them, this subclass is an empty wrapper and nothing here changes.
    """

    tool: Optional[str] = None
    on_unsure: Optional[str] = None


def tool_node(name, tool, **kwargs):
    return _Node(name=name, type="tool", tool=tool, **kwargs)


def no_model():
    """A model for a walk that must never reach a `generate` node.

    Scripted rather than absent because `FakeModel` refuses an empty script — and a
    scripted response nothing ever consumes is the sharper assertion anyway: a tool node
    that reached for the model would be answered, and every test here that counts calls
    or side effects would still catch it.
    """
    return FakeModel(['{"never": "used"}'])


class Crash(Exception):
    """Not a `JigError`: the worker died, it did not fail. Nothing catches this."""


class DyingStore(Store):
    """A real store whose worker is killed at a save of the caller's choosing.

    A checkpointed run cannot be interrupted convincingly from the outside — the
    interesting window is the few microseconds between a tool returning and the walk
    leaving its node, and only the store is called inside it. So the kill is staged from
    there: everything written before the chosen save is on disk exactly as a real crash
    would have left it.
    """

    def __init__(self):
        Store.__init__(self, ":memory:")
        self.die_on = None
        self.saved = []

    def save(self, **kwargs):
        if self.die_on is not None and self.die_on(kwargs):
            raise Crash("worker killed at node %r" % kwargs.get("node"))
        self.saved.append((kwargs.get("node"), kwargs.get("next_node")))
        return Store.save(self, **kwargs)


def sending_registry(sent, idempotent=False, fail=False, writes=("receipt",)):
    """A registry whose one tool has a side effect you can count afterwards."""
    registry = ToolRegistry()

    @registry.register("send_email", reads=["to"], writes=list(writes),
                       idempotent=idempotent)
    def send_email(to):
        if fail:
            raise RuntimeError("smtp is down")
        sent.append(to)
        return {"receipt": "sent-%d" % len(sent)}

    return registry


def sending_pack(**node_kwargs):
    """start -> send (tool) -> done, with a first node so a crash at `send` has a past."""
    return build(
        nodes=[
            Node(name="start", type="assert", expr="true"),
            tool_node("send", "send_email", **node_kwargs),
            Node(name="done", type="end"),
            Node(name="desk", type="end"),
        ],
        edges=[("start", "send"), ("send", "done")],
        entry="start",
    )


class TestToolNodesExecute(unittest.TestCase):
    """A `tool` node calls one registered action and commits what it returns."""

    def setUp(self):
        self.sent = []
        self.registry = sending_registry(self.sent)

    def test_the_tool_is_called_with_the_state_it_declared_it_reads(self):
        run(sending_pack(), no_model(), {"to": "ops@example.com"},
            tools=self.registry)
        self.assertEqual(self.sent, ["ops@example.com"])

    def test_what_it_returns_is_committed_like_a_generation(self):
        result = run(sending_pack(), no_model(), {"to": "ops@example.com"},
                     tools=self.registry)
        self.assertEqual(result.state["receipt"], "sent-1")
        self.assertEqual(result.path, ["start", "send", "done"])

    def test_an_output_key_nests_it_exactly_as_it_nests_a_generation(self):
        result = run(sending_pack(output="delivery"), no_model(),
                     {"to": "ops@example.com"}, tools=self.registry)
        self.assertEqual(result.state["delivery"], {"receipt": "sent-1"})

    def test_provenance_names_the_tool_node_that_wrote_the_field(self):
        result = run(sending_pack(), no_model(), {"to": "ops@example.com"},
                     tools=self.registry)
        self.assertEqual(result.provenance["receipt"], "send")

    def test_a_tool_node_spends_no_generations(self):
        result = run(sending_pack(), no_model(), {"to": "ops@example.com"},
                     tools=self.registry)
        self.assertEqual(result.attempts, {})

    def test_a_tool_node_never_reaches_for_the_model(self):
        """No prompt, no grammar, no ladder: same state in, same call out."""
        model = no_model()
        run(sending_pack(), model, {"to": "ops@example.com"}, tools=self.registry)
        self.assertEqual(model.calls, [])

    def test_a_tool_that_raised_is_not_retried_however_many_retries_it_declares(self):
        """A re-sample of a function is the same call again — there is nothing to vary."""
        attempted = []
        registry = ToolRegistry()

        @registry.register("send_email", reads=["to"], writes=["receipt"])
        def send_email(to):
            attempted.append(to)
            raise RuntimeError("smtp is down")

        result = run(sending_pack(retries=5, on_fail="desk"), no_model(),
                     {"to": "ops@example.com"}, tools=registry)
        self.assertEqual(attempted, ["ops@example.com"])
        self.assertEqual(result.end_node, "desk")

    def test_a_tool_node_would_overwrite_an_input_no_more_than_a_generation_would(self):
        with self.assertRaises(StateCollision):
            run(sending_pack(), no_model(),
                {"to": "ops@example.com", "receipt": "the caller's"},
                tools=self.registry)

    def test_a_tool_node_routes_on_its_edges_like_any_other_node(self):
        pack = build(
            nodes=[
                tool_node("send", "send_email"),
                Node(name="done", type="end"),
                Node(name="desk", type="end"),
            ],
            edges=[Edge("send", "desk", {"receipt": "sent-1"}), Edge("send", "done")],
            entry="send",
        )
        result = run(pack, no_model(), {"to": "ops@example.com"},
                     tools=self.registry)
        self.assertEqual(result.end_node, "desk")


class TestAToolNodeNeedsATool(unittest.TestCase):
    """Refused clearly at the node, rather than obscurely somewhere inside it."""

    def test_a_run_given_no_registry_says_what_to_pass(self):
        with self.assertRaises(ToolsNotAvailable) as caught:
            run(sending_pack(), no_model(), {"to": "ops@example.com"})
        self.assertIn("pass tools= to run()", str(caught.exception))

    def test_a_pack_with_no_tool_nodes_never_asks_for_one(self):
        pack = build(
            nodes=[generate("classify"), Node(name="done", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )
        result = run(pack, FakeModel(['{"category": "billing"}']), {"ticket": "t"})
        self.assertEqual(result.end_node, "done")

    def test_a_node_naming_a_tool_the_host_never_registered_is_refused(self):
        pack = sending_pack()
        registry = ToolRegistry()
        with self.assertRaises(ToolNotRegistered):
            run(pack, no_model(), {"to": "ops@example.com"}, tools=registry)

    def test_an_unregistered_tool_is_not_papered_over_by_on_fail(self):
        """A pack naming an action the host never allowed is a fact about the pack.

        Diverting it would finish the workflow with its acting half quietly missing,
        which is the one outcome an `on_fail` edge must not be able to produce.
        """
        pack = sending_pack(on_fail="desk")
        with self.assertRaises(ToolNotRegistered):
            run(pack, no_model(), {"to": "ops@example.com"}, tools=ToolRegistry())

    def test_a_tool_node_that_names_nothing_says_which_key_is_missing(self):
        pack = build(
            nodes=[_Node(name="send", type="tool"), Node(name="done", type="end")],
            edges=[("send", "done")],
            entry="send",
        )
        with self.assertRaises(ToolsNotAvailable) as caught:
            run(pack, no_model(), {}, tools=sending_registry([]))
        self.assertIn("`tool:`", str(caught.exception))


class TestToolFailuresTakeTheDeclaredEdge(unittest.TestCase):
    """A database being down takes the path a model failing takes."""

    def test_a_tool_that_raised_follows_on_fail(self):
        sent = []
        result = run(sending_pack(on_fail="desk"), no_model(),
                     {"to": "ops@example.com"},
                     tools=sending_registry(sent, fail=True))
        self.assertEqual(result.end_node, "desk")
        self.assertEqual(sent, [])

    def test_the_divert_is_recorded_in_failures_like_a_spent_ladder(self):
        result = run(sending_pack(on_fail="desk"), no_model(),
                     {"to": "ops@example.com"},
                     tools=sending_registry([], fail=True))
        self.assertEqual([failure.node for failure in result.failures], ["send"])
        self.assertEqual(result.failures[0].attempts, 0)
        self.assertIn("smtp is down", result.failures[0].reason)

    def test_a_tool_that_raised_with_nowhere_to_go_stops_the_run(self):
        with self.assertRaises(ToolFailed):
            run(sending_pack(), no_model(), {"to": "ops@example.com"},
                tools=sending_registry([], fail=True))

    def test_a_tool_needing_state_nobody_wrote_follows_on_fail(self):
        result = run(sending_pack(on_fail="desk"), no_model(), {},
                     tools=sending_registry([]))
        self.assertEqual(result.end_node, "desk")

    def test_a_tool_that_broke_its_own_declared_writes_follows_on_fail(self):
        registry = ToolRegistry()

        @registry.register("send_email", reads=["to"], writes=["receipt"])
        def send_email(to):
            return {"something_else": 1}

        result = run(sending_pack(on_fail="desk"), no_model(),
                     {"to": "ops@example.com"}, tools=registry)
        self.assertEqual(result.end_node, "desk")

    def test_a_broken_contract_with_nowhere_to_go_stops_the_run(self):
        registry = ToolRegistry()

        @registry.register("send_email", reads=["to"], writes=["receipt"])
        def send_email(to):
            return {"something_else": 1}

        with self.assertRaises(ToolContract):
            run(sending_pack(), no_model(), {"to": "ops@example.com"},
                tools=registry)


class TestASideEffectHappensExactlyOnce(unittest.TestCase):
    """The property the whole feature exists for.

    Committed state cannot answer "did the email go out?" — it records what a call
    *returned*, never that it *happened*. So the call itself is written down, and a
    resumed run replays it.
    """

    def setUp(self):
        self.sent = []
        self.store = DyingStore()
        self.addCleanup(self.store.close)
        # Kill the worker on the save that would settle the node — after the call has
        # returned and been written down, before the walk has left the node.
        self.store.die_on = lambda kw: (kw.get("node") == "send"
                                        and kw.get("next_node") == "done")

    def _crash(self, registry, pack=None, inputs=None):
        with self.assertRaises(Crash):
            run(pack or sending_pack(), no_model(),
                inputs if inputs is not None else {"to": "ops@example.com"},
                run_id="r", store=self.store, tools=registry)

    def test_a_run_that_crashed_after_sending_does_not_send_again(self):
        registry = sending_registry(self.sent)
        self._crash(registry)
        self.assertEqual(self.sent, ["ops@example.com"])

        self.store.die_on = None
        result = resume(sending_pack(), no_model(), "r", self.store, tools=registry)

        self.assertEqual(self.sent, ["ops@example.com"])
        self.assertEqual(result.end_node, "done")

    def test_the_replayed_result_is_committed_as_if_the_call_had_just_happened(self):
        registry = sending_registry(self.sent)
        self._crash(registry)
        self.store.die_on = None
        result = resume(sending_pack(), no_model(), "r", self.store, tools=registry)
        self.assertEqual(result.state["receipt"], "sent-1")
        self.assertEqual(result.provenance["receipt"], "send")

    def test_the_checkpoint_holds_the_node_the_arguments_and_the_result(self):
        self._crash(sending_registry(self.sent))
        [record] = self.store.latest("r").tool_calls
        self.assertEqual(record["node"], "send")
        self.assertEqual(record["tool"], "send_email")
        self.assertEqual(record["args"], {"to": "ops@example.com"})
        self.assertEqual(record["result"], {"receipt": "sent-1"})

    def test_the_record_is_written_before_the_walk_leaves_the_node(self):
        """`next_node` still points at the tool node, so a resume lands back on it."""
        self._crash(sending_registry(self.sent))
        self.assertEqual(self.store.latest("r").next_node, "send")

    def test_the_record_is_cleared_once_the_node_has_been_left(self):
        self.store.die_on = None
        run(sending_pack(), no_model(), {"to": "ops@example.com"}, run_id="r",
            store=self.store, tools=sending_registry(self.sent))
        for checkpoint in self.store.history("r"):
            self.assertEqual(checkpoint.tool_calls, [])

    def test_an_idempotent_tool_skips_the_bookkeeping_and_is_simply_called_again(self):
        """The documented trade: `idempotent=True` buys a cheaper checkpoint and pays
        for it by being re-run, which is exactly what declaring it promises is safe."""
        registry = sending_registry(self.sent, idempotent=True)
        self._crash(registry)
        self.assertEqual(self.sent, ["ops@example.com"])
        self.assertEqual(self.store.latest("r").tool_calls, [])

        self.store.die_on = None
        resume(sending_pack(), no_model(), "r", self.store, tools=registry)

        self.assertEqual(self.sent, ["ops@example.com", "ops@example.com"])

    def test_a_tool_inside_a_loop_is_called_once_per_visit(self):
        """Clearing the record on the way out is what keeps a loop honest."""
        calls = []
        registry = ToolRegistry()

        @registry.register("tick", writes=["count", "finished"])
        def tick():
            calls.append(len(calls) + 1)
            return {"count": len(calls), "finished": len(calls) >= 3}

        pack = build(
            nodes=[tool_node("tick", "tick"),
                   Node(name="done", type="end", output=["count"])],
            edges=[Edge("tick", "done", {"finished": True}), Edge("tick", "tick")],
            entry="tick",
        )
        store = Store(":memory:")
        self.addCleanup(store.close)
        result = run(pack, no_model(), run_id="loop", store=store, tools=registry)
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(result.output, {"count": 3})

    def test_a_call_whose_node_cannot_be_left_stays_pending(self):
        """Routing failed after the call landed, so a repaired resume must not re-call."""
        sent = []
        registry = sending_registry(sent)
        pack = build(
            nodes=[tool_node("send", "send_email"), Node(name="done", type="end")],
            edges=[Edge("send", "done", {"receipt": "never"})],
            entry="send",
        )
        store = Store(":memory:")
        self.addCleanup(store.close)
        with self.assertRaises(DeadEnd):
            run(pack, no_model(), {"to": "ops@example.com"}, run_id="d",
                store=store, tools=registry)

        checkpoint = store.latest("d")
        self.assertEqual(checkpoint.next_node, "send")
        self.assertEqual(len(checkpoint.tool_calls), 1)

        with self.assertRaises(DeadEnd):
            resume(pack, no_model(), "d", store, tools=registry)
        self.assertEqual(sent, ["ops@example.com"])

    def _bumping(self, calls):
        """seed (generate) -> bump (tool reading and writing one field) -> done.

        A tool whose own output lands in a field it also reads is the case that catches a
        resume point taken after the commit instead of before it: the arguments would
        have moved under the record, and a call that must be replayed would be refused as
        a mismatch instead.
        """
        registry = ToolRegistry()

        @registry.register("bump", reads=["count"], writes=["count"])
        def bump(count):
            calls.append(count)
            return {"count": count + 1}

        pack = build(
            nodes=[generate("seed"), tool_node("bump", "bump"),
                   Node(name="done", type="end", output=["count"])],
            edges=[("seed", "bump"), ("bump", "done")],
            entry="seed",
        )
        return pack, registry

    def test_a_tool_that_reads_what_it_writes_still_replays_after_a_crash(self):
        calls = []
        pack, registry = self._bumping(calls)
        self.store.die_on = lambda kw: (kw.get("node") == "bump"
                                        and kw.get("next_node") == "done")
        with self.assertRaises(Crash):
            run(pack, FakeModel(['{"count": 0}']), run_id="r", store=self.store,
                tools=registry)

        self.store.die_on = None
        result = resume(pack, no_model(), "r", self.store, tools=registry)

        self.assertEqual(calls, [0])
        self.assertEqual(result.output, {"count": 1})

    def test_a_tool_that_reads_what_it_writes_still_replays_after_a_dead_end(self):
        calls = []
        pack, registry = self._bumping(calls)
        pack = build(
            nodes=list(pack.nodes.values()),
            edges=[("seed", "bump"), Edge("bump", "done", {"count": "never"})],
            entry="seed",
        )
        with self.assertRaises(DeadEnd):
            run(pack, FakeModel(['{"count": 0}']), run_id="r", store=self.store,
                tools=registry)
        with self.assertRaises(DeadEnd):
            resume(pack, no_model(), "r", self.store, tools=registry)
        self.assertEqual(calls, [0])

    def test_a_record_that_no_longer_matches_the_state_stops_the_run(self):
        """Neither choice is safe alone, so neither is made quietly."""
        registry = sending_registry(self.sent)
        self._crash(registry)
        checkpoint = self.store.latest("r")
        self.store.die_on = None
        # A checkpoint that has come apart from the call recorded in it. Re-saving a step
        # replaces it, which is how this is staged rather than how it would happen.
        self.store.save(run_id="r", step=checkpoint.step, node="send",
                        next_node="send", state={"to": "someone-else@example.com"},
                        path=checkpoint.path, provenance=checkpoint.provenance,
                        failures=checkpoint.failures, pack=sending_pack(),
                        tool_calls=checkpoint.tool_calls)

        with self.assertRaises(ToolReplayMismatch) as caught:
            resume(sending_pack(), no_model(), "r", self.store, tools=registry)

        self.assertIn("(to differ)", str(caught.exception))
        self.assertNotIn("someone-else@example.com", str(caught.exception))
        self.assertEqual(self.sent, ["ops@example.com"])

    def test_a_store_that_cannot_record_the_call_says_so_and_still_runs(self):
        saved = []

        class OlderStore:
            def save(self, run_id, step, node, next_node, state, path=None,
                     provenance=None, failures=None, output=None, pack=None,
                     pack_version=None, attempts=None):
                saved.append(node)

        with self.assertLogs("jig.graph", level="WARNING") as caught:
            result = run(sending_pack(), no_model(), {"to": "ops@example.com"},
                         run_id="old", store=OlderStore(),
                         tools=sending_registry(self.sent))
        self.assertEqual(result.end_node, "done")
        self.assertEqual(self.sent, ["ops@example.com"])
        self.assertTrue(any("tool.unrecorded" in line for line in caught.output))

    def test_a_run_with_no_store_at_all_is_still_perfectly_legal(self):
        result = run(sending_pack(), no_model(), {"to": "ops@example.com"},
                     tools=sending_registry(self.sent))
        self.assertEqual(result.end_node, "done")
        self.assertEqual(self.sent, ["ops@example.com"])


def unsure_signal(node="classify", reason="independent samples disagreed", attempts=3):
    """jig's unsure outcome, built without depending on its constructor.

    `jig.verify` owns the class and what raises it; this file owns only where the walker
    sends it. Constructing it defensively is what lets these tests keep proving the
    routing whichever shape the signal settles into.
    """
    try:
        return Unsure(node, reason, attempts=attempts)
    except TypeError:  # pragma: no cover - a signal with a different constructor
        try:
            return Unsure(reason)
        except TypeError:
            return Unsure()


def raise_unsure(*args, **kwargs):
    raise unsure_signal()


class TestUnsureIsRoutedApartFromFailure(unittest.TestCase):
    """"I cannot believe this answer" is not "there is no answer"."""

    def _pack(self, **kwargs):
        return build(
            nodes=[generate("classify", **kwargs),
                   Node(name="done", type="end"),
                   Node(name="desk", type="end"),
                   Node(name="rescue", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )

    def test_a_node_declaring_on_unsure_sends_it_to_a_person(self):
        pack = build(
            nodes=[_Node(name="classify", type="generate", prompt="go", grammar={},
                         on_unsure="desk", on_fail="rescue"),
                   Node(name="done", type="end"),
                   Node(name="desk", type="end"),
                   Node(name="rescue", type="end")],
            edges=[("classify", "done")],
            entry="classify",
        )
        with mock.patch("jig.graph.run_node", raise_unsure):
            result = run(pack, no_model(), {"ticket": "t"})
        self.assertEqual(result.end_node, "desk")

    def test_a_node_with_no_on_unsure_falls_back_to_on_fail(self):
        with mock.patch("jig.graph.run_node", raise_unsure):
            result = run(self._pack(on_fail="rescue"), no_model(), {"ticket": "t"})
        self.assertEqual(result.end_node, "rescue")

    def test_a_node_with_neither_aborts_rather_than_committing_on_a_coin_flip(self):
        with mock.patch("jig.graph.run_node", raise_unsure):
            with self.assertRaises(Unsure):
                run(self._pack(), no_model(), {"ticket": "t"})

    def test_the_divert_is_recorded_in_failures(self):
        with mock.patch("jig.graph.run_node", raise_unsure):
            result = run(self._pack(on_fail="rescue"), no_model(), {"ticket": "t"})
        self.assertEqual([failure.node for failure in result.failures], ["classify"])

    def test_nothing_the_unsure_node_would_have_written_reaches_state(self):
        with mock.patch("jig.graph.run_node", raise_unsure):
            result = run(self._pack(on_fail="rescue"), no_model(), {"ticket": "t"})
        self.assertEqual(result.state, {"ticket": "t"})

    def test_the_route_is_checkpointed_like_any_other_divert(self):
        store = Store(":memory:")
        self.addCleanup(store.close)
        with mock.patch("jig.graph.run_node", raise_unsure):
            run(self._pack(on_fail="rescue"), no_model(), {"ticket": "t"},
                run_id="u", store=store)
        self.assertEqual([point.next_node for point in store.history("u")],
                         ["rescue", None])
