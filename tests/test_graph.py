"""T4 — walking a graph from its entry node with a scripted model."""

import unittest

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
from jig.state import Store
from jig.graph import StateCollision, commit, replay, run


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
