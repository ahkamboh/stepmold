"""T5 — think -> emit, and the scratchpad that must never reach state."""

import unittest

from jig.codegen import SCRATCHPAD, generate_once
from jig.graph import run
from jig.model import FakeModel
from jig.pack import Edge, Node, Pack

SCHEMA = {"type": "object", "properties": {"category": {"type": "string"}}}


def node(name="classify", **kwargs):
    options = {
        "type": "generate",
        "prompt": "Classify: {ticket}",
        "grammar": SCHEMA,
        "max_tokens": 64,
        "think_max_tokens": 16,
    }
    options.update(kwargs)
    return Node(name=name, **options)


def pack_of(target):
    return Pack(
        path="<memory>",
        name="t",
        version=1,
        entry=target.name,
        model=None,
        nodes={target.name: target, "done": Node(name="done", type="end")},
        edges=[Edge(target.name, "done", None)],
    )


class TestSingleStage(unittest.TestCase):
    def test_makes_exactly_one_model_call(self):
        model = FakeModel(['{"category": "billing"}'])
        generate_once(node(), {"ticket": "t"}, model)
        self.assertEqual(model.call_count, 1)

    def test_the_call_is_constrained_by_the_node_grammar(self):
        model = FakeModel(['{"category": "billing"}'])
        generate_once(node(), {"ticket": "t"}, model)
        self.assertEqual(model.calls[0].grammar["schema"], SCHEMA)
        self.assertEqual(model.calls[0].max_tokens, 64)

    def test_returns_the_emitted_text_and_no_scratchpad(self):
        model = FakeModel(['{"category": "billing"}'])
        attempt = generate_once(node(), {"ticket": "t"}, model)
        self.assertEqual(attempt.text, '{"category": "billing"}')
        self.assertIsNone(attempt.scratchpad)
        self.assertEqual(attempt.calls, 1)


class TestTwoStage(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel(["the customer was billed twice", '{"category": "billing"}'])
        self.attempt = generate_once(
            node(two_stage=True), {"ticket": "charged twice"}, self.model
        )

    def test_makes_exactly_two_model_calls(self):
        self.assertEqual(self.model.call_count, 2)

    def test_the_think_call_is_unconstrained_and_token_capped(self):
        think = self.model.calls[0]
        self.assertIsNone(think.grammar)
        self.assertEqual(think.max_tokens, 16)

    def test_the_emit_call_is_constrained(self):
        emit = self.model.calls[1]
        self.assertEqual(emit.grammar["schema"], SCHEMA)
        self.assertEqual(emit.max_tokens, 64)

    def test_the_emit_prompt_is_conditioned_on_the_scratchpad(self):
        self.assertIn("the customer was billed twice", self.model.calls[1].prompt)

    def test_the_think_prompt_renders_state_too(self):
        self.assertIn("charged twice", self.model.calls[0].prompt)

    def test_the_scratchpad_comes_back_on_the_attempt_only(self):
        self.assertEqual(self.attempt.scratchpad, "the customer was billed twice")
        self.assertEqual(self.attempt.text, '{"category": "billing"}')
        self.assertEqual(self.attempt.calls, 2)

    def test_a_node_think_prompt_file_overrides_the_default(self):
        model = FakeModel(["notes", '{"category": "x"}'])
        generate_once(
            node(two_stage=True, think_prompt="Custom think about {ticket}"),
            {"ticket": "t"},
            model,
        )
        self.assertEqual(model.calls[0].prompt, "Custom think about t")

    def test_an_explicit_scratchpad_placeholder_is_honoured(self):
        model = FakeModel(["notes here", '{"category": "x"}'])
        generate_once(
            node(two_stage=True, prompt="Notes: {%s}\nNow answer for {ticket}" % SCRATCHPAD),
            {"ticket": "t"},
            model,
        )
        self.assertEqual(model.calls[1].prompt, "Notes: notes here\nNow answer for t")

    def test_the_scratchpad_is_reused_when_one_is_supplied(self):
        model = FakeModel(['{"category": "x"}'])
        attempt = generate_once(
            node(two_stage=True), {"ticket": "t"}, model, scratchpad="earlier notes"
        )
        self.assertEqual(model.call_count, 1)
        self.assertIn("earlier notes", model.calls[0].prompt)
        self.assertEqual(attempt.scratchpad, "earlier notes")


class TestErrorFeedback(unittest.TestCase):
    def test_an_error_is_appended_to_the_emit_prompt(self):
        model = FakeModel(['{"category": "billing"}'])
        generate_once(node(), {"ticket": "t"}, model, error="category: required")
        prompt = model.calls[0].prompt
        self.assertIn("category: required", prompt)
        self.assertIn("Classify: t", prompt)

    def test_without_an_error_no_correction_block_appears(self):
        model = FakeModel(['{"category": "billing"}'])
        generate_once(node(), {"ticket": "t"}, model)
        self.assertEqual(model.calls[0].prompt, "Classify: t")


class TestScratchpadNeverReachesState(unittest.TestCase):
    def test_a_two_stage_node_commits_only_its_emitted_object(self):
        model = FakeModel(["SECRET REASONING", '{"category": "billing"}'])
        result = run(pack_of(node(two_stage=True)), model, {"ticket": "t"})
        self.assertEqual(result.state, {"ticket": "t", "category": "billing"})
        self.assertNotIn(SCRATCHPAD, result.state)
        self.assertNotIn("SECRET REASONING", repr(result.state))
        self.assertNotIn("SECRET REASONING", repr(result.output))

    def test_a_later_node_prompt_cannot_see_the_scratchpad(self):
        first = node("classify", two_stage=True)
        second = node("extract", prompt="Given {category}, extract from {ticket}")
        target = Pack(
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
        model = FakeModel(
            ["SECRET REASONING", '{"category": "billing"}', '{"amount": 1}']
        )
        run(target, model, {"ticket": "t"})
        self.assertNotIn("SECRET REASONING", model.calls[2].prompt)

    def test_a_single_stage_node_in_a_walk_makes_one_call(self):
        model = FakeModel(['{"category": "billing"}'])
        run(pack_of(node()), model, {"ticket": "t"})
        self.assertEqual(model.call_count, 1)
