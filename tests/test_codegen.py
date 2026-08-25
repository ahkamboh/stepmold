"""T5 — think -> emit, and the scratchpad that must never reach state."""

import unittest

from stepmold.codegen import SCRATCHPAD, Sampling, accepts_sampling, generate_once
from stepmold.errors import BackendError
from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import Edge, Node, Pack

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


class TestTheSamplingHint(unittest.TestCase):
    """The optional per-call knob the retry ladder spends (`verify.run_node`)."""

    class Hintable:
        def __init__(self, responses):
            self.responses = list(responses)
            self.seen = []

        def generate(self, prompt, grammar=None, max_tokens=512, sampling=None):
            self.seen.append(sampling)
            return self.responses.pop(0)

    class Flexible:
        """A model that takes anything — a wrapper or a mock, and still a `Model`."""

        def __init__(self):
            self.seen = []

        def generate(self, prompt, **kwargs):
            self.seen.append(kwargs.get("sampling"))
            return '{"category": "billing"}'

    def test_a_model_that_declares_it_is_sent_it(self):
        model = self.Hintable(['{"category": "billing"}'])
        hint = Sampling(temperature=0.7, seed=1)
        generate_once(node(), {"ticket": "t"}, model, sampling=hint)
        self.assertEqual(model.seen, [hint])

    def test_both_stages_of_a_two_stage_node_get_it(self):
        model = self.Hintable(["notes", '{"category": "billing"}'])
        hint = Sampling(temperature=0.7, seed=1)
        generate_once(node(two_stage=True), {"ticket": "t"}, model, sampling=hint)
        self.assertEqual(model.seen, [hint, hint])

    def test_a_model_that_does_not_declare_it_never_sees_it(self):
        """FakeModel is written against the three-argument protocol and must stay valid."""
        model = FakeModel(['{"category": "billing"}'])
        self.assertFalse(accepts_sampling(model))
        generate_once(node(), {"ticket": "t"}, model,
                      sampling=Sampling(temperature=1.0))
        self.assertEqual(model.call_count, 1)

    def test_a_model_that_takes_arbitrary_keywords_counts_as_declaring_it(self):
        model = self.Flexible()
        self.assertTrue(accepts_sampling(model))
        generate_once(node(), {"ticket": "t"}, model, sampling=Sampling(temperature=0.5))
        self.assertEqual(model.seen, [Sampling(temperature=0.5)])

    def test_no_hint_means_the_call_is_exactly_what_it_always_was(self):
        model = self.Hintable(['{"category": "billing"}'])
        generate_once(node(), {"ticket": "t"}, model)
        self.assertEqual(model.seen, [None])


class TestNotesSurviveABackendFailure(unittest.TestCase):
    """A think stage that already ran is not paid for twice by a backend hiccup."""

    class ThinkThenFail:
        def generate(self, prompt, grammar=None, max_tokens=512):
            if grammar is None:
                return "MY-NOTES"
            raise BackendError("no content in that choice")

    def test_the_scratchpad_rides_out_on_the_exception(self):
        with self.assertRaises(BackendError) as caught:
            generate_once(node(two_stage=True), {"ticket": "t"}, self.ThinkThenFail())
        self.assertEqual(caught.exception.scratchpad, "MY-NOTES")
