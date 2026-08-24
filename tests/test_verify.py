"""T6 — nothing enters state until it has been verified, and the retry ladder."""

import unittest

from jig.codegen import Sampling
from jig.errors import BackendError, MissingVariable, NodeFailed
from jig.graph import run
from jig.model import FakeModel
from jig.pack import Edge, Node, Pack
from jig.verify import (
    EmptyCompletion,
    Rejected,
    extract_json,
    run_node,
    sampling_for,
    verify,
)

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

    def test_the_last_object_wins_over_one_the_model_merely_quoted(self):
        """An echoed example or a quoted ticket is preamble; the answer comes last."""
        text = 'You asked for {"category": "technical"}. My answer: {"category": "billing"}'
        self.assertEqual(extract_json(text), {"category": "billing"})

    def test_an_earlier_object_is_still_recovered_when_the_last_one_is_not_json(self):
        text = 'Answer: {"category": "billing"} (shape: {category: string})'
        self.assertEqual(extract_json(text), {"category": "billing"})

    def test_a_fence_after_prose_is_found(self):
        self.assertEqual(
            extract_json('I said {"category": "technical"}\n```json\n'
                         '{"category": "billing"}\n```'),
            {"category": "billing"},
        )


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

    def test_a_two_stage_node_re_thinks_after_a_rejection(self):
        """The notes that produced a rejected answer are part of what was rejected.

        Re-emitting from them is not a re-sample: it is the same reasoning asked to
        format itself twice, which is why the long-horizon two_stage arm used to sit
        exactly on the naive analytical curve.
        """
        model = FakeModel(["FIRST-NOTES", BAD_ENUM, "SECOND-NOTES", GOOD])

        run_node(node(two_stage=True), {"ticket": "t"}, model)

        self.assertEqual(model.call_count, 4)
        self.assertIsNone(model.calls[0].grammar)         # think
        self.assertIsNotNone(model.calls[1].grammar)      # emit, rejected
        self.assertIsNone(model.calls[2].grammar)         # think again
        self.assertNotIn("FIRST-NOTES", model.calls[2].prompt)
        self.assertIn("SECOND-NOTES", model.calls[3].prompt)
        self.assertNotIn("FIRST-NOTES", model.calls[3].prompt)

    def test_a_single_stage_node_pays_for_no_thinking(self):
        model = FakeModel([BAD_ENUM, GOOD])
        run_node(node(), {"ticket": "t"}, model)
        self.assertEqual([call.grammar is None for call in model.calls], [False, False])


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


class SamplingModel:
    """A model that declares the ladder's optional sampling hint and records it."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.sampling = []

    def generate(self, prompt, grammar=None, max_tokens=512, sampling=None):
        self.prompts.append(prompt)
        self.sampling.append(sampling)
        return self.responses.pop(0)


class TestTheResampleIsAnIndependentDraw(unittest.TestCase):
    """A re-sample is only worth its tokens if it is a different draw (ARCHITECTURE.md §3)."""

    def test_the_first_attempt_asks_for_nothing(self):
        """A run that never stumbles must be the run the operator configured."""
        model = SamplingModel([GOOD])
        run_node(node(), {"ticket": "t"}, model)
        self.assertEqual(model.sampling, [None])

    def test_every_rung_after_the_first_asks_for_a_different_draw(self):
        model = SamplingModel([BAD_ENUM, BAD_ENUM, GOOD])

        run_node(node(), {"ticket": "t"}, model)

        self.assertIsNone(model.sampling[0])
        self.assertIsInstance(model.sampling[1], Sampling)
        self.assertGreater(model.sampling[2].temperature, model.sampling[1].temperature)
        self.assertNotEqual(model.sampling[2].seed, model.sampling[1].seed)

    def test_both_stages_of_a_re_thought_rung_carry_the_hint(self):
        """Re-thinking at the temperature that produced the discarded notes re-thinks
        its way back to them."""
        model = SamplingModel(["notes", BAD_ENUM, "other notes", GOOD])

        run_node(node(two_stage=True), {"ticket": "t"}, model)

        self.assertEqual(model.sampling[0], model.sampling[1])   # both None
        self.assertEqual(model.sampling[2], model.sampling[3])
        self.assertIsNotNone(model.sampling[2])

    def test_a_model_without_the_keyword_is_called_exactly_as_before(self):
        """The hint is optional in the protocol; sending it blindly would break every
        backend written against the three-argument signature."""
        model = FakeModel([BAD_ENUM, BAD_ENUM, GOOD])
        self.assertEqual(run_node(node(), {"ticket": "t"}, model), {"category": "billing"})
        self.assertEqual(model.call_count, 3)

    def test_the_ladder_stops_climbing_at_the_top_of_the_measured_range(self):
        self.assertLessEqual(sampling_for(1).temperature, sampling_for(9).temperature)
        self.assertLessEqual(sampling_for(9).temperature, 1.0)


class EmptyThen:
    """A backend that answers 200 with no text `misses` times, then answers properly."""

    def __init__(self, misses, then=GOOD, error=None):
        self.misses = misses
        self.then = then
        self.error = error
        self.calls = 0

    def generate(self, prompt, grammar=None, max_tokens=512):
        self.calls += 1
        if self.calls <= self.misses:
            raise self.error or EmptyCompletion(
                "the model spent its whole token budget reasoning and emitted no answer"
            )
        return self.then


class TestAContentLessTwoHundredIsASample(unittest.TestCase):
    """A 200 with `message.content` null is a bad draw, not a backend that is down.

    The endpoint answered. A re-sample is exactly the right response, and aborting the
    run past a declared `on_fail` is exactly the wrong one.
    """

    def test_the_ladder_re_samples_past_it(self):
        model = EmptyThen(misses=2)
        self.assertEqual(run_node(node(), {"ticket": "t"}, model), {"category": "billing"})
        self.assertEqual(model.calls, 3)

    def test_spending_the_ladder_on_it_fails_the_node_rather_than_the_run(self):
        model = EmptyThen(misses=9)
        with self.assertRaises(NodeFailed) as caught:
            run_node(node(), {"ticket": "t"}, model)
        self.assertEqual(caught.exception.attempts, 3)
        # The backend's diagnostic is what an operator needs; keep it verbatim.
        self.assertIn("token budget reasoning", str(caught.exception))

    def test_the_model_is_told_its_answer_arrived_empty(self):
        model = SamplingModel([GOOD])
        model.generate = lambda *a, **kw: (_ for _ in ()).throw(EmptyCompletion("x"))
        with self.assertRaises(NodeFailed):
            run_node(node(), {"ticket": "t"}, model)

    def test_a_backend_that_is_actually_down_still_stops_the_run(self):
        """No rung can fix an unreachable endpoint, and burning three delays the error."""
        model = EmptyThen(misses=1, error=BackendError("could not reach http://x"))
        with self.assertRaises(BackendError):
            run_node(node(), {"ticket": "t"}, model)
        self.assertEqual(model.calls, 1)

    def test_a_two_stage_node_keeps_notes_nobody_ever_judged(self):
        """Nothing about the reasoning was rejected — there was no answer to reject."""
        thoughts = []

        class ThinkThenEmpty:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, grammar=None, max_tokens=512):
                self.calls += 1
                if grammar is None:
                    thoughts.append(prompt)
                    return "MY-NOTES"
                if self.calls == 2:
                    raise EmptyCompletion("no content in that choice")
                return GOOD

        model = ThinkThenEmpty()
        run_node(node(two_stage=True), {"ticket": "t"}, model)

        self.assertEqual(len(thoughts), 1, "an empty answer is no reason to re-think")
        self.assertEqual(model.calls, 3)

    def test_the_node_still_takes_its_on_fail_edge(self):
        result = run(pack_of(node(on_fail="give_up")), EmptyThen(misses=9), {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "give_up"])
        self.assertIn("token budget", result.failures[0].reason)


class TestAttemptsAreCounted(unittest.TestCase):
    """A rescued run has to be distinguishable from a clean one (README problem 4)."""

    def test_run_node_tallies_into_the_caller_s_dict(self):
        counts = {}
        run_node(node(), {"ticket": "t"}, FakeModel([BAD_ENUM, GOOD]), attempts=counts)
        self.assertEqual(counts, {"classify": 2})

    def test_a_spent_ladder_is_counted_too(self):
        counts = {}
        with self.assertRaises(NodeFailed):
            run_node(node(), {"ticket": "t"}, FakeModel([BAD_ENUM] * 3), attempts=counts)
        self.assertEqual(counts, {"classify": 3})

    def test_a_run_reports_what_each_node_spent(self):
        result = run(pack_of(node()), FakeModel([BAD_ENUM, GOOD]), {"ticket": "t"})
        self.assertEqual(result.attempts, {"classify": 2})

    def test_a_clean_run_reports_one_attempt_per_node(self):
        result = run(pack_of(node()), FakeModel([GOOD]), {"ticket": "t"})
        self.assertEqual(result.attempts, {"classify": 1})


class DeeplyNestedOutputIsRejectedNotFatal(unittest.TestCase):
    """A model emitting 10,000 nested arrays is a bad generation, not a broken runtime.

    Before CPython 3.12, json.loads raises RecursionError rather than ValueError for input
    nested past the decoder's stack. That is not a ValueError, so it escaped extract_json's
    handler, was not a Rejected, and bypassed the retry ladder and the node's on_fail edge
    to kill the run. CI on 3.9, 3.10 and 3.11 caught it; the development interpreter (3.14)
    never could, so these force the decoder to raise and cover the path everywhere.
    """

    def _with_exploding_decoder(self, call):
        import jig.verify as verify_module

        real = verify_module.json.loads

        def exploding(*args, **kwargs):
            raise RecursionError("maximum recursion depth exceeded")

        verify_module.json.loads = exploding
        try:
            return call()
        finally:
            verify_module.json.loads = real

    def test_a_recursion_error_from_the_decoder_becomes_a_rejected(self):
        with self.assertRaises(Rejected) as caught:
            self._with_exploding_decoder(lambda: extract_json("[[[[[nested]]]]]"))
        self.assertIn("nested too deeply", str(caught.exception))

    def test_the_feedback_tells_the_model_what_to_do_instead(self):
        with self.assertRaises(Rejected) as caught:
            self._with_exploding_decoder(lambda: extract_json("[[[["))
        self.assertIn("flat JSON object", caught.exception.feedback)
