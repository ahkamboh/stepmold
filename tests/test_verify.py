"""T6 — nothing enters state until it has been verified, and the retry ladder.

Plus the confidence gate that sits on top of it: a node that draws more than once and
commits only what its draws agree on.
"""

import logging
import unittest
from dataclasses import dataclass

from jig.codegen import Sampling
from jig.errors import BackendError, MissingVariable, NodeFailed, RunError
from jig.graph import run
from jig.model import FakeModel
from jig.pack import Edge, Node, Pack
from jig.verify import (
    DRAW_TEMPERATURE,
    Consensus,
    EmptyCompletion,
    GateError,
    Rejected,
    Unsure,
    extract_json,
    gate_for,
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
OTHER = '{"category": "technical"}'
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


@dataclass(frozen=True)
class GatedNode(Node):
    """A node carrying the gate's keys, for as long as `jig.pack.Node` does not.

    `verify.gate_for` reads `samples` and `agree` with `getattr`, so the runtime works
    whether or not the pack format has grown them yet — and `gated()` below builds the
    real `Node` the moment it has, so these tests move to the production record without
    being rewritten.
    """

    samples: int = 1
    agree: int = 0


def gated(name="classify", samples=2, agree=0, **kwargs):
    """A generate node with `samples`/`agree` on it."""
    options = {
        "type": "generate",
        "prompt": "Classify: {ticket}",
        "grammar": SCHEMA,
    }
    options.update(kwargs)
    try:
        return Node(name=name, samples=samples, agree=agree, **options)
    except TypeError:
        return GatedNode(name=name, samples=samples, agree=agree, **options)


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


# ------------------------------------------------------- the confidence gate (samples)

PAIR = {"type": "object", "properties": {"category": {"type": "string"},
                                         "amount": {"type": "number"}}}

BILLING_10 = '{"category": "billing", "amount": 10}'
BILLING_99 = '{"category": "billing", "amount": 99}'

# A payload shaped so that a leak into a log line or an exception message is unambiguous.
POISON = "REFUND-SCAM-9f3a-ssn-123-45-6789"


class TestTheGateIsReadOffTheNode(unittest.TestCase):
    """`gate_for` is the whole configuration surface: what to draw, and what counts."""

    def test_a_node_that_asks_for_nothing_draws_once(self):
        self.assertEqual(gate_for(node()), (1, 1))

    def test_an_unset_agree_is_a_strict_majority(self):
        self.assertEqual(gate_for(gated(samples=3)), (3, 2))
        self.assertEqual(gate_for(gated(samples=4)), (4, 3))
        self.assertEqual(gate_for(gated(samples=5)), (5, 3))

    def test_two_samples_default_to_needing_both(self):
        """A majority of two is two — the gate a pack most often means by `samples: 2`."""
        self.assertEqual(gate_for(gated(samples=2)), (2, 2))

    def test_an_explicit_agree_is_used_as_written(self):
        self.assertEqual(gate_for(gated(samples=5, agree=5)), (5, 5))

    def test_a_threshold_no_run_can_reach_is_refused(self):
        with self.assertRaises(GateError) as caught:
            gate_for(gated(samples=3, agree=4))
        message = str(caught.exception)
        self.assertIn("classify", message)
        self.assertIn("agree", message)
        self.assertIn("samples", message)

    def test_a_threshold_of_one_is_refused_because_it_never_fires(self):
        """`agree: 1` accepts the first answer, so the other draws are never taken.

        Silently allowing it hands an author a gate they believe in and a run that does
        not have one, which is worse than no gate at all.
        """
        with self.assertRaises(GateError):
            gate_for(gated(samples=3, agree=1))

    def test_agree_without_samples_is_refused(self):
        with self.assertRaises(GateError):
            gate_for(gated(samples=1, agree=2))

    def test_zero_samples_is_refused(self):
        with self.assertRaises(GateError):
            gate_for(gated(samples=0))

    def test_a_boolean_is_not_a_count(self):
        """`samples: yes` is a plausible slip in YAML, and Python calls True an int."""
        with self.assertRaises(GateError):
            gate_for(gated(samples=True))

    def test_a_string_is_not_a_count(self):
        with self.assertRaises(GateError):
            gate_for(gated(samples="3"))

    def test_a_gate_error_is_a_run_error_so_a_walker_can_hold_it(self):
        self.assertTrue(issubclass(GateError, RunError))


class TestANodeWithoutSamplesIsUnchanged(unittest.TestCase):
    """Backwards compatibility, stated as tests rather than as a claim.

    Every pack that exists was written before the gate did. None of them may pay a token,
    change a request, or gain a field because the gate now exists.
    """

    def test_it_draws_exactly_once(self):
        model = FakeModel([GOOD, OTHER])
        self.assertEqual(run_node(node(), {"ticket": "t"}, model), {"category": "billing"})
        self.assertEqual(model.call_count, 1)

    def test_it_reports_no_consensus(self):
        report = {}
        run_node(node(), {"ticket": "t"}, FakeModel([GOOD]), consensus=report)
        self.assertEqual(report, {}, "a node that drew once compared nothing")

    def test_it_can_never_be_unsure(self):
        """One draw disagrees with nothing, whatever it says."""
        model = FakeModel([BAD_ENUM, GOOD])
        self.assertEqual(run_node(node(), {"ticket": "t"}, model), {"category": "billing"})

    def test_its_requests_are_byte_for_byte_the_ones_it_always_made(self):
        self.assertIsNone(sampling_for(0))
        for rung in range(1, 5):
            self.assertEqual(sampling_for(rung), sampling_for(rung, 0))
            self.assertEqual(sampling_for(rung).seed, rung)

    def test_a_walk_over_an_ungated_pack_is_untouched(self):
        result = run(pack_of(node()), FakeModel([BAD_ENUM, GOOD]), {"ticket": "t"})
        self.assertEqual(result.state, {"ticket": "t", "category": "billing"})
        self.assertEqual(result.attempts, {"classify": 2})


class TestAgreementIsWhatGetsCommitted(unittest.TestCase):
    """What `agree` means: the committed object, compared whole."""

    def test_matching_draws_commit_their_answer(self):
        model = FakeModel([GOOD, GOOD, GOOD])
        value = run_node(gated(samples=3, agree=2), {"ticket": "t"}, model)
        self.assertEqual(value, {"category": "billing"})

    def test_a_majority_wins_over_the_odd_one_out(self):
        """The committed value is the group's, not necessarily the first draw's."""
        model = FakeModel([GOOD, OTHER, OTHER])
        value = run_node(gated(samples=3, agree=2), {"ticket": "t"}, model)
        self.assertEqual(value, {"category": "technical"})

    def test_field_order_is_not_disagreement(self):
        """Two draws that emitted the same object differently are one answer."""
        model = FakeModel(['{"a": 1, "b": 2}', '{"b": 2, "a": 1}'])
        value = run_node(gated(samples=2, grammar=None), {"ticket": "t"}, model)
        self.assertEqual(value, {"a": 1, "b": 2})

    def test_a_difference_in_a_field_the_schema_does_not_gate_is_still_a_difference(self):
        """The reason agreement is not 'the fields that matter'.

        Nothing at this layer knows which fields matter, and the expensive mistake is
        the one in this direction: two draws that match on the category and differ on
        the amount are not a confident answer when the next node spends the amount.
        """
        model = FakeModel([BILLING_10, BILLING_99])
        with self.assertRaises(Unsure) as caught:
            run_node(gated(samples=2, grammar=PAIR), {"ticket": "t"}, model)
        self.assertEqual(caught.exception.consensus.agreed, 1)

    def test_one_and_one_point_zero_are_different_draws(self):
        """Documented strictness: canonical JSON, not Python equality.

        `{"n": 1}` and `{"n": 1.0}` are equal to Python and different on the wire. On a
        gate whose whole job is to notice that the model was not consistent, the strict
        reading is the safe one — it costs a rerun, never a wrong commit.
        """
        model = FakeModel(['{"n": 1}', '{"n": 1.0}'])
        with self.assertRaises(Unsure):
            run_node(gated(samples=2, grammar=None), {"ticket": "t"}, model)

    def test_nested_differences_are_seen(self):
        model = FakeModel(['{"a": {"b": [1, 2]}}', '{"a": {"b": [2, 1]}}'])
        with self.assertRaises(Unsure):
            run_node(gated(samples=2, grammar=None), {"ticket": "t"}, model)


class TestDisagreementIsUnsureNotRejected(unittest.TestCase):
    """A rejection means the output was invalid. Unsure means the model was inconsistent.

    They route differently because they mean different things, so they are different
    signals — `Unsure` is a sibling of `NodeFailed`, not a subclass of it.
    """

    def test_draws_that_do_not_agree_raise_unsure(self):
        model = FakeModel([GOOD, OTHER])
        with self.assertRaises(Unsure) as caught:
            run_node(gated(samples=2), {"ticket": "t"}, model)
        self.assertEqual(caught.exception.node, "classify")

    def test_unsure_is_routable_but_is_not_a_failure(self):
        self.assertTrue(issubclass(Unsure, RunError))
        self.assertFalse(issubclass(Unsure, NodeFailed))
        self.assertFalse(issubclass(Unsure, Rejected))

    def test_unsure_carries_the_answer_that_came_closest(self):
        """For a walker that hands it to a human rather than throwing it away."""
        model = FakeModel([OTHER, OTHER, GOOD])
        with self.assertRaises(Unsure) as caught:
            run_node(gated(samples=3, agree=3), {"ticket": "t"}, model)
        self.assertEqual(caught.exception.value, {"category": "technical"})

    def test_a_tie_goes_to_the_earliest_draw(self):
        model = FakeModel([GOOD, OTHER])
        with self.assertRaises(Unsure) as caught:
            run_node(gated(samples=2), {"ticket": "t"}, model)
        self.assertEqual(caught.exception.value, {"category": "billing"})

    def test_nothing_is_committed_and_state_is_untouched(self):
        state = {"ticket": "t"}
        model = FakeModel([GOOD, OTHER])
        with self.assertRaises(Unsure):
            run_node(gated(samples=2), state, model)
        self.assertEqual(state, {"ticket": "t"})

    def test_the_message_is_counts_and_names_only(self):
        """Unlike a NodeFailed, this one is safe at any log level whole.

        Every draw here was *valid* output — the model's answer about a customer's
        ticket. It rides on `.value` for a caller that wants it, and nowhere else.
        """
        model = FakeModel(['{"c": "%s"}' % POISON, '{"c": "other"}'])
        with self.assertRaises(Unsure) as caught:
            run_node(gated(samples=2, grammar=None), {"ticket": "t"}, model)
        self.assertNotIn(POISON, str(caught.exception))
        self.assertNotIn(POISON, caught.exception.feedback)
        self.assertNotIn(POISON, repr(caught.exception.consensus))
        self.assertEqual(caught.exception.value, {"c": POISON})

    def test_it_names_the_same_fields_a_node_failure_does(self):
        """One failure-recording path in the walker has to be able to hold both."""
        model = FakeModel([GOOD, OTHER])
        with self.assertRaises(Unsure) as caught:
            run_node(gated(samples=2), {"ticket": "t"}, model)
        unsure = caught.exception
        self.assertEqual(unsure.node, "classify")
        self.assertEqual(unsure.attempts, 2)
        self.assertTrue(unsure.reason)
        self.assertEqual(unsure.feedback, unsure.reason)


class TestSamplingCosts(unittest.TestCase):
    """n samples cost n generations, so the loop stops when the answer cannot change."""

    def test_it_stops_as_soon_as_enough_draws_match(self):
        model = FakeModel([GOOD, GOOD, GOOD])
        run_node(gated(samples=3, agree=2), {"ticket": "t"}, model)
        self.assertEqual(model.call_count, 2, "the third draw could not change anything")

    def test_it_stops_when_no_group_can_still_reach_the_threshold(self):
        """`samples: 3, agree: 3` is decided the moment two draws differ."""
        model = FakeModel([GOOD, OTHER, GOOD])
        with self.assertRaises(Unsure) as caught:
            run_node(gated(samples=3, agree=3), {"ticket": "t"}, model)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(caught.exception.consensus.drawn, 2)
        self.assertEqual(caught.exception.consensus.asked, 3)

    def test_it_pays_for_every_draw_it_actually_needs(self):
        model = FakeModel([GOOD, OTHER, OTHER])
        run_node(gated(samples=3, agree=2), {"ticket": "t"}, model)
        self.assertEqual(model.call_count, 3)

    def test_every_draw_is_on_the_node_s_bill(self):
        counts = {}
        run_node(gated(samples=3, agree=3), {"ticket": "t"},
                 FakeModel([GOOD, GOOD, GOOD]), attempts=counts)
        self.assertEqual(counts, {"classify": 3})


class TestTheLadderAndTheGate(unittest.TestCase):
    """A sample that fails verification is a rejection, not a disagreement."""

    def test_a_rejected_draw_climbs_the_ladder_and_does_not_count_as_a_vote(self):
        model = FakeModel([BAD_ENUM, GOOD, GOOD])
        report = {}
        value = run_node(gated(samples=2), {"ticket": "t"}, model, consensus=report)
        self.assertEqual(value, {"category": "billing"})
        self.assertEqual(model.call_count, 3)
        self.assertEqual(report["classify"].drawn, 2, "two answers, not three")
        self.assertEqual(report["classify"].agreed, 2)
        self.assertEqual(report["classify"].generations, 3, "the rejection cost a call")

    def test_each_draw_starts_its_ladder_clean(self):
        """A draw conditioned on another draw's rejection is not an independent draw."""
        model = FakeModel([BAD_ENUM, GOOD, GOOD])
        run_node(gated(samples=2), {"ticket": "t"}, model)
        self.assertIn("rejected", model.calls[1].prompt)      # rung 2 of draw 1
        self.assertNotIn("rejected", model.calls[2].prompt)   # rung 1 of draw 2

    def test_a_two_stage_node_thinks_once_per_draw(self):
        model = FakeModel(["NOTES-A", GOOD, "NOTES-B", GOOD])
        run_node(gated(samples=2, two_stage=True), {"ticket": "t"}, model)
        self.assertEqual(model.call_count, 4)
        self.assertNotIn("NOTES-A", model.calls[3].prompt,
                         "draw 2 must not emit from draw 1's reasoning")

    def test_a_draw_that_spends_its_whole_ladder_fails_the_node(self):
        """Not one dissenting voice: a node that produced no valid answer proved nothing."""
        model = FakeModel([GOOD, BAD_ENUM])
        with self.assertRaises(NodeFailed) as caught:
            run_node(gated(samples=2, retries=0), {"ticket": "t"}, model)
        self.assertEqual(caught.exception.node, "classify")

    def test_the_failure_reports_the_whole_visit_s_bill(self):
        model = FakeModel([GOOD, BAD_ENUM])
        counts = {}
        with self.assertRaises(NodeFailed) as caught:
            run_node(gated(samples=2, retries=0), {"ticket": "t"}, model, attempts=counts)
        self.assertEqual(caught.exception.attempts, 2, "the first draw was paid for too")
        self.assertEqual(counts, {"classify": 2})

    def test_a_failure_on_the_first_draw_is_todays_failure_untouched(self):
        model = FakeModel([BAD_ENUM] * 3)
        counts = {}
        with self.assertRaises(NodeFailed) as caught:
            run_node(gated(samples=2), {"ticket": "t"}, model, attempts=counts)
        self.assertEqual(caught.exception.attempts, 3)
        self.assertIn("category", str(caught.exception))
        self.assertEqual(counts, {"classify": 3})

    def test_an_empty_completion_is_still_a_rung_not_a_vote(self):
        class EmptyOnce:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, grammar=None, max_tokens=512):
                self.calls += 1
                if self.calls == 1:
                    raise EmptyCompletion("no content in that choice")
                return GOOD

        model = EmptyOnce()
        value = run_node(gated(samples=2), {"ticket": "t"}, model)
        self.assertEqual(value, {"category": "billing"})
        self.assertEqual(model.calls, 3)


class TestTheDrawsAreIndependent(unittest.TestCase):
    """Two identical requests are one draw charged twice — and would agree by construction."""

    def test_the_first_draw_is_still_the_one_the_operator_configured(self):
        model = SamplingModel([GOOD, GOOD])
        run_node(gated(samples=2), {"ticket": "t"}, model)
        self.assertIsNone(model.sampling[0])

    def test_every_later_draw_asks_for_a_different_one(self):
        model = SamplingModel([GOOD, GOOD, GOOD])
        run_node(gated(samples=3, agree=3), {"ticket": "t"}, model)
        self.assertEqual(model.sampling[1].temperature, DRAW_TEMPERATURE)
        self.assertEqual(model.sampling[2].temperature, DRAW_TEMPERATURE)
        self.assertNotEqual(model.sampling[1].seed, model.sampling[2].seed)

    def test_an_extra_draw_is_not_a_retry_and_does_not_climb(self):
        """A temperature ramp across draws would measure the ramp, not the model."""
        model = SamplingModel([GOOD, GOOD, GOOD, GOOD])
        run_node(gated(samples=4, agree=4), {"ticket": "t"}, model)
        temperatures = {hint.temperature for hint in model.sampling if hint}
        self.assertEqual(temperatures, {DRAW_TEMPERATURE})

    def test_no_two_generations_of_one_node_make_the_same_request(self):
        """Including across draw *and* rung — two draws that were retried identically
        would produce the same answer twice and be counted as agreement."""
        model = SamplingModel([BAD_ENUM, GOOD, BAD_ENUM, GOOD])
        run_node(gated(samples=2, retries=1), {"ticket": "t"}, model)
        seeds = [hint.seed for hint in model.sampling if hint]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(len(model.sampling), 4)

    def test_a_backend_that_cannot_vary_its_sampling_is_reported(self):
        """The silent failure: identical requests 'agree', and the pack reports a
        confidence nobody measured."""
        with self.assertLogs("jig.verify", level="WARNING") as caught:
            run_node(gated(samples=2), {"ticket": "t"}, FakeModel([GOOD, GOOD]))
        self.assertIn("node.samples.blind", [record.msg for record in caught.records])

    def test_a_backend_that_can_is_not_reported(self):
        with self.assertLogs("jig.verify", level="WARNING") as caught:
            # One record has to exist for assertLogs to pass, so a rejection provides it.
            run_node(gated(samples=2), {"ticket": "t"},
                     SamplingModel([BAD_ENUM, GOOD, GOOD]))
        self.assertNotIn("node.samples.blind", [record.msg for record in caught.records])

    def test_an_ungated_node_never_reports_it(self):
        with self.assertLogs("jig.verify", level="WARNING") as caught:
            # assertLogs needs at least one record to pass; this is it, and it is the
            # only one that may be there.
            logging.getLogger("jig.verify").warning("marker")
            run_node(node(), {"ticket": "t"}, FakeModel([GOOD]))
        self.assertEqual([record.msg for record in caught.records], ["marker"])


class TestTheGateReportsItself(unittest.TestCase):
    """`jig eval` has to be able to say how sure each node was."""

    def test_an_agreeing_node_records_what_it_found(self):
        report = {}
        run_node(gated(samples=3, agree=2), {"ticket": "t"},
                 FakeModel([GOOD, OTHER, OTHER]), consensus=report)
        record = report["classify"]
        self.assertEqual(
            (record.node, record.asked, record.drawn, record.agreed, record.required),
            ("classify", 3, 3, 2, 2),
        )
        self.assertEqual(record.generations, 3)
        self.assertFalse(record.unsure)

    def test_an_unsure_node_records_what_it_found_too(self):
        report = {}
        with self.assertRaises(Unsure) as caught:
            run_node(gated(samples=2), {"ticket": "t"}, FakeModel([GOOD, OTHER]),
                     consensus=report)
        self.assertTrue(report["classify"].unsure)
        self.assertEqual(report["classify"].agreed, 1)
        self.assertIs(caught.exception.consensus, report["classify"])

    def test_a_record_holds_no_model_output(self):
        """It is meant to be logged, checkpointed and printed. All three are downstreams
        a customer's ticket must not reach by accident."""
        report = {}
        run_node(gated(samples=2, grammar=None), {"ticket": "t"},
                 FakeModel(['{"c": "%s"}' % POISON] * 2), consensus=report)
        self.assertNotIn(POISON, repr(report["classify"]))

    def test_a_node_visited_twice_reports_its_latest_visit(self):
        report = {}
        gate = gated(samples=2)
        run_node(gate, {"ticket": "t"}, FakeModel([GOOD, GOOD]), consensus=report)
        with self.assertRaises(Unsure):
            run_node(gate, {"ticket": "t"}, FakeModel([GOOD, OTHER]), consensus=report)
        self.assertTrue(report["classify"].unsure)

    def test_the_report_is_optional(self):
        self.assertEqual(
            run_node(gated(samples=2), {"ticket": "t"}, FakeModel([GOOD, GOOD])),
            {"category": "billing"},
        )

    def test_unsure_is_derived_rather_than_stored(self):
        self.assertTrue(Consensus("n", asked=3, drawn=3, agreed=1, required=2,
                                  generations=3).unsure)
        self.assertFalse(Consensus("n", asked=3, drawn=2, agreed=2, required=2,
                                   generations=2).unsure)

    def test_the_shape_of_a_disagreement_is_reported_too(self):
        """Two defensible readings and four guesses are different problems."""
        report = {}
        with self.assertRaises(Unsure):
            run_node(gated(samples=4, agree=3), {"ticket": "t"},
                     FakeModel([GOOD, GOOD, OTHER, OTHER]), consensus=report)
        self.assertEqual(report["classify"].distinct, 2)
        self.assertEqual(report["classify"].agreed, 2)

    def test_agreeing_draws_report_one_distinct_answer(self):
        report = {}
        run_node(gated(samples=2), {"ticket": "t"}, FakeModel([GOOD, GOOD]),
                 consensus=report)
        self.assertEqual(report["classify"].distinct, 1)
