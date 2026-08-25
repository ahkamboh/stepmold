"""Stage 4 of `stepmold build` — the scripted offline model, derived from the gold answers.

The interesting cases are all about *reach*: a branching graph calls a node once per case
that gets there, not once per case, and a two-stage node is called twice on one prompt.
Those are what these tests are about; the happy path is one test at the top.
"""

import json
import os
import shutil
import tempfile
import unittest

from stepmold.build.analyze import analyze
from stepmold.build.script import (
    THINK_ANSWER,
    THINK_KEY,
    check_script,
    keys_for,
    node_key,
    route,
    script_for,
)
from stepmold.build.spec import BuildError, FieldSpec, GraphPlan, NodePlan, TaskSpec
from stepmold.eval import evaluate
from stepmold.model import FakeModel
from stepmold.pack import load_pack

EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)
EXAMPLE = os.path.join(EXAMPLES, "support_triage")


def spec(name, type="string", **kwargs):
    return FieldSpec(name=name, type=type, **kwargs)


def task_of(cases, fields, inputs=("text",)):
    return TaskSpec(
        name="t",
        description="a task",
        inputs=list(inputs),
        fields=list(fields),
        cases=list(cases),
    )


def case(name, input, expect, end=None):
    out = {"name": name, "input": input, "expect": expect}
    if end:
        out["end"] = end
    return out


def answers(script, name):
    return [json.loads(text) for text in script[node_key(name)]]


# --------------------------------------------------------------- the mechanical part


class TestAnswers(unittest.TestCase):
    """A node's answer is the gold expectation for the fields it writes."""

    def setUp(self):
        self.plan = GraphPlan(
            entry="classify",
            nodes=[
                NodePlan(name="classify", writes=["kind"], purpose="Classify."),
                NodePlan(name="detail", writes=["score", "note"], purpose="Detail."),
            ],
            endings=["done"],
            edges=[
                {"from": "classify", "to": "detail"},
                {"from": "detail", "to": "done"},
            ],
        )
        self.task = task_of(
            [
                case("a", {"text": "one"}, {"kind": "bug", "score": 1, "note": "n1"}),
                case("b", {"text": "two"}, {"kind": "ask", "score": 2, "note": "n2"}),
            ],
            [spec("kind"), spec("score", "integer"), spec("note")],
        )

    def test_each_node_answers_with_only_the_fields_it_writes(self):
        script = script_for(self.task, self.plan)
        self.assertEqual(answers(script, "classify"), [{"kind": "bug"}, {"kind": "ask"}])
        self.assertEqual(
            answers(script, "detail"),
            [{"score": 1, "note": "n1"}, {"score": 2, "note": "n2"}],
        )

    def test_the_key_is_the_marker_phrase_a_prompt_must_contain(self):
        script = script_for(self.task, self.plan)
        self.assertEqual(sorted(script), sorted(["the classify step", "the detail step"]))

    def test_every_value_is_a_queue_even_when_one_case_reaches_the_node(self):
        # A bare string answers every prompt, so it can never run out — and a node that
        # can never run out is a node whose branching is never checked.
        one = task_of(self.task.cases[:1], self.task.fields)
        script = script_for(one, self.plan)
        self.assertIsInstance(script["the classify step"], list)
        self.assertEqual(len(script["the classify step"]), 1)

    def test_queue_order_is_evalset_order(self):
        # `stepmold eval` builds one FakeModel for the whole run, so the queue is consumed
        # across cases in file order.
        script = script_for(self.task, self.plan)
        self.assertEqual([a["kind"] for a in answers(script, "classify")], ["bug", "ask"])

    def test_a_field_that_is_null_in_half_the_cases_is_scripted_verbatim(self):
        plan = GraphPlan(
            entry="extract",
            nodes=[NodePlan(name="extract", writes=["order_id"], purpose="Extract.")],
            endings=["done"],
            edges=[{"from": "extract", "to": "done"}],
        )
        task = task_of(
            [
                case("a", {"text": "x"}, {"order_id": "A-1"}),
                case("b", {"text": "y"}, {"order_id": None}),
                case("c", {"text": "z"}, {"order_id": "C-3"}),
                case("d", {"text": "w"}, {"order_id": None}),
            ],
            [spec("order_id", optional=True, examples=["A-1", None])],
        )
        script = script_for(task, plan)
        self.assertEqual(
            [a["order_id"] for a in answers(script, "extract")],
            ["A-1", None, "C-3", None],
        )

    def test_a_field_a_case_does_not_pin_is_filled_from_the_enum(self):
        # The node's grammar requires every field it writes, so omitting one is a
        # rejection at a node the evalset was not testing.
        plan = GraphPlan(
            entry="route",
            nodes=[NodePlan(name="route", writes=["kind", "queue"], purpose="Route.")],
            endings=["done"],
            edges=[{"from": "route", "to": "done"}],
        )
        task = task_of(
            [case("a", {"text": "x"}, {"kind": "bug"})],
            [spec("kind"), spec("queue", enum=["eng", "ops"])],
        )
        self.assertEqual(
            answers(script_for(task, plan), "route"), [{"kind": "bug", "queue": "eng"}]
        )

    def unpinned(self, field):
        """What the script invents for a field `route` writes and no case pins."""
        plan = GraphPlan(
            entry="route",
            nodes=[NodePlan(name="route", writes=["kind", "ref"], purpose="Route.")],
            endings=["done"],
            edges=[{"from": "route", "to": "done"}],
        )
        task = task_of([case("a", {"text": "x"}, {"kind": "bug"})], [spec("kind"), field])
        return answers(script_for(task, plan), "route")[0]["ref"]

    def test_an_unpinned_optional_field_takes_an_observed_value_not_null(self):
        # `FieldSpec.schema` — the grammar the compiled pack ships — is
        # `{"type": "string"}` with no "null" in it, whatever `optional` says. So null is
        # the one value the node's own grammar is guaranteed to reject, and an observed
        # value is both legal and honest about the data.
        self.assertEqual(
            self.unpinned(spec("ref", optional=True, examples=["R-1", None])), "R-1"
        )

    def test_an_unpinned_field_with_no_observation_falls_back_to_a_typed_zero(self):
        # Nothing observed and nothing enumerated leaves only the zero of the type. It is
        # a lie about the data, but a grammar-legal one, and `check_script` names every
        # field it happens to.
        self.assertEqual(self.unpinned(spec("ref", optional=True)), "")
        self.assertEqual(self.unpinned(spec("ref", "integer", optional=True)), 0)
        self.assertEqual(self.unpinned(spec("ref", "boolean")), False)

    def test_an_unpinned_array_or_object_field_is_not_an_empty_string(self):
        # `_ZERO.get(type, "")` used to hand an array field `""`, which is rejected by
        # every grammar that declares the field an array — a whole pack scoring zero on a
        # field nobody was testing. `analyze` induces both types, so both need a zero.
        self.assertEqual(self.unpinned(spec("ref", "array", optional=True)), [])
        self.assertEqual(self.unpinned(spec("ref", "object", optional=True)), {})
        self.assertEqual(
            self.unpinned(spec("ref", "array", optional=True, examples=[["a"], None])),
            ["a"],
        )

    def test_an_enum_still_beats_an_observed_value(self):
        # An enum'd field admits nothing outside the enum, so the enum has to come first
        # even when `examples` holds something that looks more natural.
        self.assertEqual(
            self.unpinned(spec("ref", enum=["eng", "ops"], examples=["ops"])), "eng"
        )


# --------------------------------------------------------------- branching


def branching_plan():
    """triage -> (dropped) or detail -> done. Only non-spam reaches `detail`."""
    return GraphPlan(
        entry="triage",
        nodes=[
            NodePlan(name="triage", writes=["kind"], purpose="Triage."),
            NodePlan(name="detail", writes=["score"], purpose="Detail."),
        ],
        endings=["dropped", "done"],
        edges=[
            {"from": "triage", "to": "dropped", "when": {"kind": "spam"}},
            {"from": "triage", "to": "detail"},
            {"from": "detail", "to": "done"},
        ],
    )


def branching_task():
    return task_of(
        [
            case("ham1", {"text": "a"}, {"kind": "real", "score": 1}, end="done"),
            case("spam1", {"text": "b"}, {"kind": "spam"}, end="dropped"),
            case("ham2", {"text": "c"}, {"kind": "real", "score": 2}, end="done"),
            case("spam2", {"text": "d"}, {"kind": "spam"}, end="dropped"),
        ],
        [spec("kind", enum=["real", "spam"]), spec("score", "integer")],
    )


class TestBranching(unittest.TestCase):
    def setUp(self):
        self.plan, self.task = branching_plan(), branching_task()

    def test_a_node_is_answered_only_by_the_cases_that_reach_it(self):
        script = script_for(self.task, self.plan)
        self.assertEqual(len(script[node_key("triage")]), 4)
        self.assertEqual(len(script[node_key("detail")]), 2)

    def test_the_queue_skips_the_cases_that_ended_early(self):
        script = script_for(self.task, self.plan)
        self.assertEqual(answers(script, "detail"), [{"score": 1}, {"score": 2}])

    def test_route_names_the_nodes_and_the_ending(self):
        spam = route(self.task, self.plan, self.task.cases[1])
        self.assertEqual((spam.nodes, spam.ending), (["triage"], "dropped"))
        ham = route(self.task, self.plan, self.task.cases[0])
        self.assertEqual((ham.nodes, ham.ending), (["triage", "detail"], "done"))

    def test_a_case_no_edge_matches_is_named(self):
        plan = GraphPlan(
            entry="triage",
            nodes=[NodePlan(name="triage", writes=["kind"], purpose="Triage.")],
            endings=["dropped"],
            edges=[{"from": "triage", "to": "dropped", "when": {"kind": "spam"}}],
        )
        task = task_of(
            [
                case("ham1", {"text": "a"}, {"kind": "real"}),
                case("spam1", {"text": "b"}, {"kind": "spam"}),
            ],
            [spec("kind", enum=["real", "spam"])],
        )
        with self.assertRaises(BuildError) as caught:
            script_for(task, plan)
        self.assertIn("ham1", str(caught.exception))
        self.assertIn("triage", str(caught.exception))

    def test_a_cycle_is_reported_rather_than_hung_on(self):
        plan = GraphPlan(
            entry="a",
            nodes=[
                NodePlan(name="a", writes=["kind"], purpose="A."),
                NodePlan(name="b", writes=["score"], purpose="B."),
            ],
            endings=["done"],
            edges=[{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
        )
        with self.assertRaises(BuildError) as caught:
            script_for(self.task, plan)
        self.assertIn("loop", str(caught.exception))

    def test_an_edgeless_plan_is_read_as_a_straight_line(self):
        plan = GraphPlan(
            entry="triage",
            nodes=[
                NodePlan(name="triage", writes=["kind"], purpose="Triage."),
                NodePlan(name="detail", writes=["score"], purpose="Detail."),
            ],
            endings=["done"],
        )
        task = task_of(
            [case("a", {"text": "x"}, {"kind": "real", "score": 3})],
            [spec("kind"), spec("score", "integer")],
        )
        self.assertEqual(
            route(task, plan, task.cases[0]).nodes, ["triage", "detail"]
        )

    def test_an_edge_missing_a_side_says_what_shape_an_edge_is(self):
        plan = GraphPlan(
            entry="triage",
            nodes=[NodePlan(name="triage", writes=["kind"], purpose="Triage.")],
            endings=["done"],
            edges=[{"from": "triage"}],
        )
        task = task_of(
            [case("a", {"text": "a"}, {"kind": "real"})],
            [spec("kind", enum=["real", "spam"])],
        )
        with self.assertRaises(BuildError) as caught:
            script_for(task, plan)
        self.assertIn("'to'", str(caught.exception))


# --------------------------------------------------------------- two-stage nodes


def two_stage_plan():
    return GraphPlan(
        entry="weigh",
        nodes=[
            NodePlan(name="weigh", writes=["level"], purpose="Weigh.", two_stage=True),
        ],
        endings=["done"],
        edges=[{"from": "weigh", "to": "done"}],
    )


class TestTwoStage(unittest.TestCase):
    def setUp(self):
        self.plan = two_stage_plan()
        self.task = task_of(
            [
                case("a", {"text": "x"}, {"level": "p0"}),
                case("b", {"text": "y"}, {"level": "p2"}),
            ],
            [spec("level", enum=["p0", "p1", "p2"])],
        )

    def test_the_think_call_gets_its_own_entry(self):
        script = script_for(self.task, self.plan)
        self.assertEqual(script[THINK_KEY], THINK_ANSWER)

    def test_the_think_entry_is_a_string_not_a_queue(self):
        # The ladder re-thinks whenever it rejects a two-stage answer, so a queue here
        # runs out at the first retry.
        self.assertIsInstance(script_for(self.task, self.plan)[THINK_KEY], str)

    def test_the_node_queue_is_one_answer_per_case_not_two(self):
        script = script_for(self.task, self.plan)
        self.assertEqual(len(script[node_key("weigh")]), 2)

    def test_no_think_entry_when_nothing_is_two_stage(self):
        plan = GraphPlan(
            entry="weigh",
            nodes=[NodePlan(name="weigh", writes=["level"], purpose="Weigh.")],
            endings=["done"],
            edges=[{"from": "weigh", "to": "done"}],
        )
        self.assertNotIn(THINK_KEY, script_for(self.task, plan))

    def test_the_think_key_beats_the_node_key_on_the_think_prompt(self):
        # This is the whole trick: without a think.txt the think prompt is the emit
        # prompt plus a suffix, so both keys match it and the longer one has to win.
        from stepmold.codegen import DEFAULT_THINK_SUFFIX

        model = FakeModel(script_for(self.task, self.plan))
        emit_prompt = "You are the weigh step. Text: x"
        self.assertEqual(model.generate(emit_prompt + DEFAULT_THINK_SUFFIX), THINK_ANSWER)
        self.assertEqual(model.generate(emit_prompt), '{"level": "p0"}')

    def test_the_scripted_notes_cannot_shadow_the_emit_call(self):
        # The notes are quoted back into the emit prompt (codegen.SCRATCHPAD_BLOCK), so
        # a key hiding in them would answer the emit call with the notes.
        script = script_for(self.task, self.plan)
        for key in script:
            self.assertNotIn(key, THINK_ANSWER)


# --------------------------------------------------------------- what must not compile


class TestRefusals(unittest.TestCase):
    def test_an_empty_examples_file(self):
        plan = GraphPlan(
            entry="a",
            nodes=[NodePlan(name="a", writes=["kind"], purpose="A.")],
            endings=["done"],
            edges=[{"from": "a", "to": "done"}],
        )
        with self.assertRaises(BuildError) as caught:
            script_for(task_of([], [spec("kind")]), plan)
        self.assertIn("empty", str(caught.exception))

    def test_an_expected_field_no_node_writes_is_named(self):
        plan = GraphPlan(
            entry="a",
            nodes=[NodePlan(name="a", writes=["kind"], purpose="A.")],
            endings=["done"],
            edges=[{"from": "a", "to": "done"}],
        )
        task = task_of(
            [case("a", {"text": "x"}, {"kind": "bug", "queue": "eng"})],
            [spec("kind"), spec("queue")],
        )
        with self.assertRaises(BuildError) as caught:
            script_for(task, plan)
        self.assertIn("'queue'", str(caught.exception))

    def test_an_expectation_on_a_run_input_is_not_a_missing_field(self):
        # A case may assert that an input survived the run; no node writes it, and
        # nothing is wrong with that.
        plan = GraphPlan(
            entry="a",
            nodes=[NodePlan(name="a", writes=["kind"], purpose="A.")],
            endings=["done"],
            edges=[{"from": "a", "to": "done"}],
        )
        task = task_of(
            [case("a", {"text": "x"}, {"kind": "bug", "text": "x"})],
            [spec("kind")],
        )
        self.assertEqual(answers(script_for(task, plan), "a"), [{"kind": "bug"}])

    def test_a_plan_no_case_reaches_at_all(self):
        plan = GraphPlan(
            entry="done",
            nodes=[NodePlan(name="a", writes=["kind"], purpose="A.")],
            endings=["done"],
            edges=[],
        )
        task = task_of([case("a", {"text": "x"}, {"kind": "bug"})], [spec("kind")])
        with self.assertRaises(BuildError) as caught:
            script_for(task, plan)
        self.assertIn("nothing to script", str(caught.exception))

    def test_a_node_writing_a_field_the_examples_never_showed(self):
        plan = GraphPlan(
            entry="a",
            nodes=[NodePlan(name="a", writes=["kind", "ghost"], purpose="A.")],
            endings=["done"],
            edges=[{"from": "a", "to": "done"}],
        )
        task = task_of([case("a", {"text": "x"}, {"kind": "bug"})], [spec("kind")])
        with self.assertRaises(BuildError) as caught:
            script_for(task, plan)
        self.assertIn("ghost", str(caught.exception))


# --------------------------------------------------------------- the linter


PROMPTS = {
    "triage": "You are the triage step. Text: {text}",
    "detail": "You are the detail step. Kind: {kind}",
}


class TestCheckScript(unittest.TestCase):
    def setUp(self):
        self.plan, self.task = branching_plan(), branching_task()
        self.script = script_for(self.task, self.plan)

    def problems(self, script=None, **kwargs):
        return check_script(script or self.script, self.task, self.plan, **kwargs)

    def test_a_generated_script_lints_clean(self):
        self.assertEqual(self.problems(), [])

    def test_a_generated_script_lints_clean_against_its_prompts(self):
        self.assertEqual(self.problems(prompts=PROMPTS), [])

    def test_a_missing_node_entry(self):
        script = dict(self.script)
        del script[node_key("detail")]
        found = self.problems(script)
        self.assertEqual(len(found), 1)
        self.assertIn("'detail'", found[0])
        self.assertIn("ModelExhausted", found[0])

    def test_a_queue_that_is_too_short_names_the_starved_case(self):
        script = dict(self.script)
        script[node_key("triage")] = script[node_key("triage")][:2]
        found = self.problems(script)
        self.assertEqual(len(found), 1)
        self.assertIn("'ham2'", found[0])

    def test_a_queue_that_is_too_long(self):
        script = dict(self.script)
        script[node_key("detail")] = script[node_key("detail")] + ['{"score": 9}']
        found = self.problems(script)
        self.assertEqual(len(found), 1)
        self.assertIn("2 gold case(s) but its script holds 3", found[0])

    def test_a_node_scripted_once_per_case_when_the_branch_says_otherwise(self):
        # The mistake this whole stage exists to prevent: four cases, four answers, but
        # only two of them ever reach the node.
        script = dict(self.script)
        script[node_key("detail")] = ['{"score": 1}'] * 4
        self.assertTrue(any("holds 4" in p for p in self.problems(script)))

    def test_a_key_that_matches_no_node(self):
        script = dict(self.script)
        script["classify step"] = ['{"kind": "spam"}']
        self.assertTrue(any("matches no node" in p for p in self.problems(script)))

    def test_a_string_entry_is_reported_as_unfalsifiable(self):
        script = dict(self.script)
        script[node_key("detail")] = '{"score": 1}'
        found = self.problems(script)
        self.assertTrue(any("single string" in p for p in found))
        self.assertTrue(any("do not all expect the same answer" in p for p in found))

    def test_a_node_no_case_reaches(self):
        plan = GraphPlan(
            entry=self.plan.entry,
            nodes=self.plan.nodes + [NodePlan(name="spare", writes=[], purpose="X.")],
            endings=self.plan.endings,
            edges=self.plan.edges,
        )
        found = check_script(self.script, self.task, plan)
        self.assertTrue(any("reached by no gold case" in p for p in found))

    def test_an_answer_that_is_not_json(self):
        script = dict(self.script)
        script[node_key("detail")] = ["score: 1"]
        self.assertTrue(any("not valid JSON" in p for p in self.problems(script)))

    def test_an_answer_that_omits_a_field_the_grammar_requires(self):
        script = dict(self.script)
        script[node_key("detail")] = ["{}", "{}"]
        self.assertTrue(any("omits 'score'" in p for p in self.problems(script)))

    def test_an_answer_that_disagrees_with_its_case(self):
        script = dict(self.script)
        script[node_key("detail")] = ['{"score": 99}', '{"score": 2}']
        self.assertTrue(any("case 'ham1' expects 1" in p for p in self.problems(script)))

    def test_an_answer_outside_the_field_enum(self):
        # The 11-of-12 trap: one gold value the induced enum does not contain.
        task = task_of(
            [
                case("a", {"text": "x"}, {"kind": "real", "score": 1}, end="done"),
                case("b", {"text": "y"}, {"kind": "spam"}, end="dropped"),
                case("c", {"text": "z"}, {"kind": "real", "score": 2}, end="done"),
                case("d", {"text": "w"}, {"kind": "junk"}, end="dropped"),
            ],
            [spec("kind", enum=["real", "spam"]), spec("score", "integer")],
        )
        plan = GraphPlan(
            entry="triage",
            nodes=self.plan.nodes,
            endings=self.plan.endings,
            edges=[
                {"from": "triage", "to": "dropped", "when": {"kind": "spam"}},
                {"from": "triage", "to": "dropped", "when": {"kind": "junk"}},
                {"from": "triage", "to": "detail"},
                {"from": "detail", "to": "done"},
            ],
        )
        found = check_script(script_for(task, plan), task, plan)
        self.assertTrue(any("outside the field's enum" in p for p in found))

    def test_a_hand_written_key_the_prompt_really_contains_is_not_a_problem(self):
        # The linter's own contract is `node_key`, but a pack written by hand is under no
        # obligation to use it — "screen step", "clear-post step" and "the routing step"
        # are all in examples/. What matters is which key the prompt actually resolves to,
        # so a key checked against the prompt is a key the linter has nothing to say about.
        script = {
            "triage step": self.script[node_key("triage")],
            "detail step": self.script[node_key("detail")],
        }
        self.assertEqual(self.problems(script, prompts=PROMPTS), [])

    def test_a_key_that_only_a_rendered_prompt_contains(self):
        # examples/incident_triage keys its script on the alert id, so its keys occur in
        # no template at all — only in the prompt a case renders. The linter renders too.
        prompts = {"triage": "Alert {text}: triage it", "detail": "Alert {text}: detail it"}
        script = {}
        for text, answer in zip(["a", "b", "c", "d"], self.script[node_key("triage")]):
            script["Alert %s: triage" % text] = [answer]
        for text, answer in zip(["a", "c"], self.script[node_key("detail")]):
            script["Alert %s: detail" % text] = [answer]
        self.assertEqual(self.problems(script, prompts=prompts), [])

    def test_a_rendered_key_missing_for_one_case_names_that_case(self):
        prompts = {"triage": "Alert {text}: triage it", "detail": "Alert {text}: detail it"}
        script = {}
        for text, answer in zip(["a", "b", "c", "d"], self.script[node_key("triage")]):
            script["Alert %s: triage" % text] = [answer]
        script["Alert a: detail"] = [self.script[node_key("detail")][0]]
        found = self.problems(script, prompts=prompts)
        self.assertEqual(len(found), 1)
        self.assertIn("'ham2'", found[0])
        self.assertIn("ModelExhausted", found[0])

    def test_a_prompt_that_matches_another_nodes_key_first(self):
        prompts = dict(PROMPTS)
        prompts["detail"] = "Read what the triage step said and add detail: {kind}"
        found = self.problems(prompts=prompts)
        self.assertTrue(any("before its own key" in p for p in found))

    def test_a_prompt_that_carries_no_marker_at_all(self):
        prompts = dict(PROMPTS)
        prompts["detail"] = "Score this: {kind}"
        found = self.problems(prompts=prompts)
        self.assertTrue(any("none of the script's keys" in p for p in found))

    def test_an_ending_the_plan_does_not_route_to(self):
        task = task_of(
            [case("a", {"text": "x"}, {"kind": "spam"}, end="done")],
            [spec("kind", enum=["real", "spam"]), spec("score", "integer")],
        )
        found = check_script(script_for(task, self.plan), task, self.plan)
        self.assertTrue(any("take it to 'dropped'" in p for p in found))

    def test_a_field_no_case_pins_is_reported_once_not_once_per_case(self):
        plan = GraphPlan(
            entry="triage",
            nodes=[
                NodePlan(name="triage", writes=["kind"], purpose="Triage."),
                NodePlan(name="detail", writes=["score", "summary"], purpose="Detail."),
            ],
            endings=self.plan.endings,
            edges=self.plan.edges,
        )
        task = task_of(
            self.task.cases,
            list(self.task.fields) + [spec("summary", examples=["a summary"])],
        )
        found = [p for p in check_script(script_for(task, plan), task, plan)
                 if "summary" in p]
        self.assertEqual(len(found), 1)
        self.assertIn("no gold case pins", found[0])

    def test_a_missing_think_entry_for_a_two_stage_node(self):
        plan, task = two_stage_plan(), task_of(
            [case("a", {"text": "x"}, {"level": "p0"})], [spec("level")]
        )
        script = script_for(task, plan)
        del script[THINK_KEY]
        found = check_script(script, task, plan)
        self.assertTrue(any("no think entry" in p for p in found))

    def test_a_think_entry_with_no_two_stage_node(self):
        script = dict(self.script)
        script[THINK_KEY] = THINK_ANSWER
        self.assertTrue(any("no node in the plan is two-stage" in p
                            for p in self.problems(script)))

    def test_a_think_entry_that_is_a_queue(self):
        plan, task = two_stage_plan(), task_of(
            [case("a", {"text": "x"}, {"level": "p0"})], [spec("level")]
        )
        script = script_for(task, plan)
        script[THINK_KEY] = [THINK_ANSWER]
        found = check_script(script, task, plan)
        self.assertTrue(any("runs out mid-run" in p for p in found))

    def test_an_ordered_script_cannot_survive_a_branch(self):
        found = check_script(['{"kind": "spam"}'], self.task, self.plan)
        self.assertEqual(len(found), 1)
        self.assertIn("branching graph", found[0])

    def test_a_plan_that_cannot_be_walked_is_reported_not_raised(self):
        plan = GraphPlan(
            entry="triage",
            nodes=self.plan.nodes,
            endings=["dropped"],
            edges=[{"from": "triage", "to": "dropped", "when": {"kind": "spam"}}],
        )
        found = check_script(self.script, self.task, plan)
        self.assertTrue(any("cannot be walked" in p for p in found))

    def test_a_node_that_writes_a_run_input(self):
        plan = GraphPlan(
            entry="triage",
            nodes=[
                NodePlan(name="triage", writes=["kind"], purpose="Triage."),
                NodePlan(name="detail", writes=["score", "text"], purpose="Detail."),
            ],
            endings=self.plan.endings,
            edges=self.plan.edges,
        )
        found = check_script(self.script, self.task, plan)
        self.assertTrue(any("StateCollision" in p for p in found))

    def test_a_case_that_declares_a_rescue(self):
        task = task_of(
            [dict(self.task.cases[0], rescued=True)] + list(self.task.cases[1:]),
            self.task.fields,
        )
        found = check_script(self.script, task, self.plan)
        self.assertTrue(any("rescued: true" in p for p in found))

    def test_a_field_two_nodes_both_write(self):
        plan = GraphPlan(
            entry="triage",
            nodes=[
                NodePlan(name="triage", writes=["kind"], purpose="Triage."),
                NodePlan(name="detail", writes=["kind", "score"], purpose="Detail."),
            ],
            endings=self.plan.endings,
            edges=self.plan.edges,
        )
        found = check_script(self.script, self.task, plan)
        self.assertTrue(any("written by both" in p for p in found))


# --------------------------------------------------------------- keys off real prompts


class TestKeysFor(unittest.TestCase):
    """Where a key comes from when the pack's prompts already exist."""

    def setUp(self):
        self.plan = branching_plan()

    def test_with_no_prompts_the_key_is_the_published_marker(self):
        self.assertEqual(keys_for(self.plan), {"triage": ("the triage step", None),
                                               "detail": ("the detail step", None)})

    def test_the_marker_wins_when_the_prompt_carries_it(self):
        keys = keys_for(self.plan, PROMPTS)
        self.assertEqual(keys["triage"][0], "the triage step")

    def test_a_prompt_that_names_the_step_differently_gives_up_its_first_literal_line(self):
        # examples/content_moderation calls its `clear_post` node "the clear-post step"
        # and examples/incident_triage calls `route` "the routing step". No marker will
        # ever cover both, so the key is read off the prompt instead.
        prompts = {
            "triage": "You are the triage-and-screen step.\nText: {text}",
            "detail": "You are the detail step. Kind: {kind}",
        }
        keys = keys_for(self.plan, prompts)
        self.assertEqual(keys["triage"][0], "You are the triage-and-screen step.")
        self.assertEqual(keys["detail"][0], "the detail step")

    def test_a_line_holding_a_placeholder_cannot_be_a_key(self):
        # It is a different string in every case, so it would match no rendered prompt.
        prompts = {
            "triage": "Case {text}\nYou are the first-look step.",
            "detail": "You are the detail step. Kind: {kind}",
        }
        self.assertEqual(keys_for(self.plan, prompts)["triage"][0],
                         "You are the first-look step.")

    def test_a_key_two_prompts_share_is_refused_rather_than_shipped(self):
        # Shipping it would mean one node answering out of the other's queue, silently.
        prompts = {
            "triage": "You are a step of the workflow. Text: {text}",
            "detail": "You are a step of the workflow. Kind: {kind}",
        }
        with self.assertRaises(BuildError) as caught:
            keys_for(self.plan, prompts)
        self.assertIn("no other prompt", str(caught.exception))

    def test_a_two_stage_node_without_a_think_template_shares_the_suffix_key(self):
        keys = keys_for(two_stage_plan(), {"weigh": "You are the weigh step. {text}"})
        self.assertEqual(keys["weigh"], ("the weigh step", THINK_KEY))


# --------------------------------------------------------------- against a real pack


def support_triage_plan():
    """The four-node plan `induce` would propose for the shipped support_triage pack."""
    return GraphPlan(
        entry="classify",
        nodes=[
            NodePlan(name="classify", writes=["category"], purpose="Classify."),
            NodePlan(
                name="extract",
                writes=["order_id", "amount_usd", "sentiment"],
                purpose="Extract.",
            ),
            NodePlan(
                name="priority",
                writes=["priority", "reason"],
                purpose="Prioritise.",
                two_stage=True,
            ),
            NodePlan(
                name="emit",
                writes=["queue", "summary", "escalate"],
                purpose="Emit.",
            ),
        ],
        endings=["escalated", "done", "needs_human"],
        edges=[
            {"from": "classify", "to": "extract"},
            {"from": "extract", "to": "priority"},
            {"from": "priority", "to": "emit"},
            {"from": "emit", "to": "escalated", "when": {"priority": "p0"}},
            {"from": "emit", "to": "done"},
        ],
    )


def support_triage_task(pack_dir):
    """Its `TaskSpec`, with the two free-text fields no `expect` pins spelled out."""
    cases = []
    with open(os.path.join(pack_dir, "evalset.jsonl")) as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    fields = [
        spec("category", enum=["billing", "technical", "account", "other"]),
        spec("order_id", optional=True),
        spec("amount_usd", "number", optional=True),
        spec("sentiment", enum=["calm", "frustrated", "angry"]),
        spec("priority", enum=["p0", "p1", "p2", "p3"]),
        # Neither of these is in any `expect`, so both are answered by a placeholder.
        spec("reason", examples=["The customer is blocked."]),
        spec("summary", examples=["Customer reports a problem with their order."]),
        spec("queue", enum=["billing-ops", "eng-support", "identity", "general"]),
        spec("escalate", "boolean"),
    ]
    return TaskSpec(
        name="support_triage",
        description="triage a support ticket",
        inputs=["ticket"],
        fields=fields,
        cases=cases,
    )


class TestAgainstSupportTriage(unittest.TestCase):
    """The end-to-end proof: a generated script scores the shipped 12-case evalset.

    The pack is copied first and its hand-written `priority.think.txt` dropped, because a
    compiled pack has no think template — which is exactly the case the think entry is
    for. `TestAgainstSupportTriageWithItsThinkPrompt` below leaves it in place.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="stepmold-script-")
        self.pack_dir = os.path.join(self.root, "support_triage")
        shutil.copytree(EXAMPLE, self.pack_dir)
        os.remove(os.path.join(self.pack_dir, "prompts", "priority.think.txt"))
        self.addCleanup(shutil.rmtree, self.root)

    def plan(self):
        return support_triage_plan()

    def task(self):
        return support_triage_task(self.pack_dir)

    def test_the_generated_script_scores_the_shipped_evalset(self):
        task, plan = self.task(), self.plan()
        script = script_for(task, plan)
        with open(os.path.join(self.pack_dir, "fakes", "script.json"), "w") as handle:
            json.dump(script, handle, indent=2)
        report = evaluate(load_pack(self.pack_dir), FakeModel(script))
        self.assertEqual((report.passed, report.total), (12, 12), report.summary())

    def test_the_shipped_prompts_carry_the_marker_the_keys_need(self):
        task, plan = self.task(), self.plan()
        prompts = {}
        for node in plan.nodes:
            path = os.path.join(self.pack_dir, "prompts", "%s.txt" % node.name)
            with open(path) as handle:
                prompts[node.name] = handle.read()
        found = [
            problem
            for problem in check_script(script_for(task, plan), task, plan, prompts)
            # `reason` and `summary` are genuinely unpinned in this pack; that report is
            # the linter working, not a failure.
            if "no gold case pins" not in problem
        ]
        self.assertEqual(found, [])


# ------------------------------------------------------- the think prompt this pack ships


class TestAgainstSupportTriageWithItsThinkPrompt(unittest.TestCase):
    """The same pack, left as it ships — `prompts/priority.think.txt` still in place.

    A two-stage node makes two calls per visit and the pack decides what the first one is
    given. When the pack writes that prompt itself, the default suffix never appears in
    it, so the entry keyed on the suffix answers nothing and the think call finds no key
    at all. That is a 12/12 pack scoring 0/12, and it is invisible to anything that
    reasons about one call per node.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="stepmold-script-think-")
        self.pack_dir = os.path.join(self.root, "support_triage")
        shutil.copytree(EXAMPLE, self.pack_dir)
        self.addCleanup(shutil.rmtree, self.root)
        self.task = support_triage_task(self.pack_dir)
        self.plan = support_triage_plan()
        self.prompts, self.think_prompts = {}, {}
        for node in self.plan.nodes:
            self.prompts[node.name] = self.read("%s.txt" % node.name)
            if node.two_stage:
                self.think_prompts[node.name] = self.read("%s.think.txt" % node.name)

    def read(self, name):
        with open(os.path.join(self.pack_dir, "prompts", name)) as handle:
            return handle.read()

    def score(self, script):
        with open(os.path.join(self.pack_dir, "fakes", "script.json"), "w") as handle:
            json.dump(script, handle, indent=2)
        return evaluate(load_pack(self.pack_dir), FakeModel(script))

    def test_a_script_keyed_on_the_default_suffix_scores_nothing_here(self):
        # The failure the linter has to be able to see, demonstrated first so the check
        # below is testing something real rather than an imagined fault.
        report = self.score(script_for(self.task, self.plan))
        self.assertEqual(report.passed, 0, report.summary())
        self.assertIn("ModelExhausted", report.summary())

    def test_the_linter_names_the_think_call_no_key_can_answer(self):
        found = check_script(
            script_for(self.task, self.plan),
            self.task,
            self.plan,
            self.prompts,
            self.think_prompts,
        )
        self.assertTrue(
            any("'priority'" in p and "think prompt" in p and "ModelExhausted" in p
                for p in found),
            found,
        )

    def test_the_linter_passes_it_without_the_think_prompt(self):
        # Same script, same prompts, minus the think template: now the suffix-keyed entry
        # is the one that answers, and there is nothing to report. Proof that the check
        # above is about the think prompt and not about the pack.
        found = self.lint(script_for(self.task, self.plan), think_prompts={})
        self.assertEqual(found, [])

    def test_given_the_think_prompt_the_generated_script_scores_the_shipped_evalset(self):
        script = script_for(
            self.task, self.plan, self.prompts, self.think_prompts
        )
        report = self.score(script)
        self.assertEqual((report.passed, report.total), (12, 12), report.summary())
        self.assertEqual(self.lint(script), [])

    def test_the_think_key_is_read_off_the_think_template(self):
        keys = keys_for(self.plan, self.prompts, self.think_prompts)
        self.assertEqual(keys["priority"][1],
                         "You are the priority reasoning step of a support-ticket "
                         "triage workflow.")
        # And the emit key stays the published marker, which this prompt does carry.
        self.assertEqual(keys["priority"][0], node_key("priority"))

    def lint(self, script, think_prompts=None):
        found = check_script(
            script,
            self.task,
            self.plan,
            self.prompts,
            self.think_prompts if think_prompts is None else think_prompts,
        )
        # `reason` and `summary` are genuinely unpinned in this pack; that report is the
        # linter working, not a failure.
        return [p for p in found if "no gold case pins" not in p]


# --------------------------------------------------------------- every pack in examples/


def _load_case_lines(pack_dir):
    with open(os.path.join(pack_dir, "evalset.jsonl")) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _splice_asserts(pack):
    """The pack's edges with its `assert` nodes removed, and the entry moved past them.

    A `GraphPlan` holds generate nodes and endings; a pack may also route through nodes
    that evaluate an expression and cost no model call. An assert node that passes is a
    pass-through, so `P -> A -> T` is `P -> T` with the two `when` clauses merged, which
    is the plan a compiler would have produced for the same routing.
    """
    generate = {n for n, node in pack.nodes.items() if node.type == "generate"}
    ending = {n for n, node in pack.nodes.items() if node.type == "end"}
    edges = [{"from": e.source, "to": e.target, "when": e.when} for e in pack.edges]
    entry = pack.entry
    while entry not in generate and entry not in ending:
        entry = [edge["to"] for edge in edges if edge["from"] == entry][0]
    for _ in range(len(pack.nodes)):
        spliced = []
        for edge in edges:
            if edge["to"] in generate or edge["to"] in ending:
                spliced.append(edge)
                continue
            for follow in [e for e in edges if e["from"] == edge["to"]]:
                when = dict(edge.get("when") or {})
                when.update(follow.get("when") or {})
                spliced.append(
                    {"from": edge["from"], "to": follow["to"], "when": when or None}
                )
        edges = [edge for edge in spliced if edge["from"] in generate]
    return entry, edges


# `invoice_extract` routes on four `assert` nodes whose expressions are arithmetic over
# committed state (`abs(subtotal + tax_amount - total_amount) < 0.01`). A `GraphPlan`
# `when` clause is an equality on one key, so the expressions cannot be carried across;
# these edges name the value that makes each check fail instead, which reproduces the same
# routing over this gold set. It is the plan's shape that matters here, not its prose.
INVOICE_EDGES = [
    {"from": "header", "to": "amounts"},
    {"from": "amounts", "to": "due"},
    {"from": "due", "to": "flag_totals", "when": {"total_amount": 118.0}},
    {"from": "due", "to": "flag_tax", "when": {"tax_amount": 160.0}},
    {"from": "due", "to": "flag_currency", "when": {"currency": "PKR"}},
    {"from": "due", "to": "flag_overdue", "when": {"due_date": "2026-05-01"}},
    {"from": "due", "to": "review"},
    {"from": "review", "to": "flagged", "when": {"needs_review": True}},
    {"from": "review", "to": "accepted"},
    {"from": "flag_totals", "to": "flagged"},
    {"from": "flag_tax", "to": "flagged"},
    {"from": "flag_currency", "to": "flagged"},
    {"from": "flag_overdue", "to": "flagged"},
]
PACK_EDGES = {"invoice_extract": INVOICE_EDGES}


def compile_inputs(name):
    """Everything stage 4 needs to regenerate one shipped pack's offline model.

    The `TaskSpec` is `analyze` over the pack's own gold cases, and the `GraphPlan` is
    read off its `graph.yaml` and its grammars — one `NodePlan` per generate node, writing
    exactly the fields that node's grammar declares. Two adjustments, both because the
    pack's grammars are hand-written and stay that way while only the script is replaced:

    * a field the pack's own grammar closes with an enum takes that enum, since that is
      what the answer will be validated against. Only when every node that writes the
      field agrees: one `FieldSpec` cannot hold two different closed sets.
    * a field no gold case ever mentions is invisible to `analyze`, so its spec comes off
      the grammar too. `review_note`, `rationale`, `fit_reason` and `summary` are all of
      this kind — free text the evalset does not score.
    """
    pack_dir = os.path.join(EXAMPLES, name)
    pack = load_pack(pack_dir)
    task = analyze(pack.manifest.get("description", ""), _load_case_lines(pack_dir), name)

    nodes, prompts, think_prompts = [], {}, {}
    for node_name, node in pack.nodes.items():
        if node.type != "generate":
            continue
        properties = (node.grammar or {}).get("properties") or {}
        nodes.append(NodePlan(name=node_name, writes=list(properties),
                              purpose="Do the %s step." % node_name,
                              two_stage=node.two_stage))
        prompts[node_name] = node.prompt
        if node.think_prompt:
            think_prompts[node_name] = node.think_prompt
    entry, edges = _splice_asserts(pack)
    plan = GraphPlan(
        entry=entry,
        nodes=nodes,
        endings=[n for n, node in pack.nodes.items() if node.type == "end"],
        edges=PACK_EDGES.get(name, edges),
    )

    declared = {}
    for node_name, node in pack.nodes.items():
        for field, schema in ((node.grammar or {}).get("properties") or {}).items():
            declared.setdefault(field, []).append(json.dumps(schema, sort_keys=True))
    agreed = {field: json.loads(shapes[0]) for field, shapes in declared.items()
              if len(set(shapes)) == 1}

    fields = []
    for field in task.fields:
        schema = agreed.get(field.name) or {}
        if schema.get("enum") and list(schema["enum"]) != list(field.enum or []):
            field = FieldSpec(name=field.name, type=field.type, enum=list(schema["enum"]),
                              optional=field.optional, examples=field.examples)
        fields.append(field)
    known = {field.name for field in fields}
    for field, schema in agreed.items():
        if field in known:
            continue
        kind = schema.get("type", "string")
        if isinstance(kind, list):                      # {"type": ["string", "null"]}
            kind = [item for item in kind if item != "null"][0]
        fields.append(FieldSpec(name=field, type=kind, enum=schema.get("enum"),
                                optional=True))
    task = TaskSpec(name=task.name, description=task.description, inputs=task.inputs,
                    fields=fields, cases=task.cases)
    return pack_dir, task, plan, prompts, think_prompts


class ShippedPackCase(unittest.TestCase):
    """Regenerate one pack's offline model and score the pack with it.

    This is the only evidence that matters for this stage: not that the script looks
    right, but that substituting it for the one the pack ships leaves the evalset where
    it was.
    """

    pack = None

    def compile(self):
        return compile_inputs(self.pack)

    def generated(self):
        _, task, plan, prompts, think_prompts = self.compile()
        return script_for(task, plan, prompts, think_prompts)

    def score(self, script):
        pack_dir, _, _, _, _ = self.compile()
        root = tempfile.mkdtemp(prefix="stepmold-pack-")
        self.addCleanup(shutil.rmtree, root)
        dest = os.path.join(root, self.pack)
        shutil.copytree(pack_dir, dest)
        with open(os.path.join(dest, "fakes", "script.json"), "w") as handle:
            json.dump(script, handle, indent=2)
        return evaluate(load_pack(dest), FakeModel(script))

    def lint(self, script):
        _, task, plan, prompts, think_prompts = self.compile()
        return [
            problem
            for problem in check_script(script, task, plan, prompts, think_prompts)
            # A field no gold case pins is reported by design: nothing tests it. It is
            # information about the evalset, not a fault in the script.
            if "no gold case pins" not in problem
        ]

    def shipped(self):
        pack_dir, _, _, _, _ = self.compile()
        with open(os.path.join(pack_dir, "fakes", "script.json")) as handle:
            return json.load(handle)

    def assert_full_marks(self):
        script = self.generated()
        report = self.score(script)
        self.assertEqual((report.passed, report.total),
                         (report.total, report.total), report.summary())
        self.assertEqual(self.lint(script), [])

    def assert_shipped_script_lints_clean(self):
        self.assertEqual(self.lint(self.shipped()), [])


class TestSupportTriagePack(ShippedPackCase):
    pack = "support_triage"

    def test_the_generated_script_scores_full_marks(self):
        self.assert_full_marks()

    def test_the_shipped_script_lints_clean(self):
        self.assert_shipped_script_lints_clean()


class TestIncidentTriagePack(ShippedPackCase):
    pack = "incident_triage"

    def test_the_generated_script_scores_full_marks(self):
        self.assert_full_marks()

    def test_the_shipped_script_lints_clean(self):
        # This pack keys its script on the alert id, so not one of its keys occurs in a
        # prompt *template* — only in a rendered prompt. A linter that read keys off the
        # templates called every node of it unscripted.
        self.assert_shipped_script_lints_clean()


class TestInvoiceExtractPack(ShippedPackCase):
    pack = "invoice_extract"

    def test_the_generated_script_scores_full_marks(self):
        self.assert_full_marks()

    def test_the_shipped_script_lints_clean(self):
        # Five nodes here write `review_reason`, one per terminal reason, and four of them
        # are scripted with a plain string. Both are correct, and both used to be reported.
        self.assertEqual(check_script(self.shipped(), *self.compile()[1:]), [])


class TestLeadQualifyPack(ShippedPackCase):
    pack = "lead_qualify"

    def test_the_generated_script_scores_full_marks(self):
        self.assert_full_marks()

    def test_the_shipped_script_lints_clean(self):
        self.assert_shipped_script_lints_clean()


class TestMeetingActionsPack(ShippedPackCase):
    """The pack this stage cannot script, and why — recorded rather than glossed over.

    `decisions` is a list of objects that no gold case mentions at all, and the node that
    writes it asserts `decision_count == len(decisions)` against a `decision_count` every
    case does pin. There is no value a compiler can invent that satisfies an equation
    whose other side the evalset never states, so this pack's evalset cannot be scored
    from its own gold answers. The array placeholder is still worth getting right: `""`
    for a list field is rejected by the grammar at the node that writes it, which is a
    different and much less legible failure.
    """

    pack = "meeting_actions"

    def test_the_unpinned_array_fields_are_arrays(self):
        script = self.generated()
        actions = json.loads(script[node_key("actions")][1])
        self.assertIsInstance(actions["action_items"], list)
        decisions = json.loads(script[node_key("decisions")][0])
        self.assertEqual(decisions["decisions"], [])

    def test_the_linter_names_the_field_that_makes_it_unscriptable(self):
        _, task, plan, prompts, think_prompts = self.compile()
        found = check_script(self.generated(), task, plan, prompts, think_prompts)
        self.assertTrue(
            any("'decisions'" in p and "no gold case pins" in p for p in found), found
        )

    def test_the_shipped_script_lints_clean(self):
        self.assert_shipped_script_lints_clean()


class TestContentModerationPack(ShippedPackCase):
    """The other pack this stage cannot script, for a different reason.

    `signal` is the field the pack's first branch reads, and no gold case pins it. Every
    case therefore walks the same branch of the plan, so the queues are built for a
    routing the pack does not have — and `check_script` says exactly that rather than
    pretending otherwise. Nothing here is fixable from `stepmold/build/script.py`: the evalset
    would have to state the field its own graph branches on.
    """

    pack = "content_moderation"

    def test_the_branch_field_is_unpinned(self):
        _, task, _, _, _ = self.compile()
        pinned = set()
        for case_ in task.cases:
            pinned.update(case_["expect"])
        self.assertNotIn("signal", pinned)

    def test_the_linter_reports_the_nodes_the_plan_can_never_reach(self):
        _, task, plan, prompts, think_prompts = self.compile()
        found = check_script(self.shipped(), task, plan, prompts, think_prompts)
        self.assertTrue(
            any("'classify'" in p and "reached by no gold case" in p for p in found),
            found,
        )


if __name__ == "__main__":
    unittest.main()
