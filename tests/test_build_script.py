"""Stage 4 of `jig build` — the scripted offline model, derived from the gold answers.

The interesting cases are all about *reach*: a branching graph calls a node once per case
that gets there, not once per case, and a two-stage node is called twice on one prompt.
Those are what these tests are about; the happy path is one test at the top.
"""

import json
import os
import shutil
import tempfile
import unittest

from jig.build.script import (
    THINK_ANSWER,
    THINK_KEY,
    check_script,
    node_key,
    route,
    script_for,
)
from jig.build.spec import BuildError, FieldSpec, GraphPlan, NodePlan, TaskSpec
from jig.eval import evaluate
from jig.model import FakeModel
from jig.pack import load_pack

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "support_triage",
)


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
        # `jig eval` builds one FakeModel for the whole run, so the queue is consumed
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

    def test_an_unpinned_optional_field_without_an_enum_is_null(self):
        plan = GraphPlan(
            entry="route",
            nodes=[NodePlan(name="route", writes=["kind", "ref"], purpose="Route.")],
            endings=["done"],
            edges=[{"from": "route", "to": "done"}],
        )
        task = task_of(
            [case("a", {"text": "x"}, {"kind": "bug"})],
            [spec("kind"), spec("ref", optional=True, examples=["R-1"])],
        )
        self.assertEqual(answers(script_for(task, plan), "route")[0]["ref"], None)


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
        from jig.codegen import DEFAULT_THINK_SUFFIX

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


# --------------------------------------------------------------- against a real pack


class TestAgainstSupportTriage(unittest.TestCase):
    """The end-to-end proof: a generated script scores the shipped 12-case evalset.

    The pack is copied first and its hand-written `priority.think.txt` dropped, because a
    compiled pack has no think template — which is exactly the case the think entry is
    for.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="jig-script-")
        self.pack_dir = os.path.join(self.root, "support_triage")
        shutil.copytree(EXAMPLE, self.pack_dir)
        os.remove(os.path.join(self.pack_dir, "prompts", "priority.think.txt"))
        self.addCleanup(shutil.rmtree, self.root)

    def plan(self):
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

    def task(self):
        cases = []
        with open(os.path.join(self.pack_dir, "evalset.jsonl")) as handle:
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


if __name__ == "__main__":
    unittest.main()
