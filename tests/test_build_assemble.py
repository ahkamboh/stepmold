"""Stage 5 — the compiler writes a pack, and the pack scores against its own gold cases.

Everything here is offline and deterministic: the pack's model is the scripted stand-in
the compiler itself emits, so a green run of this file proves the whole write → load →
`jig eval` loop with no GPU, no network and no API key.

The cases worth writing are the ones where a plausible emitter is quietly wrong: a value
jig's YAML subset resolves to something other than a string, a field that is null in half
the examples, a plan that writes a field twice or not at all, and a pack whose gold cases
do not pass — which the compiler must report rather than repair.
"""

import json
import os
import shutil
import tempfile
import unittest

from jig.build.assemble import compile_report, verify_pack, write_pack
from jig.build.spec import BuildError, FieldSpec, GraphPlan, NodePlan, TaskSpec
from jig.pack import load_pack
from jig.yamlish import parse as parse_yaml


# --------------------------------------------------------------------------- fixtures


def task_spec(**overrides):
    """A two-node triage task: `label` writes the kind, `detail` writes owner + score.

    `owner` is null in two of the three gold cases, which is the shape that decides how
    optional fields have to be expressed in a closed grammar.
    """
    spec = dict(
        name="note_triage",
        description="Triage a short internal note.\n\nLabel it, then pull the owner out.",
        inputs=["note"],
        fields=[
            FieldSpec("kind", "string", enum=["bug", "idea", "chore"], examples=["bug"]),
            FieldSpec("owner", "string", optional=True, examples=["ana"]),
            FieldSpec("score", "integer", examples=[0, 3]),
        ],
        cases=gold_cases(),
    )
    spec.update(overrides)
    return TaskSpec(**spec)


def gold_cases():
    return [
        {
            "name": "crash",
            "input": {"note": "crash on save"},
            "expect": {"kind": "bug", "owner": "ana", "score": 3},
            "end": "done",
        },
        {
            "name": "idea",
            "input": {"note": "maybe dark mode"},
            "expect": {"kind": "idea", "owner": None, "score": 1},
            "end": "done",
        },
        {
            "name": "chore",
            "input": {"note": "tidy the logs"},
            "expect": {"kind": "chore", "owner": None, "score": 0},
            "end": "done",
        },
    ]


def graph_plan(**overrides):
    spec = dict(
        entry="label",
        nodes=[
            NodePlan("label", ["kind"], "Label the note."),
            NodePlan(
                "detail",
                ["owner", "score"],
                "Name the owner and score the note.",
                two_stage=True,
                reads=["kind"],
            ),
        ],
        endings=["done"],
    )
    spec.update(overrides)
    return GraphPlan(**spec)


PROMPTS = {
    "label": 'label step\n\nNote: {note}\n\nAnswer {{"kind": "bug"}}.',
    "detail": "detail step\n\nNote: {note}\nKind: {kind}",
    "detail.think": "detail thinking step\n\nNote: {note}",
}

SCRIPT = {
    "label step": ['{"kind": "bug"}', '{"kind": "idea"}', '{"kind": "chore"}'],
    "detail thinking step": ["notes", "notes", "notes"],
    "detail step": [
        '{"owner": "ana", "score": 3}',
        '{"owner": null, "score": 1}',
        '{"owner": null, "score": 0}',
    ],
}


class PackCase(unittest.TestCase):
    """Every test gets its own throwaway directory; nothing here touches the repo."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.directory = os.path.join(self.root, "pack")

    def compile(self, task=None, plan=None, prompts=None, script=None, **kwargs):
        return write_pack(
            self.directory,
            task or task_spec(),
            plan or graph_plan(),
            PROMPTS if prompts is None else prompts,
            SCRIPT if script is None else script,
            **kwargs
        )

    def read(self, relative):
        with open(os.path.join(self.directory, relative.replace("/", os.sep))) as handle:
            return handle.read()


# ------------------------------------------------------------------------ end to end


class TestEndToEnd(PackCase):
    def test_the_written_pack_loads_and_scores_its_own_gold_cases(self):
        self.compile()
        report = verify_pack(self.directory)
        self.assertTrue(report.passed_all)
        self.assertEqual((report.passed, report.total), (3, 3))
        self.assertEqual(compile_report(report), "note_triage: 3/3 cases passed")

    def test_every_required_file_is_there(self):
        self.compile()
        for relative in (
            "manifest.yaml",
            "graph.yaml",
            "prompts/label.txt",
            "prompts/detail.txt",
            "prompts/detail.think.txt",
            "grammars/label.json",
            "grammars/detail.json",
            "evalset.jsonl",
            "fakes/script.json",
        ):
            self.assertTrue(
                os.path.isfile(os.path.join(self.directory, relative.replace("/", os.sep))),
                relative,
            )

    def test_the_pack_ships_its_own_offline_model(self):
        self.compile()
        pack = load_pack(self.directory)
        self.assertEqual(pack.model, "fake:fakes/script.json")
        self.assertEqual(json.loads(self.read("fakes/script.json")), SCRIPT)

    def test_write_pack_returns_the_directory_it_wrote(self):
        self.assertEqual(self.compile(), self.directory)

    def test_the_think_template_lands_where_jig_looks_for_it(self):
        self.compile()
        pack = load_pack(self.directory)
        self.assertTrue(pack.nodes["detail"].two_stage)
        self.assertEqual(
            pack.nodes["detail"].think_prompt, PROMPTS["detail.think"] + "\n"
        )
        self.assertIsNone(pack.nodes["label"].think_prompt)

    def test_two_stage_is_written_bare_because_a_quoted_false_is_true(self):
        # `two_stage` is the one node key jig does not shape-check: bool("false") is
        # True, so quoting the value would silently double the node's model calls.
        self.compile()
        self.assertIn("two_stage: true", self.read("graph.yaml"))
        self.assertNotIn('two_stage: "', self.read("graph.yaml"))


# --------------------------------------------------------------------------- grammars


class TestGrammars(PackCase):
    def test_a_node_grammar_is_closed_over_exactly_the_fields_it_writes(self):
        self.compile()
        grammar = json.loads(self.read("grammars/detail.json"))
        self.assertEqual(sorted(grammar["properties"]), ["owner", "score"])
        self.assertEqual(grammar["additionalProperties"], False)
        self.assertEqual(grammar["required"], ["owner", "score"])

    def test_an_enum_survives_into_the_grammar(self):
        self.compile()
        grammar = json.loads(self.read("grammars/label.json"))
        self.assertEqual(
            grammar["properties"]["kind"],
            {"type": "string", "enum": ["bug", "idea", "chore"]},
        )

    def test_a_field_null_in_half_the_cases_stays_required_and_gains_null(self):
        """Optional means "may be null", not "may be absent".

        A field that may be absent cannot be asserted by a gold case expecting null —
        the case would fail as "missing from output" — so the closed schema keeps the
        key required and widens the value instead.
        """
        self.compile()
        grammar = json.loads(self.read("grammars/detail.json"))
        self.assertEqual(grammar["properties"]["owner"], {"type": ["string", "null"]})
        self.assertIn("owner", grammar["required"])

    def test_an_optional_enum_field_admits_null_as_a_choice(self):
        """An enum that is null in one case would otherwise reject its own gold answer."""
        task = task_spec(
            fields=[
                FieldSpec("kind", "string", enum=["bug", "idea", "chore"]),
                FieldSpec("owner", "string", enum=["ana", "bo"], optional=True),
                FieldSpec("score", "integer"),
            ]
        )
        self.compile(task=task)
        grammar = json.loads(self.read("grammars/detail.json"))
        self.assertEqual(
            grammar["properties"]["owner"],
            {"type": ["string", "null"], "enum": ["ana", "bo", None]},
        )
        # And the pack still scores: case 2 and 3 answer null for that field.
        self.assertTrue(verify_pack(self.directory).passed_all)

    def test_the_emitted_grammar_is_one_jig_accepts(self):
        # load_pack runs check_schema, so an unsupported keyword would fail here.
        self.compile()
        self.assertEqual(
            load_pack(self.directory).nodes["label"].grammar["additionalProperties"],
            False,
        )


# ------------------------------------------------------------- the yaml subset traps


class TestYamlSubset(PackCase):
    def test_a_when_value_that_yaml_would_resolve_is_quoted_and_stays_a_string(self):
        """`when: {kind: no}` tests for False. The emitter has to quote it."""
        task = task_spec(
            fields=[
                FieldSpec("kind", "string", enum=["bug", "no"]),
                FieldSpec("owner", "string", optional=True),
                FieldSpec("score", "integer"),
            ]
        )
        plan = graph_plan(
            endings=["done", "dropped"],
            edges=[
                {"from": "label", "to": "dropped", "when": {"kind": "no"}},
                {"from": "label", "to": "detail"},
                {"from": "detail", "to": "done"},
            ],
        )
        self.compile(task=task, plan=plan)
        self.assertIn('when: {kind: "no"}', self.read("graph.yaml"))
        edge = [e for e in load_pack(self.directory).edges if e.target == "dropped"][0]
        self.assertEqual(edge.when, {"kind": "no"})

    def test_a_value_that_looks_like_a_number_stays_a_string(self):
        """`007` resolves to 7 in jig's subset, and `1.0` to a float."""
        plan = graph_plan(
            endings=["done", "dropped"],
            edges=[
                {"from": "label", "to": "dropped", "when": {"kind": "007"}},
                {"from": "label", "to": "detail"},
                {"from": "detail", "to": "done"},
            ],
        )
        self.compile(plan=plan)
        edge = [e for e in load_pack(self.directory).edges if e.target == "dropped"][0]
        self.assertEqual(edge.when, {"kind": "007"})

    def test_a_pack_name_that_yaml_would_read_as_a_boolean_survives(self):
        task = task_spec(name="off")
        self.compile(task=task)
        self.assertEqual(load_pack(self.directory).name, "off")

    def test_a_multi_line_description_reads_back_character_for_character(self):
        self.compile()
        manifest = parse_yaml(self.read("manifest.yaml"), "manifest.yaml")
        self.assertTrue(
            manifest["description"].startswith(
                "Triage a short internal note.\n\nLabel it, then pull the owner out."
            ),
            manifest["description"],
        )

    def test_a_description_full_of_yaml_punctuation_still_round_trips(self):
        nasty = 'colon: here # hash\n\ttab\n  - dash "quoted" \\ backslash'
        task = task_spec(description=nasty)
        self.compile(task=task)
        manifest = parse_yaml(self.read("manifest.yaml"), "manifest.yaml")
        self.assertTrue(manifest["description"].startswith(nasty), manifest["description"])

    def test_the_graph_is_the_graph_that_was_planned(self):
        self.compile()
        graph = parse_yaml(self.read("graph.yaml"), "graph.yaml")
        self.assertEqual(
            graph["nodes"]["done"], {"type": "end", "output": ["kind", "owner", "score"]}
        )
        self.assertEqual(
            graph["edges"],
            [{"from": "label", "to": "detail"}, {"from": "detail", "to": "done"}],
        )


# --------------------------------------------------------------------- the contract


class TestTheContractIsNotEdited(PackCase):
    def test_the_evalset_is_the_gold_cases_verbatim(self):
        cases = gold_cases()
        self.compile(task=task_spec(cases=cases))
        written = [
            json.loads(line) for line in self.read("evalset.jsonl").splitlines() if line
        ]
        self.assertEqual(written, cases)

    def test_a_pack_that_fails_its_gold_cases_is_reported_not_repaired(self):
        """The compiler may not move the target it is measured against.

        The script answers `chore` where the third gold case says `bug`, so the pack is
        genuinely wrong. What must not happen is the evalset changing to match.
        """
        cases = gold_cases()
        cases[2]["expect"]["kind"] = "bug"
        script = dict(SCRIPT)
        self.compile(task=task_spec(cases=cases), script=script)

        report = verify_pack(self.directory)
        self.assertFalse(report.passed_all)
        self.assertEqual((report.passed, report.total), (2, 3))
        written = [
            json.loads(line) for line in self.read("evalset.jsonl").splitlines() if line
        ]
        self.assertEqual(written[2]["expect"]["kind"], "bug")

    def test_compile_report_names_the_node_the_eval_blamed(self):
        cases = gold_cases()
        cases[2]["expect"]["kind"] = "bug"
        self.compile(task=task_spec(cases=cases))
        summary = compile_report(verify_pack(self.directory))
        self.assertIn("note_triage: 2/3 cases passed", summary)
        self.assertIn("blamed on: label=1", summary)
        self.assertIn("FAIL chore [label]", summary)

    def test_compile_report_stops_after_three_failures(self):
        cases = gold_cases() + [
            dict(case, name=case["name"] + "-again") for case in gold_cases()
        ]
        for case in cases:
            case["expect"] = dict(case["expect"], score=999)
        script = dict(SCRIPT)
        script["label step"] = SCRIPT["label step"] * 2
        script["detail thinking step"] = SCRIPT["detail thinking step"] * 2
        script["detail step"] = SCRIPT["detail step"] * 2
        self.compile(task=task_spec(cases=cases), script=script)
        summary = compile_report(verify_pack(self.directory))
        self.assertEqual(summary.count("  FAIL "), 3)
        self.assertIn("... and 3 more", summary)

    def test_a_case_naming_an_ending_the_plan_does_not_have_is_refused(self):
        cases = gold_cases()
        cases[1]["end"] = "escalated"
        with self.assertRaises(BuildError) as caught:
            self.compile(task=task_spec(cases=cases))
        self.assertIn("escalated", str(caught.exception))

    def test_an_empty_examples_file_is_refused_before_anything_is_written(self):
        with self.assertRaises(BuildError) as caught:
            self.compile(task=task_spec(cases=[]))
        self.assertIn("evalset", str(caught.exception))
        self.assertFalse(os.path.exists(self.directory))

    def test_verify_pack_refuses_a_pack_with_no_cases_to_verify(self):
        self.compile()
        open(os.path.join(self.directory, "evalset.jsonl"), "w").close()
        with self.assertRaises(BuildError):
            verify_pack(self.directory)


# ------------------------------------------------------------------- plan refusals


class TestPlanRefusals(PackCase):
    def test_a_field_no_node_writes_is_refused(self):
        plan = graph_plan(
            nodes=[
                NodePlan("label", ["kind"], "Label it."),
                NodePlan("detail", ["owner"], "Name the owner.", two_stage=True),
            ]
        )
        with self.assertRaises(BuildError) as caught:
            self.compile(plan=plan)
        self.assertIn("score", str(caught.exception))

    def test_a_field_two_nodes_write_is_refused(self):
        plan = graph_plan(
            nodes=[
                NodePlan("label", ["kind", "score"], "Label it."),
                NodePlan("detail", ["owner", "score"], "Detail it.", two_stage=True),
            ]
        )
        with self.assertRaises(BuildError) as caught:
            self.compile(plan=plan)
        self.assertIn("more than one node", str(caught.exception))

    def test_a_node_writing_a_field_the_examples_never_showed_is_refused(self):
        plan = graph_plan(
            nodes=[
                NodePlan("label", ["kind"], "Label it."),
                NodePlan("detail", ["owner", "score", "vibes"], "Detail it."),
            ]
        )
        with self.assertRaises(BuildError) as caught:
            self.compile(plan=plan)
        self.assertIn("vibes", str(caught.exception))

    def test_a_field_named_like_a_run_input_is_refused(self):
        """A generate node may not overwrite an input; that is a StateCollision at run."""
        task = task_spec(
            inputs=["note", "score"],
            fields=[
                FieldSpec("kind", "string", enum=["bug"]),
                FieldSpec("owner", "string", optional=True),
                FieldSpec("score", "integer"),
            ],
        )
        with self.assertRaises(BuildError) as caught:
            self.compile(task=task)
        self.assertIn("run input", str(caught.exception))

    def test_a_node_reading_a_name_nothing_produces_is_refused(self):
        plan = graph_plan(
            nodes=[
                NodePlan("label", ["kind"], "Label it."),
                NodePlan("detail", ["owner", "score"], "Detail it.", reads=["severity"]),
            ]
        )
        with self.assertRaises(BuildError) as caught:
            self.compile(plan=plan)
        self.assertIn("severity", str(caught.exception))

    def test_a_generate_node_with_no_prompt_is_refused(self):
        prompts = {"label": PROMPTS["label"]}
        with self.assertRaises(BuildError) as caught:
            self.compile(prompts=prompts)
        self.assertIn("detail", str(caught.exception))

    def test_a_think_template_for_a_single_stage_node_is_refused(self):
        """jig would never read it, so accepting it would be accepting a silent no-op."""
        prompts = dict(PROMPTS, **{"label.think": "thinking"})
        with self.assertRaises(BuildError) as caught:
            self.compile(prompts=prompts)
        self.assertIn("two_stage", str(caught.exception))

    def test_a_prompt_for_a_node_outside_the_plan_is_refused(self):
        prompts = dict(PROMPTS, **{"triage": "orphan"})
        with self.assertRaises(BuildError):
            self.compile(prompts=prompts)

    def test_a_node_name_that_is_not_a_plain_identifier_is_refused(self):
        plan = graph_plan(
            nodes=[
                NodePlan("../etc/passwd", ["kind"], "Label it."),
                NodePlan("detail", ["owner", "score"], "Detail it."),
            ],
            entry="../etc/passwd",
        )
        with self.assertRaises(BuildError) as caught:
            self.compile(plan=plan)
        self.assertIn("file names", str(caught.exception))

    def test_an_entry_outside_the_plan_is_refused(self):
        with self.assertRaises(BuildError):
            self.compile(plan=graph_plan(entry="nosuch"))

    def test_a_plan_with_no_endings_is_refused(self):
        with self.assertRaises(BuildError):
            self.compile(plan=graph_plan(endings=[]))

    def test_an_edge_out_of_an_ending_is_refused(self):
        plan = graph_plan(
            edges=[
                {"from": "label", "to": "detail"},
                {"from": "detail", "to": "done"},
                {"from": "done", "to": "label"},
            ]
        )
        with self.assertRaises(BuildError) as caught:
            self.compile(plan=plan)
        self.assertIn("ending", str(caught.exception))

    def test_a_node_with_no_outgoing_edge_is_refused_before_the_write(self):
        plan = graph_plan(edges=[{"from": "label", "to": "done"}])
        with self.assertRaises(BuildError) as caught:
            self.compile(plan=plan)
        self.assertIn("detail", str(caught.exception))
        self.assertFalse(os.path.exists(self.directory))

    def test_an_edge_pointing_at_nothing_is_refused(self):
        plan = graph_plan(
            edges=[{"from": "label", "to": "detail"}, {"from": "detail", "to": "review"}]
        )
        with self.assertRaises(BuildError) as caught:
            self.compile(plan=plan)
        self.assertIn("review", str(caught.exception))

    def test_an_empty_script_is_refused(self):
        with self.assertRaises(BuildError):
            self.compile(script={})


# ------------------------------------------------------------------ the target dir


class TestOverwrite(PackCase):
    def test_a_non_empty_directory_is_refused(self):
        os.makedirs(self.directory)
        with open(os.path.join(self.directory, "NOTES.md"), "w") as handle:
            handle.write("hand-tuned")
        with self.assertRaises(BuildError) as caught:
            self.compile()
        self.assertIn("overwrite=True", str(caught.exception))
        self.assertEqual(os.listdir(self.directory), ["NOTES.md"])

    def test_an_empty_directory_is_fine(self):
        os.makedirs(self.directory)
        self.compile()
        self.assertTrue(verify_pack(self.directory).passed_all)

    def test_overwrite_replaces_the_pack_and_drops_stale_prompts(self):
        self.compile()
        stale = os.path.join(self.directory, "prompts", "gone.txt")
        with open(stale, "w") as handle:
            handle.write("a node that no longer exists")
        self.compile(overwrite=True)
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(verify_pack(self.directory).passed_all)

    def test_overwrite_leaves_files_the_compiler_does_not_own(self):
        self.compile()
        keep = os.path.join(self.directory, "NOTES.md")
        with open(keep, "w") as handle:
            handle.write("hand-written")
        self.compile(overwrite=True)
        with open(keep) as handle:
            self.assertEqual(handle.read(), "hand-written")


# ------------------------------------------------------------------- token budgets


class TestBudgets(PackCase):
    def test_the_emit_budget_covers_the_longest_gold_answer(self):
        """A ceiling below the answer turns a correct generation into a rejected one."""
        long_owner = "a-very-long-owner-name-" * 8
        cases = gold_cases()
        cases[0]["expect"]["owner"] = long_owner
        script = dict(SCRIPT)
        script["detail step"] = [
            json.dumps({"owner": long_owner, "score": 3}),
            '{"owner": null, "score": 1}',
            '{"owner": null, "score": 0}',
        ]
        self.compile(task=task_spec(cases=cases), script=script)
        detail = load_pack(self.directory).nodes["detail"]
        self.assertGreater(detail.max_tokens, len(long_owner))
        self.assertTrue(verify_pack(self.directory).passed_all)


if __name__ == "__main__":
    unittest.main()
