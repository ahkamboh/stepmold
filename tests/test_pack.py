"""T2 — a JigPack loads, and every malformed pack fails with a specific error."""

import os
import shutil
import tempfile
import unittest

from jig.pack import (
    RESERVED_STATE_NAMES,
    EvalsetError,
    GrammarError,
    GraphError,
    ManifestError,
    MissingArtifactError,
    PackError,
    load_pack,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture(name):
    return os.path.join(FIXTURES, name)


class TestValidPack(unittest.TestCase):
    def setUp(self):
        self.pack = load_pack(fixture("valid_pack"))

    def test_manifest_fields(self):
        self.assertEqual(self.pack.name, "fixture_pack")
        self.assertEqual(self.pack.version, 2)
        self.assertEqual(self.pack.entry, "classify")
        self.assertEqual(self.pack.model, "fake:fakes/script.json")

    def test_every_node_is_loaded_with_its_type(self):
        self.assertEqual(
            {name: node.type for name, node in self.pack.nodes.items()},
            {
                "classify": "generate",
                "check": "assert",
                "done": "end",
                "give_up": "end",
            },
        )

    def test_generate_node_carries_its_prompt_and_grammar(self):
        node = self.pack.nodes["classify"]
        self.assertIn("{ticket}", node.prompt)
        self.assertEqual(node.grammar["required"], ["category"])
        self.assertEqual(
            node.grammar["properties"]["category"]["enum"],
            ["billing", "technical", "other"],
        )

    def test_generate_node_options(self):
        node = self.pack.nodes["classify"]
        self.assertTrue(node.two_stage)
        self.assertEqual(node.output, "classification")
        self.assertEqual(node.max_tokens, 64)
        self.assertEqual(node.retries, 2)
        self.assertEqual(node.on_fail, "give_up")

    def test_two_stage_node_picks_up_its_think_prompt(self):
        self.assertIn("Think step by step", self.pack.nodes["classify"].think_prompt)

    def test_defaults_are_applied_to_unset_options(self):
        node = self.pack.nodes["check"]
        self.assertFalse(node.two_stage)
        self.assertIsNone(node.output)
        self.assertEqual(node.expr, 'classification.category != "unknown"')

    def test_end_node_output_projection(self):
        self.assertEqual(self.pack.nodes["done"].output, ["classification"])
        self.assertIsNone(self.pack.nodes["give_up"].output)

    def test_edges_keep_order_and_conditions(self):
        self.assertEqual(
            [(e.source, e.target, e.when) for e in self.pack.edges],
            [
                ("classify", "check", None),
                ("check", "done", {"ok": True}),
                ("check", "give_up", None),
            ],
        )

    def test_edges_from_returns_only_that_node_in_declared_order(self):
        self.assertEqual(
            [e.target for e in self.pack.edges_from("check")], ["done", "give_up"]
        )
        self.assertEqual(self.pack.edges_from("done"), [])

    def test_evalset_is_loaded(self):
        self.assertEqual(len(self.pack.evalset), 2)
        self.assertEqual(self.pack.evalset[0].input, {"ticket": "I was charged twice"})
        self.assertEqual(self.pack.evalset[0].expect, {"category": "billing"})

    def test_max_steps_comes_from_the_graph(self):
        self.assertEqual(self.pack.max_steps, 20)

    def test_pack_remembers_where_it_was_loaded_from(self):
        self.assertEqual(self.pack.path, fixture("valid_pack"))


class TestMalformedPacks(unittest.TestCase):
    def _load(self, name, expected):
        with self.assertRaises(expected) as caught:
            load_pack(fixture(name))
        return str(caught.exception)

    def test_missing_entry_key(self):
        message = self._load("bad_no_entry", ManifestError)
        self.assertIn("entry", message)
        self.assertIn("manifest.yaml", message)

    def test_entry_names_a_node_that_does_not_exist(self):
        message = self._load("bad_missing_node", GraphError)
        self.assertIn("ghost", message)
        self.assertIn("entry", message)

    def test_dangling_edge(self):
        message = self._load("bad_dangling_edge", GraphError)
        self.assertIn("nowhere", message)
        self.assertIn("edge", message)

    def test_prompt_file_absent(self):
        message = self._load("bad_missing_prompt", MissingArtifactError)
        self.assertIn("prompts/classify.txt", message)

    def test_grammar_file_absent(self):
        message = self._load("bad_missing_grammar", MissingArtifactError)
        self.assertIn("grammars/classify.json", message)

    def test_unknown_node_type(self):
        message = self._load("bad_node_type", GraphError)
        self.assertIn("teleport", message)
        self.assertIn("weird", message)

    def test_grammar_is_not_an_enforceable_schema(self):
        message = self._load("bad_grammar_schema", GrammarError)
        self.assertIn("grammars/classify.json", message)
        self.assertIn("strng", message)

    def test_manifest_absent(self):
        message = self._load("bad_no_manifest", MissingArtifactError)
        self.assertIn("manifest.yaml", message)

    def test_malformed_evalset_case(self):
        message = self._load("bad_evalset", EvalsetError)
        self.assertIn("2", message)
        self.assertIn("expect", message)

    def test_pack_directory_absent(self):
        message = self._load("no_such_pack_anywhere", MissingArtifactError)
        self.assertIn("no_such_pack_anywhere", message)

    def test_every_error_is_a_pack_error(self):
        for name in (
            "bad_no_entry",
            "bad_missing_node",
            "bad_dangling_edge",
            "bad_missing_prompt",
            "bad_missing_grammar",
            "bad_node_type",
            "bad_grammar_schema",
            "bad_no_manifest",
            "bad_evalset",
        ):
            with self.assertRaises(PackError):
                load_pack(fixture(name))


class TestReservedStateNames(unittest.TestCase):
    """`scratchpad` is jig's own binding in a run's scope, so a pack may not commit there.

    `codegen.think` renders the think template with a `scratchpad` of its own, and the
    prompt labels that slot "your notes from thinking this through". A node whose output
    landed there would be handing the next think stage a value the model reads as its own
    reasoning — the most persuasive position in a prompt, filled by something that is not
    reasoning at all. Refusing it at load time means no run can reach that state.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _pack(self, output):
        directory = os.path.join(self.root, "p%d" % len(os.listdir(self.root)))
        os.makedirs(os.path.join(directory, "prompts"))
        os.makedirs(os.path.join(directory, "grammars"))
        _write(directory, "manifest.yaml", "name: p\nversion: 1\nentry: a\n")
        _write(directory, "graph.yaml",
               "nodes:\n"
               "  a:\n"
               "    type: generate\n"
               "    output: %s\n"
               "  z:\n"
               "    type: end\n"
               "edges:\n"
               "  - from: a\n"
               "    to: z\n" % output)
        _write(directory, os.path.join("prompts", "a.txt"), "go\n")
        _write(directory, os.path.join("grammars", "a.json"), '{"type": "object"}')
        return directory

    def test_the_reserved_names_are_public_so_callers_can_check_their_own_inputs(self):
        self.assertIn("scratchpad", RESERVED_STATE_NAMES)

    def test_a_node_that_commits_to_the_scratchpad_is_refused(self):
        with self.assertRaises(GraphError) as caught:
            load_pack(self._pack("scratchpad"))
        self.assertIn("scratchpad", str(caught.exception))

    def test_an_ordinary_output_name_still_loads(self):
        self.assertEqual(load_pack(self._pack("result")).nodes["a"].output, "result")


def _write(directory, relative, text):
    with open(os.path.join(directory, relative), "w") as handle:
        handle.write(text)
