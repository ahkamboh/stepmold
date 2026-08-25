"""T2 — a StepmoldPack loads, and every malformed pack fails with a specific error.

The `tool` half of this file is the same idea one step further out: a tool node is the
first node type whose mistakes cost a side effect, so every way of wiring one wrongly is
refused at load — before the run has done part of a job it cannot finish.
"""

import os
import shutil
import tempfile
import unittest

from stepmold.pack import (
    NODE_TYPES,
    RESERVED_STATE_NAMES,
    EvalsetError,
    GrammarError,
    GraphError,
    ManifestError,
    MissingArtifactError,
    PackError,
    ToolWiringError,
    check_tools,
    load_pack,
)
from stepmold.tools import ToolNotRegistered, ToolRegistry

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
    """`scratchpad` is stepmold's own binding in a run's scope, so a pack may not commit there.

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


# --------------------------------------------------------------------------- tools


TOOL_GRAPH = (
    "nodes:\n"
    "  fetch:\n"
    "    type: tool\n"
    "    tool: lookup_order\n"
    "    output: order\n"
    "    on_fail: needs_human\n"
    "    on_unsure: needs_human\n"
    "  done:\n"
    "    type: end\n"
    "    output: [order]\n"
    "  needs_human:\n"
    "    type: end\n"
    "edges:\n"
    "  - from: fetch\n"
    "    to: done\n"
)

OPEN_GRAMMAR = '{"type": "object"}'
ORDER_ID_GRAMMAR = (
    '{"type": "object", "properties": {"order_id": {"type": "string"}},'
    ' "required": ["order_id"], "additionalProperties": false}'
)


class PackOnDisk(unittest.TestCase):
    """Builds packs file by file, so a test can leave a file out on purpose."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def pack(self, graph, manifest=None, files=None, evalset=None):
        directory = os.path.join(self.root, "p%d" % len(os.listdir(self.root)))
        os.makedirs(directory)
        _write(directory, "manifest.yaml",
               manifest if manifest is not None else "name: t\nversion: 1\nentry: fetch\n")
        _write(directory, "graph.yaml", graph)
        if evalset is not None:
            _write(directory, "evalset.jsonl", evalset)
        for relative, text in (files or {}).items():
            full = os.path.join(directory, relative)
            if not os.path.isdir(os.path.dirname(full)):
                os.makedirs(os.path.dirname(full))
            with open(full, "w") as handle:
                handle.write(text)
        return directory


class TestToolNodes(PackOnDisk):
    """`tool` is a node type, and a wrong one is refused at load, not at step four."""

    def test_tool_is_one_of_the_node_types(self):
        self.assertIn("tool", NODE_TYPES)

    def test_a_tool_node_needs_no_prompt_and_no_grammar_on_disk(self):
        # The whole pack is two files. A tool node names a function the host registered;
        # there is nothing in the pack to read for it, so demanding prompts/fetch.txt
        # would refuse a pack that is perfectly well formed.
        pack = load_pack(self.pack(TOOL_GRAPH))
        node = pack.nodes["fetch"]
        self.assertEqual(node.type, "tool")
        self.assertEqual(node.tool, "lookup_order")
        self.assertIsNone(node.prompt)
        self.assertIsNone(node.grammar)

    def test_a_tool_node_carries_its_output_and_both_rescue_edges(self):
        node = load_pack(self.pack(TOOL_GRAPH)).nodes["fetch"]
        self.assertEqual(node.output, "order")
        self.assertEqual(node.on_fail, "needs_human")
        self.assertEqual(node.on_unsure, "needs_human")

    def test_tool_and_on_unsure_default_to_none_on_every_other_node(self):
        pack = load_pack(fixture("valid_pack"))
        for node in pack.nodes.values():
            self.assertIsNone(node.tool)
            self.assertIsNone(node.on_unsure)

    def test_a_tool_node_without_a_tool_key_is_refused(self):
        graph = TOOL_GRAPH.replace("    tool: lookup_order\n", "")
        with self.assertRaises(GraphError) as caught:
            load_pack(self.pack(graph))
        self.assertIn("fetch", str(caught.exception))
        self.assertIn("tool:", str(caught.exception))

    def test_an_empty_tool_name_is_refused(self):
        graph = TOOL_GRAPH.replace("tool: lookup_order", 'tool: ""')
        with self.assertRaises(GraphError):
            load_pack(self.pack(graph))

    def test_tool_on_a_generate_node_is_refused(self):
        graph = (
            "nodes:\n"
            "  fetch:\n"
            "    type: generate\n"
            "    tool: lookup_order\n"
            "  done:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        with self.assertRaises(GraphError) as caught:
            load_pack(self.pack(graph, files={"prompts/fetch.txt": "go\n",
                                              "grammars/fetch.json": OPEN_GRAMMAR}))
        message = str(caught.exception)
        self.assertIn("fetch", message)
        self.assertIn("generate", message)
        self.assertIn("lookup_order", message)

    def test_tool_on_an_assert_node_is_refused(self):
        graph = (
            "nodes:\n"
            "  fetch:\n"
            "    type: assert\n"
            "    expr: 1 == 1\n"
            "    tool: lookup_order\n"
            "  done:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        with self.assertRaises(GraphError):
            load_pack(self.pack(graph))

    def test_tool_on_an_end_node_is_refused(self):
        graph = (
            "nodes:\n"
            "  fetch:\n"
            "    type: end\n"
            "    tool: lookup_order\n"
        )
        with self.assertRaises(GraphError):
            load_pack(self.pack(graph))

    def test_generate_and_assert_keys_are_refused_on_a_tool_node_by_name(self):
        # Each of these would otherwise sit in the pack reading like something the
        # walker does — a retry ladder around a side effect, a grammar that shapes a
        # result nothing generated — and be ignored.
        for key, value in (
            ("prompt", "prompts/other.txt"),
            ("grammar", "grammars/other.json"),
            ("two_stage", "true"),
            ("retries", "3"),
            ("max_tokens", "64"),
            ("think_max_tokens", "64"),
            ("assert", "order != null"),
            ("expr", "1 == 1"),
        ):
            graph = TOOL_GRAPH.replace(
                "    tool: lookup_order\n",
                "    tool: lookup_order\n    %s: %s\n" % (key, value))
            with self.assertRaises(GraphError) as caught:
                load_pack(self.pack(graph))
            message = str(caught.exception)
            self.assertIn(repr(key), message, "%r must be refused by name" % key)
            self.assertIn("fetch", message)

    def test_every_forbidden_key_is_reported_at_once(self):
        graph = TOOL_GRAPH.replace(
            "    tool: lookup_order\n",
            "    tool: lookup_order\n    retries: 3\n    two_stage: true\n")
        with self.assertRaises(GraphError) as caught:
            load_pack(self.pack(graph))
        self.assertIn("'retries'", str(caught.exception))
        self.assertIn("'two_stage'", str(caught.exception))

    def test_a_tool_nodes_output_must_be_a_single_state_key(self):
        graph = TOOL_GRAPH.replace("output: order\n", "output: [order]\n")
        with self.assertRaises(GraphError) as caught:
            load_pack(self.pack(graph))
        self.assertIn("fetch", str(caught.exception))
        self.assertIn("output", str(caught.exception))

    def test_a_tool_node_may_omit_output_and_merge_what_the_tool_returns(self):
        graph = TOOL_GRAPH.replace("    output: order\n", "").replace(
            "output: [order]", "output: [order_total]")
        self.assertIsNone(load_pack(self.pack(graph)).nodes["fetch"].output)

    def test_a_tool_node_still_needs_an_outgoing_edge(self):
        graph = TOOL_GRAPH.replace("edges:\n  - from: fetch\n    to: done\n", "edges: []\n")
        with self.assertRaises(GraphError) as caught:
            load_pack(self.pack(graph))
        self.assertIn("fetch", str(caught.exception))

    def test_on_unsure_must_name_a_defined_node(self):
        graph = TOOL_GRAPH.replace("on_unsure: needs_human", "on_unsure: nowhere")
        with self.assertRaises(GraphError) as caught:
            load_pack(self.pack(graph))
        self.assertIn("nowhere", str(caught.exception))
        self.assertIn("on_unsure", str(caught.exception))

    def test_the_unknown_type_message_offers_tool(self):
        graph = TOOL_GRAPH.replace("type: tool", "type: teleport")
        with self.assertRaises(GraphError) as caught:
            load_pack(self.pack(graph))
        self.assertIn("tool", str(caught.exception))

    def test_a_tool_node_may_not_commit_to_a_reserved_name(self):
        graph = TOOL_GRAPH.replace("output: order\n", "output: scratchpad\n")
        with self.assertRaises(GraphError) as caught:
            load_pack(self.pack(graph))
        self.assertIn("scratchpad", str(caught.exception))


def _registry():
    """The host's side of the contract: two tools, each declaring what it touches."""
    registry = ToolRegistry()

    @registry.register("lookup_order", reads=["order_id"], writes=["order_total"])
    def lookup_order(order_id):
        return {"order_total": 0}

    @registry.register("refund", reads=["order_total"], writes=["refund_id"])
    def refund(order_total):
        return {"refund_id": "r1"}

    return registry


ONE_CASE = '{"input": {"order_id": "A-1"}, "expect": {}}\n'
NO_INPUT_CASE = '{"input": {}, "expect": {}}\n'


class TestToolRegistration(PackOnDisk):
    """A pack can only call what the host already registered — checked before the run."""

    def test_without_a_registry_the_pack_still_loads(self):
        # `stepmold validate` on a machine whose tools live in someone else's process must
        # not be a hard error: a check that cannot run is not a check that failed.
        pack = load_pack(self.pack(TOOL_GRAPH, evalset=ONE_CASE))
        self.assertEqual(pack.nodes["fetch"].tool, "lookup_order")

    def test_an_unregistered_tool_is_refused_at_load(self):
        graph = TOOL_GRAPH.replace("tool: lookup_order", "tool: wire_money")
        with self.assertRaises(ToolNotRegistered) as caught:
            load_pack(self.pack(graph, evalset=ONE_CASE), tools=_registry())
        message = str(caught.exception)
        self.assertIn("wire_money", message)
        self.assertIn("fetch", message)
        self.assertIn("lookup_order", message)  # says what *is* available

    def test_a_registered_tool_loads(self):
        pack = load_pack(self.pack(TOOL_GRAPH, evalset=ONE_CASE), tools=_registry())
        self.assertEqual(pack.nodes["fetch"].tool, "lookup_order")

    def test_a_pack_with_no_tool_nodes_ignores_the_registry(self):
        pack = load_pack(fixture("valid_pack"), tools=ToolRegistry())
        self.assertEqual(pack.name, "fixture_pack")

    def test_a_plain_dict_is_not_a_registry(self):
        # `{}.get(name, node_name)` answers with the node name and would sail straight
        # past the registration check, so the shape is refused rather than trusted.
        with self.assertRaises(TypeError):
            load_pack(self.pack(TOOL_GRAPH, evalset=ONE_CASE), tools={"lookup_order": 1})

    def test_check_tools_runs_on_an_already_loaded_pack(self):
        pack = load_pack(self.pack(TOOL_GRAPH, evalset=ONE_CASE))
        self.assertIsNone(check_tools(pack, _registry()))


class TestToolReadsAreSatisfiable(PackOnDisk):
    """The field a tool reads must be one this graph will actually have by then.

    This is the check that turns "the run died at step 4 because the tool wanted a field
    nobody wrote" into "this pack is wired wrong, here is the field".
    """

    def _chain(self, writer, entry="classify", grammar=ORDER_ID_GRAMMAR,
               evalset=NO_INPUT_CASE):
        graph = (
            "nodes:\n"
            "  classify:\n"
            "    type: generate\n"
            + writer +
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "    output: order\n"
            "  done:\n"
            "    type: end\n"
            "    output: [order]\n"
            "edges:\n"
            "  - from: classify\n"
            "    to: fetch\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        return self.pack(
            graph,
            manifest="name: t\nversion: 1\nentry: %s\n" % entry,
            files={"prompts/classify.txt": "go\n", "grammars/classify.json": grammar},
            evalset=evalset,
        )

    def test_a_field_nobody_writes_is_named(self):
        directory = self.pack(TOOL_GRAPH,
                              evalset='{"input": {"customer": "c"}, "expect": {}}\n')
        with self.assertRaises(ToolWiringError) as caught:
            load_pack(directory, tools=_registry())
        message = str(caught.exception)
        self.assertIn("'order_id'", message)   # the field, by name
        self.assertIn("fetch", message)        # the node
        self.assertIn("lookup_order", message)  # the tool
        self.assertIn("customer", message)     # what the pack does declare

    def test_a_wiring_error_is_a_pack_error(self):
        directory = self.pack(TOOL_GRAPH, evalset=NO_INPUT_CASE)
        for expected in (ToolWiringError, GraphError, PackError):
            with self.assertRaises(expected):
                load_pack(directory, tools=_registry())

    def test_a_run_input_from_the_evalset_satisfies_it(self):
        load_pack(self.pack(TOOL_GRAPH, evalset=ONE_CASE), tools=_registry())

    def test_a_run_input_declared_in_the_manifest_satisfies_it(self):
        directory = self.pack(
            TOOL_GRAPH, manifest="name: t\nversion: 1\nentry: fetch\ninputs: [order_id]\n")
        load_pack(directory, tools=_registry())

    def test_an_earlier_nodes_output_key_satisfies_it(self):
        load_pack(self._chain("    output: order_id\n"), tools=_registry())

    def test_an_earlier_merge_mode_grammars_property_satisfies_it(self):
        # No `output:`, so the grammar's property names *are* state keys.
        load_pack(self._chain(""), tools=_registry())

    def test_a_later_node_writing_it_does_not_satisfy_it(self):
        # Same two nodes, other way round: the field arrives one step too late.
        graph = (
            "nodes:\n"
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "    output: order\n"
            "  classify:\n"
            "    type: generate\n"
            "    output: order_id\n"
            "  done:\n"
            "    type: end\n"
            "    output: [order]\n"
            "edges:\n"
            "  - from: fetch\n"
            "    to: classify\n"
            "  - from: classify\n"
            "    to: done\n"
        )
        directory = self.pack(
            graph,
            files={"prompts/classify.txt": "go\n", "grammars/classify.json": OPEN_GRAMMAR},
            evalset=NO_INPUT_CASE,
        )
        with self.assertRaises(ToolWiringError) as caught:
            load_pack(directory, tools=_registry())
        self.assertIn("'order_id'", str(caught.exception))

    def test_one_tool_can_read_what_an_earlier_tool_writes(self):
        graph = (
            "nodes:\n"
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "  pay:\n"
            "    type: tool\n"
            "    tool: refund\n"
            "    output: receipt\n"
            "  done:\n"
            "    type: end\n"
            "    output: [receipt]\n"
            "edges:\n"
            "  - from: fetch\n"
            "    to: pay\n"
            "  - from: pay\n"
            "    to: done\n"
        )
        # `fetch` has no `output:`, so what it merges into state is the tool's `writes`.
        load_pack(self.pack(graph, evalset=ONE_CASE), tools=_registry())

    def test_a_field_lost_to_an_on_fail_divert_does_not_count(self):
        # `fetch` is only ever reached when `classify` burned its ladder — and a node
        # that failed committed nothing, so `order_id` is not in state on that path.
        graph = (
            "nodes:\n"
            "  classify:\n"
            "    type: generate\n"
            "    output: order_id\n"
            "    on_fail: fetch\n"
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "    output: order\n"
            "  done:\n"
            "    type: end\n"
            "    output: [order]\n"
            "edges:\n"
            "  - from: classify\n"
            "    to: done\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        directory = self.pack(
            graph,
            manifest="name: t\nversion: 1\nentry: classify\n",
            files={"prompts/classify.txt": "go\n", "grammars/classify.json": OPEN_GRAMMAR},
            evalset=NO_INPUT_CASE,
        )
        with self.assertRaises(ToolWiringError) as caught:
            load_pack(directory, tools=_registry())
        self.assertIn("'order_id'", str(caught.exception))

    def test_a_field_from_before_the_failed_node_still_counts(self):
        # The failed node's own output is gone; everything written before it is not.
        graph = (
            "nodes:\n"
            "  seed:\n"
            "    type: generate\n"
            "    output: order_id\n"
            "  classify:\n"
            "    type: generate\n"
            "    output: kind\n"
            "    on_fail: fetch\n"
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "    output: order\n"
            "  done:\n"
            "    type: end\n"
            "    output: [order]\n"
            "edges:\n"
            "  - from: seed\n"
            "    to: classify\n"
            "  - from: classify\n"
            "    to: done\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        load_pack(
            self.pack(
                graph,
                manifest="name: t\nversion: 1\nentry: seed\n",
                files={"prompts/seed.txt": "go\n", "grammars/seed.json": OPEN_GRAMMAR,
                       "prompts/classify.txt": "go\n",
                       "grammars/classify.json": OPEN_GRAMMAR},
                evalset=NO_INPUT_CASE,
            ),
            tools=_registry(),
        )

    def test_a_field_written_on_an_on_unsure_path_still_counts(self):
        # Unsure about a value is not the same as not having produced one, so a tool
        # reached by `on_unsure` may still read what that node committed.
        graph = (
            "nodes:\n"
            "  classify:\n"
            "    type: generate\n"
            "    output: order_id\n"
            "    on_unsure: fetch\n"
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "    output: order\n"
            "  done:\n"
            "    type: end\n"
            "    output: [order]\n"
            "edges:\n"
            "  - from: classify\n"
            "    to: done\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        load_pack(
            self.pack(
                graph,
                manifest="name: t\nversion: 1\nentry: classify\n",
                files={"prompts/classify.txt": "go\n",
                       "grammars/classify.json": OPEN_GRAMMAR},
                evalset=NO_INPUT_CASE,
            ),
            tools=_registry(),
        )

    def test_a_pack_that_declares_no_inputs_is_left_alone(self):
        # No evalset and no manifest `inputs:` — the pack says nothing about what the
        # caller passes, so an unwritten field is unproven rather than wrong.
        load_pack(self.pack(TOOL_GRAPH), tools=_registry())

    def test_an_open_grammar_upstream_keeps_the_check_quiet(self):
        # A generate node in merge mode with no declared properties may write anything,
        # so nothing here can be called missing.
        load_pack(self._chain("", grammar=OPEN_GRAMMAR), tools=_registry())

    def test_a_tool_that_reads_nothing_needs_nothing(self):
        registry = ToolRegistry()
        registry.add("lookup_order", lambda: {"order_total": 1}, writes=["order_total"])
        load_pack(self.pack(TOOL_GRAPH, evalset=NO_INPUT_CASE), tools=registry)

    def test_an_unreachable_tool_node_is_not_reported(self):
        # It can never run, which is a graph problem and not this check's business.
        graph = (
            "nodes:\n"
            "  start:\n"
            "    type: assert\n"
            "    expr: 1 == 1\n"
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "    output: order\n"
            "  done:\n"
            "    type: end\n"
            "    output: [order]\n"
            "edges:\n"
            "  - from: start\n"
            "    to: done\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        directory = self.pack(graph,
                              manifest="name: t\nversion: 1\nentry: start\n",
                              evalset=NO_INPUT_CASE)
        load_pack(directory, tools=_registry())


class TestManifestInputs(PackOnDisk):
    """`inputs:` in the manifest is read, so its shape is checked rather than assumed."""

    def test_a_list_of_names_is_accepted(self):
        directory = self.pack(
            TOOL_GRAPH,
            manifest="name: t\nversion: 1\nentry: fetch\ninputs: [order_id, customer]\n")
        self.assertEqual(load_pack(directory).manifest["inputs"], ["order_id", "customer"])

    def test_a_non_list_is_refused(self):
        directory = self.pack(
            TOOL_GRAPH, manifest="name: t\nversion: 1\nentry: fetch\ninputs: order_id\n")
        with self.assertRaises(ManifestError) as caught:
            load_pack(directory)
        self.assertIn("inputs", str(caught.exception))

    def test_the_manifest_and_the_evalset_are_both_read(self):
        registry = ToolRegistry()
        registry.add("lookup_order", lambda order_id, customer: {"order_total": 1},
                     reads=["order_id", "customer"], writes=["order_total"])
        directory = self.pack(
            TOOL_GRAPH,
            manifest="name: t\nversion: 1\nentry: fetch\ninputs: [customer]\n",
            evalset=ONE_CASE)
        load_pack(directory, tools=registry)


class TestToolCheckOnOddGraphs(PackOnDisk):
    """The check reports wiring; it never becomes the thing that breaks the load."""

    def test_an_upstream_output_of_the_wrong_shape_does_not_crash_the_check(self):
        # `output: [order_id]` on a generate node is a shape the CLI refuses separately.
        # Reaching it from here must produce a verdict, not a TypeError.
        graph = (
            "nodes:\n"
            "  classify:\n"
            "    type: generate\n"
            "    output: [order_id]\n"
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "    output: order\n"
            "  done:\n"
            "    type: end\n"
            "    output: [order]\n"
            "edges:\n"
            "  - from: classify\n"
            "    to: fetch\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        directory = self.pack(
            graph,
            manifest="name: t\nversion: 1\nentry: classify\n",
            files={"prompts/classify.txt": "go\n", "grammars/classify.json": OPEN_GRAMMAR},
            evalset=NO_INPUT_CASE,
        )
        load_pack(directory, tools=_registry())

    def test_an_end_node_upstream_writes_nothing_and_breaks_nothing(self):
        graph = (
            "nodes:\n"
            "  classify:\n"
            "    type: generate\n"
            "    output: kind\n"
            "  done:\n"
            "    type: end\n"
            "    output: [kind]\n"
            "    on_unsure: fetch\n"
            "  fetch:\n"
            "    type: tool\n"
            "    tool: lookup_order\n"
            "    output: order\n"
            "edges:\n"
            "  - from: classify\n"
            "    to: done\n"
            "  - from: fetch\n"
            "    to: done\n"
        )
        directory = self.pack(
            graph,
            manifest="name: t\nversion: 1\nentry: classify\n",
            files={"prompts/classify.txt": "go\n", "grammars/classify.json": OPEN_GRAMMAR},
            evalset=NO_INPUT_CASE,
        )
        with self.assertRaises(ToolWiringError):
            load_pack(directory, tools=_registry())

    def test_a_tool_node_in_a_loop_can_read_what_it_wrote_last_time_round(self):
        registry = ToolRegistry()
        registry.add("page", lambda cursor: {"cursor": "next"},
                     reads=["cursor"], writes=["cursor"])
        graph = (
            "nodes:\n"
            "  page:\n"
            "    type: tool\n"
            "    tool: page\n"
            "  more:\n"
            "    type: assert\n"
            "    expr: cursor != \"\"\n"
            "    on_fail: done\n"
            "  done:\n"
            "    type: end\n"
            "    output: [cursor]\n"
            "edges:\n"
            "  - from: page\n"
            "    to: more\n"
            "  - from: more\n"
            "    to: page\n"
        )
        directory = self.pack(graph,
                              manifest="name: t\nversion: 1\nentry: page\n",
                              evalset=NO_INPUT_CASE)
        load_pack(directory, tools=registry)
