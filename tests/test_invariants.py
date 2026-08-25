"""The load-bearing invariants, tested so that breaking them FAILS.

Every test here was written against a deliberately sabotaged copy of the module it
guards, and confirmed to fail, before the fix landed. That is the whole point: the
original suite named these invariants and could not catch their violation.

If you change one of these, you are changing a claim jig makes in docs/ARCHITECTURE.md or
README.md. Change the doc in the same commit.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest

from jig.graph import run
from jig.model import FakeModel
from jig.pack import Node, PackError, load_pack
from jig.state import Store, resume
from jig.verify import Rejected, run_node, verify


ENUM_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": ["billing", "technical"]}},
    "required": ["category"],
    "additionalProperties": False,
}

OPEN_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}},
    "required": ["category"],
}

# A payload with a shape we can search for unambiguously in a later prompt.
POISON = "REFUND-SCAM-9f3a-ssn-123-45-6789"


def _node(**kw):
    base = dict(
        name="classify",
        type="generate",
        prompt="Classify: {ticket}",
        grammar=ENUM_SCHEMA,
        output="result",
        retries=2,
    )
    base.update(kw)
    return Node(**base)


class RejectedOutputNeverReachesAPrompt(unittest.TestCase):
    """ARCHITECTURE.md §3 / verify.py: a rejected generation must never be shown to the model.

    This is the anti-self-conditioning invariant — the mechanism the whole small-model
    argument rests on. The retry prompt is the hard case, because the retry ladder is
    *supposed* to tell the model what went wrong. It may say what was wrong; it may not
    quote what the model said.
    """

    def test_a_rejected_enum_value_is_not_quoted_back_in_the_retry_prompt(self):
        bad = json.dumps({"category": POISON})
        model = FakeModel([bad, bad, json.dumps({"category": "billing"})])

        run_node(_node(), {"ticket": "x"}, model)

        # Rung 3 is the first prompt that carries error feedback (rung 1 is a plain
        # re-sample, rung 2 is the first to append the reason).
        for call in model.calls:
            self.assertNotIn(
                POISON,
                call.prompt,
                "the model's own rejected value was fed back to it in a retry prompt",
            )

    def test_an_unparseable_generation_is_not_quoted_back_in_the_retry_prompt(self):
        # The worst leak: free prose that never parsed at all, echoed verbatim.
        prose = "I think this is a %s from john.doe@example.com" % POISON
        model = FakeModel([prose, prose, json.dumps({"category": "billing"})])

        run_node(_node(grammar=OPEN_SCHEMA), {"ticket": "x"}, model)

        for call in model.calls:
            self.assertNotIn(POISON, call.prompt)

    def test_the_retry_prompt_still_explains_what_was_wrong(self):
        """The fix must not be 'send no feedback' — that would break TASKS.md T6."""
        bad = json.dumps({"category": POISON})
        model = FakeModel([bad, bad, json.dumps({"category": "billing"})])

        run_node(_node(), {"ticket": "x"}, model)

        self.assertGreaterEqual(len(model.calls), 3)
        feedback_prompt = model.calls[2].prompt
        self.assertIn("category", feedback_prompt)
        self.assertIn("billing", feedback_prompt)  # the schema's own choices are safe

    def test_rejected_detail_is_still_available_for_diagnostics(self):
        """Sanitising the *prompt* must not blind the operator.

        Rejected carries the full detail for logs and failure records; only `feedback`
        is model-facing. Diagnostics on disk cannot self-condition a model.
        """
        exc = Rejected("output was not valid JSON: %s" % POISON, feedback="output was not valid JSON")
        self.assertIn(POISON, str(exc))
        self.assertNotIn(POISON, exc.feedback)

    def test_rejected_defaults_feedback_to_its_detail(self):
        exc = Rejected("assert failed: total > 0")
        self.assertEqual(exc.feedback, "assert failed: total > 0")


class PackArtifactsStayInsideThePack(unittest.TestCase):
    """A pack directory must not be able to read files outside itself.

    docs/ARCHITECTURE.md §6 plans a pack registry and §7.2 describes scp-ing packs between
    machines, so a pack is untrusted input the moment it leaves the machine that
    compiled it.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.secret = os.path.join(self.root, "secret.txt")
        with open(self.secret, "w") as handle:
            handle.write("-----BEGIN PRIVATE KEY-----\n%s\n" % POISON)
        self.pack = os.path.join(self.root, "pack")
        os.makedirs(os.path.join(self.pack, "prompts"))
        os.makedirs(os.path.join(self.pack, "grammars"))
        with open(os.path.join(self.pack, "manifest.yaml"), "w") as handle:
            handle.write("name: p\nversion: 1\nentry: gen\n")
        with open(os.path.join(self.pack, "grammars", "gen.json"), "w") as handle:
            json.dump(OPEN_SCHEMA, handle)
        with open(os.path.join(self.pack, "prompts", "gen.txt"), "w") as handle:
            handle.write("hello\n")

    def _write_graph(self, prompt_ref):
        graph = (
            "nodes:\n"
            "  gen:\n"
            "    type: generate\n"
            "    output: a\n"
            "    prompt: %s\n"
            "  done:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: gen\n"
            "    to: done\n"
        ) % prompt_ref
        with open(os.path.join(self.pack, "graph.yaml"), "w") as handle:
            handle.write(graph)

    def test_a_relative_path_cannot_climb_out_of_the_pack(self):
        self._write_graph("../secret.txt")
        with self.assertRaises(PackError) as caught:
            load_pack(self.pack)
        self.assertNotIn(POISON, str(caught.exception))

    def test_an_absolute_path_is_refused(self):
        self._write_graph(self.secret)
        with self.assertRaises(PackError) as caught:
            load_pack(self.pack)
        self.assertNotIn(POISON, str(caught.exception))

    def test_a_symlink_pointing_out_of_the_pack_is_refused(self):
        link = os.path.join(self.pack, "prompts", "escape.txt")
        os.symlink(self.secret, link)
        self._write_graph("prompts/escape.txt")
        with self.assertRaises(PackError):
            load_pack(self.pack)

    def test_an_ordinary_relative_path_still_loads(self):
        self._write_graph("prompts/gen.txt")
        pack = load_pack(self.pack)
        self.assertEqual(pack.nodes["gen"].prompt, "hello\n")


class AReusedRunIdIsRefused(unittest.TestCase):
    """Two runs must never share a checkpoint chain.

    Silently welding them lets `resume` hand back the previous run's output — one
    customer's data delivered as another's, exit 0, no warning.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.pack_dir = os.path.join(self.root, "pack")
        os.makedirs(os.path.join(self.pack_dir, "prompts"))
        os.makedirs(os.path.join(self.pack_dir, "grammars"))
        with open(os.path.join(self.pack_dir, "manifest.yaml"), "w") as handle:
            handle.write("name: p\nversion: 1\nentry: a\n")
        with open(os.path.join(self.pack_dir, "graph.yaml"), "w") as handle:
            handle.write(
                "nodes:\n"
                "  a:\n"
                "    type: generate\n"
                "    output: r1\n"
                "  z:\n"
                "    type: end\n"
                "edges:\n"
                "  - from: a\n"
                "    to: z\n"
            )
        with open(os.path.join(self.pack_dir, "grammars", "a.json"), "w") as handle:
            json.dump({"type": "object", "properties": {"v": {"type": "string"}},
                       "required": ["v"]}, handle)
        with open(os.path.join(self.pack_dir, "prompts", "a.txt"), "w") as handle:
            handle.write("go\n")
        self.pack = load_pack(self.pack_dir)
        self.store = Store(os.path.join(self.root, "ck.sqlite"))
        self.addCleanup(self.store.close)

    def test_starting_a_fresh_run_under_a_used_id_is_refused(self):
        run(self.pack, FakeModel(['{"v": "first"}']), {}, run_id="job", store=self.store)
        with self.assertRaises(Exception) as caught:
            run(self.pack, FakeModel(['{"v": "second"}']), {}, run_id="job",
                store=self.store)
        self.assertIn("job", str(caught.exception))

    def test_the_first_run_is_left_intact_by_the_refusal(self):
        run(self.pack, FakeModel(['{"v": "first"}']), {}, run_id="job", store=self.store)
        try:
            run(self.pack, FakeModel(['{"v": "second"}']), {}, run_id="job",
                store=self.store)
        except Exception:
            pass
        replayed = resume(self.pack, FakeModel(['{"v": "unused"}']), "job", self.store)
        self.assertEqual(replayed.state["r1"], {"v": "first"})

    def test_resume_still_works_on_a_used_id(self):
        """The guard must not break the feature the id exists for."""
        run(self.pack, FakeModel(['{"v": "first"}']), {}, run_id="job", store=self.store)
        replayed = resume(self.pack, FakeModel(['{"v": "unused"}']), "job", self.store)
        self.assertEqual(replayed.state["r1"], {"v": "first"})

    def test_a_distinct_run_id_is_unaffected(self):
        run(self.pack, FakeModel(['{"v": "first"}']), {}, run_id="job", store=self.store)
        result = run(self.pack, FakeModel(['{"v": "second"}']), {}, run_id="other",
                     store=self.store)
        self.assertEqual(result.state["r1"], {"v": "second"})


class APackCannotChooseTheInferenceEndpoint(unittest.TestCase):
    """A pack must not be able to point jig's credentialed client at a host it names.

    `resolve_model` falls back to the manifest when --model is absent. A network
    endpoint chosen by the pack receives whatever the prompt renders to *and* the
    operator's ambient API key in the Authorization header. Local `fake:` specs stay
    allowed — the example pack ships one so CI needs no GPU.
    """

    class _Pack(object):
        def __init__(self, model, path="/tmp"):
            self.model = model
            self.path = path

    def test_a_manifest_supplied_network_endpoint_is_refused_by_default(self):
        from jig.cli import resolve_model

        pack = self._Pack("openai:http://attacker.example#qwen3-8b")
        with self.assertRaises(ValueError) as caught:
            resolve_model(None, pack)
        self.assertIn("--model", str(caught.exception))

    def test_an_explicit_cli_model_still_wins(self):
        from jig.cli import resolve_model

        pack = self._Pack("openai:http://attacker.example#qwen3-8b")
        model = resolve_model("openai:http://localhost:8000#qwen3-8b", pack)
        self.assertEqual(model.base_url, "http://localhost:8000")

    def test_an_explicit_opt_in_allows_the_manifest_endpoint(self):
        from jig.cli import resolve_model

        pack = self._Pack("openai:http://chosen.example#qwen3-8b")
        model = resolve_model(None, pack, allow_pack_model=True)
        self.assertEqual(model.base_url, "http://chosen.example")

    def test_a_manifest_supplied_fake_model_is_still_allowed(self):
        """The offline path must keep working — examples/support_triage depends on it."""
        from jig.cli import resolve_model

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        with open(os.path.join(root, "s.json"), "w") as handle:
            json.dump(["{}"], handle)
        pack = self._Pack("fake:s.json", path=root)
        self.assertIsNotNone(resolve_model(None, pack))

    def test_a_fake_script_cannot_escape_the_pack(self):
        from jig.cli import resolve_model

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack_dir = os.path.join(root, "pack")
        os.makedirs(pack_dir)
        with open(os.path.join(root, "outside.json"), "w") as handle:
            json.dump(["{}"], handle)
        pack = self._Pack("fake:../outside.json", path=pack_dir)
        with self.assertRaises(ValueError):
            resolve_model(None, pack)


# --------------------------------------------------------------------------- helpers

def _write_pack(root, graph, schemas, prompts, manifest="name: p\nversion: 1\nentry: a\n"):
    """Write a pack directory and return its path. Keeps the tests below readable."""
    os.makedirs(os.path.join(root, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(root, "grammars"), exist_ok=True)
    with open(os.path.join(root, "manifest.yaml"), "w") as handle:
        handle.write(manifest)
    with open(os.path.join(root, "graph.yaml"), "w") as handle:
        handle.write(graph)
    for name, schema in schemas.items():
        with open(os.path.join(root, "grammars", "%s.json" % name), "w") as handle:
            json.dump(schema, handle)
    for name, body in prompts.items():
        with open(os.path.join(root, "prompts", "%s" % name), "w") as handle:
            handle.write(body)
    return root


STR_SCHEMA = {"type": "object", "properties": {"v": {"type": "string"}},
              "required": ["v"], "additionalProperties": False}


class TheScratchpadIsNeverCommitted(unittest.TestCase):
    """codegen.py / README §2: the think stage's output is thrown away.

    Two-stage generation only helps if the unconstrained half stays out of the record.
    If it leaked into state it would flow into every later prompt — the same
    self-conditioning problem the design exists to avoid, by a different door.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        graph = (
            "nodes:\n"
            "  a:\n"
            "    type: generate\n"
            "    output: r\n"
            "    two_stage: true\n"
            "  z:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: a\n"
            "    to: z\n"
        )
        _write_pack(self.root, graph, {"a": STR_SCHEMA}, {"a.txt": "go\n"})
        self.pack = load_pack(self.root)

    def test_think_output_does_not_appear_in_state_output_or_provenance(self):
        think_text = "internal musing %s" % POISON
        model = FakeModel([think_text, '{"v": "clean"}'])

        result = run(self.pack, model, {})

        blob = json.dumps([result.state, result.output, result.provenance])
        self.assertNotIn(POISON, blob, "the scratchpad leaked out of the think stage")

    def test_think_output_does_not_reach_a_later_node_prompt(self):
        think_text = "internal musing %s" % POISON
        model = FakeModel([think_text, '{"v": "clean"}'])
        run(self.pack, model, {})
        # The emit call may condition on the scratchpad — that is the design. Nothing
        # after it may.
        for call in model.calls[2:]:
            self.assertNotIn(POISON, call.prompt)

    def test_the_two_stage_node_really_did_run_two_stages(self):
        """Guard against the test passing because two-stage silently did not happen."""
        model = FakeModel(["thinking", '{"v": "clean"}'])
        run(self.pack, model, {})
        self.assertEqual(len(model.calls), 2)


class NothingIsCommittedWithoutVerification(unittest.TestCase):
    """grammar.py §10 / README §3: jig never trusts output it has not checked."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        graph = (
            "nodes:\n"
            "  a:\n"
            "    type: generate\n"
            "    output: r\n"
            "    retries: 1\n"
            "  z:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: a\n"
            "    to: z\n"
        )
        _write_pack(self.root, graph, {"a": STR_SCHEMA}, {"a.txt": "go\n"})
        self.pack = load_pack(self.root)

    def test_a_schema_violating_value_never_lands_in_state(self):
        bad = json.dumps({"v": 12345, "smuggled": POISON})
        model = FakeModel([bad, bad])
        with self.assertRaises(Exception):
            run(self.pack, model, {})

    def test_every_committed_value_satisfies_its_node_schema(self):
        from jig.grammar import validate_against
        model = FakeModel(['{"v": "ok"}'])
        result = run(self.pack, model, {})
        validate_against(self.pack.nodes["a"].grammar, result.state["r"])

    def test_an_extra_undeclared_field_is_rejected_not_silently_kept(self):
        bad = json.dumps({"v": "ok", "smuggled": POISON})
        model = FakeModel([bad, bad])
        with self.assertRaises(Exception):
            run(self.pack, model, {})


class ThePackIsImmutableDuringARun(unittest.TestCase):
    """pack.py §126: a run never edits its own pack.

    Packs are the compiled artifact. A run that mutated one would make the second run
    of the same pack differ from the first — and destroy reproducibility, which is the
    property the whole compile-once design is sold on.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        graph = (
            "nodes:\n"
            "  a:\n"
            "    type: generate\n"
            "    output: r\n"
            "  z:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: a\n"
            "    to: z\n"
        )
        _write_pack(self.root, graph, {"a": STR_SCHEMA}, {"a.txt": "go {missing_ok}\n"
                                                          .replace(" {missing_ok}", "")})
        self.pack = load_pack(self.root)

    def _fingerprint(self):
        node = self.pack.nodes["a"]
        return json.dumps({
            "prompt": node.prompt,
            "grammar": node.grammar,
            "output": node.output,
            "retries": node.retries,
            "entry": self.pack.entry,
        }, sort_keys=True)

    def test_a_run_leaves_the_pack_byte_identical(self):
        before = self._fingerprint()
        run(self.pack, FakeModel(['{"v": "one"}']), {})
        self.assertEqual(before, self._fingerprint())

    def test_two_runs_of_the_same_pack_are_independent(self):
        first = run(self.pack, FakeModel(['{"v": "one"}']), {})
        second = run(self.pack, FakeModel(['{"v": "two"}']), {})
        self.assertEqual(first.state["r"], {"v": "one"})
        self.assertEqual(second.state["r"], {"v": "two"})


class TheModelNeverChoosesTheNextNode(unittest.TestCase):
    """graph.py §3 / README §25: the small model never plans.

    Routing is the graph's job. The model is consulted exactly once per generate node
    on the path and never asked "what next" — that is what keeps the horizon short
    enough for a small model to stay reliable.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        schema = {"type": "object",
                  "properties": {"kind": {"type": "string", "enum": ["x", "y"]}},
                  "required": ["kind"], "additionalProperties": False}
        graph = (
            "nodes:\n"
            "  a:\n"
            "    type: generate\n"
            "  bx:\n"
            "    type: end\n"
            "  by:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: a\n"
            "    to: bx\n"
            "    when:\n"
            "      kind: x\n"
            "  - from: a\n"
            "    to: by\n"
        )
        _write_pack(self.root, graph, {"a": schema}, {"a.txt": "go\n"})
        self.pack = load_pack(self.root)

    def test_exactly_one_model_call_per_generate_node_on_the_path(self):
        model = FakeModel(['{"kind": "x"}'])
        result = run(self.pack, model, {})
        generates = [n for n in result.path if self.pack.nodes[n].type == "generate"]
        self.assertEqual(len(model.calls), len(generates))

    def test_routing_follows_committed_state_not_a_second_opinion(self):
        for kind, expected in (("x", "bx"), ("y", "by")):
            result = run(self.pack, FakeModel(['{"kind": "%s"}' % kind]), {})
            self.assertEqual(result.end_node, expected)

    def test_no_prompt_ever_asks_the_model_where_to_go(self):
        model = FakeModel(['{"kind": "x"}'])
        run(self.pack, model, {})
        for call in model.calls:
            for node_name in ("bx", "by"):
                self.assertNotIn(node_name, call.prompt)


class EveryCommittedNodeIsCheckpointed(unittest.TestCase):
    """state.py §4: state is written after every node that completes.

    Resume is only as good as the checkpoint density. A node that commits without a
    checkpoint is re-executed on resume — paying twice and, on a non-idempotent tool,
    acting twice.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        graph = (
            "nodes:\n"
            "  a:\n"
            "    type: generate\n"
            "    output: r1\n"
            "  b:\n"
            "    type: generate\n"
            "    output: r2\n"
            "  z:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: a\n"
            "    to: b\n"
            "  - from: b\n"
            "    to: z\n"
        )
        _write_pack(self.root, graph, {"a": STR_SCHEMA, "b": STR_SCHEMA},
                    {"a.txt": "go\n", "b.txt": "go\n"})
        self.pack = load_pack(self.root)
        self.store = Store(os.path.join(self.root, "ck.sqlite"))
        self.addCleanup(self.store.close)

    def test_one_checkpoint_per_node_on_the_path(self):
        result = run(self.pack, FakeModel(['{"v": "1"}', '{"v": "2"}']), {},
                     run_id="r", store=self.store)
        self.assertEqual(len(self.store.history("r")), len(result.path))

    def test_the_checkpoint_records_the_state_as_committed(self):
        run(self.pack, FakeModel(['{"v": "1"}', '{"v": "2"}']), {},
            run_id="r", store=self.store)
        first = self.store.history("r")[0]
        self.assertEqual(first.node, "a")
        self.assertEqual(first.state["r1"], {"v": "1"})


class AResumeIsRefusedIfThePackMoved(unittest.TestCase):
    """state.py: a checkpoint records which pack wrote it, name AND version.

    Resuming under a graph that has since changed silently skips newly inserted nodes
    and trusts changed ones. The version is the cheap half of that guard, and it only
    works if the walker actually hands the pack over — jig/graph.py:_checkpoint used to
    pass `pack.name`, which threw the version away before it reached the store.
    """

    def _pack_at(self, root, version, extra_node=""):
        graph = (
            "nodes:\n"
            "  a:\n"
            "    type: generate\n"
            "    output: r1\n"
            "%s"
            "  z:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: a\n"
            "    to: z\n"
        ) % extra_node
        _write_pack(root, graph, {"a": STR_SCHEMA}, {"a.txt": "go\n"},
                    manifest="name: p\nversion: %s\nentry: a\n" % version)
        return load_pack(root)

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.store = Store(os.path.join(self.root, "ck.sqlite"))
        self.addCleanup(self.store.close)

    def test_an_ordinary_run_records_the_pack_version(self):
        pack = self._pack_at(os.path.join(self.root, "p"), 1)
        run(pack, FakeModel(['{"v": "x"}']), {}, run_id="r", store=self.store)
        recorded = self.store.latest("r")
        self.assertEqual(recorded.pack, "p")
        self.assertEqual(recorded.pack_version, "1",
                         "the walker dropped the version before the store saw it")

    def test_resuming_under_a_bumped_version_is_refused(self):
        pack = self._pack_at(os.path.join(self.root, "p"), 1)
        run(pack, FakeModel(['{"v": "x"}']), {}, run_id="r", store=self.store)
        moved = self._pack_at(os.path.join(self.root, "p2"), 2)
        with self.assertRaises(Exception) as caught:
            resume(moved, FakeModel(['{"v": "y"}']), "r", self.store)
        self.assertIn("version", str(caught.exception).lower())

    def test_resuming_the_same_version_still_works(self):
        pack = self._pack_at(os.path.join(self.root, "p"), 1)
        run(pack, FakeModel(['{"v": "x"}']), {}, run_id="r", store=self.store)
        again = self._pack_at(os.path.join(self.root, "p3"), 1)
        replayed = resume(again, FakeModel(['{"v": "y"}']), "r", self.store)
        self.assertEqual(replayed.state["r1"], {"v": "x"})


if __name__ == "__main__":
    unittest.main()

class TheShapeCheckRunsEvenWithoutASchema(unittest.TestCase):
    """JSON validity is not a schema question, so it cannot be skipped when there is no schema.

    `verify` guarded the whole check with `if node.grammar:`, and `{}` is falsy — while the
    pack format documents `{}` as a legal grammar for a free-form field. So the one node
    shape most likely to receive junk was the one node that never checked for it: NaN,
    Infinity and 3000-deep nesting all committed, `json.dumps` then emitted a bare `NaN`
    that no strict reader can parse, and a run with `--store` died *after* the commit —
    exactly the hazard checking-before-commit exists to prevent.

    Found by an independent reviewer fact-checking docs/pack-format.md against the source.
    """

    def _node(self, grammar):
        return Node(name="a", type="generate", prompt="p", grammar=grammar)

    def test_nan_is_rejected_even_when_the_grammar_is_empty(self):
        with self.assertRaises(Rejected):
            verify(self._node({}), '{"kind": NaN}', {})

    def test_infinity_is_rejected_even_when_the_grammar_is_empty(self):
        with self.assertRaises(Rejected):
            verify(self._node({}), '{"kind": Infinity}', {})

    def test_nan_is_rejected_when_there_is_no_grammar_at_all(self):
        with self.assertRaises(Rejected):
            verify(self._node(None), '{"kind": NaN}', {})

    def test_deep_nesting_is_rejected_even_when_the_grammar_is_empty(self):
        deep = '{"v": %s}' % ("[" * 3000 + "]" * 3000)
        with self.assertRaises(Rejected):
            verify(self._node({}), deep, {})

    def test_an_ordinary_object_still_commits_under_an_empty_grammar(self):
        """The point is a shape check, not a new constraint — free-form must stay free."""
        value = verify(self._node({}), '{"anything": 1, "at": ["all"]}', {})
        self.assertEqual(value, {"anything": 1, "at": ["all"]})

    def test_a_declared_schema_is_still_enforced(self):
        schema = {"type": "object", "properties": {"k": {"type": "string"}},
                  "required": ["k"], "additionalProperties": False}
        with self.assertRaises(Rejected):
            verify(self._node(schema), '{"k": 12345}', {})



class ValidateReportsTheToolCheck(unittest.TestCase):
    """`jig validate --tools` must say that it checked, not just check.

    A check whose success is indistinguishable from its absence is a check nobody trusts:
    the flag printed exactly what omitting it printed, so neither a reader nor a CI log
    could tell the stricter pass had run. The count is the evidence.
    """

    def _run(self, *extra):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [sys.executable, "-m", "jig", "validate", "examples/refund_desk"] + list(extra),
            capture_output=True, text=True, cwd=str(root))
        return (proc.stdout + proc.stderr).strip()

    def test_without_tools_it_claims_nothing(self):
        self.assertNotIn("checked", self._run())

    def test_with_tools_it_names_the_count(self):
        self.assertIn("2 tools checked", self._run(
            "--tools", "examples/refund_desk/tools.py:registry"))

    def test_a_pack_with_no_tools_says_zero_rather_than_staying_silent(self):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [sys.executable, "-m", "jig", "validate", "examples/support_triage",
             "--tools", "examples/refund_desk/tools.py:registry"],
            capture_output=True, text=True, cwd=str(root))
        self.assertIn("0 tools checked", proc.stdout + proc.stderr)


class RedactorCoversTheSeparatorsVendorsActuallyUse(unittest.TestCase):
    """A key must not reach a log, whichever separator its vendor chose.

    The pattern used to require a hyphen after the prefix. Groq issues `gsk_...` and
    GitHub issues `ghp_...`, so the two prefixes most likely to appear in a real 401 body
    were the two it could not match — and README promises that no key reaches a log
    through any call site. Found by audit, not by a test, which is why this exists.
    """

    LIVE_SHAPES = [
        ("groq", "gsk_NOTAREALKEY8Zt4Qn1Rb7Wm2Kd9Pf3Xj6"),
        ("github-pat", "ghp_NOTAREALKEYb2C3d4E5f6G7h8I9j0K1l2"),
        ("github-oauth", "gho_NOTAREALKEYb2C3d4E5f6G7h8I9j0K1l2"),
        ("github-server", "ghs_NOTAREALKEYb2C3d4E5f6G7h8I9j0K1l2"),
        ("anthropic", "sk-ant-api03-NOTAREALKEY12345678"),
        ("openai-project", "sk-proj-NOTAREALKEY123456789012"),
        ("cerebras", "csk-NOTAREALKEY1234567890abcdef"),
        ("xai", "xai-NOTAREALKEY1234567890"),
        ("slack-bot", "xoxb-NOTAREALKEY-1234567890"),
        ("slack-user", "xoxp-NOTAREALKEY-1234567890"),
        ("aws", "AKIA_NOTAREALKEY1234"),
    ]

    def test_no_vendor_key_shape_survives_redaction(self):
        from jig.log import redact
        leaked = []
        for vendor, key in self.LIVE_SHAPES:
            if key in redact("upstream rejected %s, retrying" % key):
                leaked.append(vendor)
        self.assertEqual([], leaked, "these key shapes reached the log: %s" % leaked)

    def test_it_does_not_redact_ordinary_text(self):
        from jig.log import redact
        line = "node classify took 12ms, kind=complaint, order_id=A-1001"
        self.assertEqual(line, redact(line))

    def test_a_key_inside_a_larger_message_is_still_caught(self):
        from jig.log import redact
        out = redact('{"error":{"message":"Incorrect key: gsk_NOTAREALKEY8Zt4Qn1Rb7Wm2Kd"}}')
        self.assertNotIn("gsk_NOTAREALKEY", out)
        self.assertIn("<redacted>", out)


class CompileRefusesAnOccupiedDestinationBeforeSpendingAnything(unittest.TestCase):
    """A compile that cannot install what it builds must not build it.

    `_install` refuses to clobber an existing pack. That refusal used to be raised inside
    the attempt loop's try, where `except BuildError` caught it, recorded it as a failed
    attempt, fed the filesystem error to the *planner* as though a decomposition were
    wrong, and re-planned — paying for `induce` and `write_prompts` again on every
    remaining attempt. With the default of three attempts, forgetting `--overwrite` cost
    three full compiles against a frontier model and then reported "could not compile a
    pack that satisfies its own examples", which was false: every attempt had scored full
    marks. Found by audit.
    """

    def setUp(self):
        import tempfile, pathlib
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="jig-install-test-"))
        (self.root / "out").mkdir()
        (self.root / "out" / "keep.txt").write_text("a hand-tuned pack lives here\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _model_that_must_never_be_called(self):
        def model(*args, **kwargs):
            raise AssertionError(
                "the planner was called; the destination check did not run first")
        return model

    def test_it_refuses_before_the_model_is_consulted(self):
        from jig.build.compile import compile_pack
        from jig.build.spec import BuildError
        cases = [{"input": {"text": "hi"}, "expect": {"verdict": "allow"}}]
        with self.assertRaises(BuildError) as caught:
            compile_pack(str(self.root / "out"), "Moderate content.", cases,
                         self._model_that_must_never_be_called(), overwrite=False)
        self.assertIn("--overwrite", str(caught.exception))

    def test_the_existing_pack_is_left_alone(self):
        from jig.build.compile import compile_pack
        from jig.build.spec import BuildError
        cases = [{"input": {"text": "hi"}, "expect": {"verdict": "allow"}}]
        try:
            compile_pack(str(self.root / "out"), "Moderate content.", cases,
                         self._model_that_must_never_be_called(), overwrite=False)
        except BuildError:
            pass
        self.assertEqual("a hand-tuned pack lives here\n",
                         (self.root / "out" / "keep.txt").read_text())


class ShippedBackendMarksAContentLessTwoHundredAsABadDraw(unittest.TestCase):
    """A 200 with no text must spend a rung, not abort the run.

    `verify.EmptyCompletion` documents the contract and names `jig.backends.openai_compat`
    as marking its errors with `empty_content` — which that module did not do. So the only
    shipped backend raised a plain `BackendError` on the first content-less answer,
    `graph.run` did not catch it, and a node with an `on_fail` edge aborted instead of
    diverting: one call, no rung spent, no divert. The machinery was right and only the
    wiring was missing. Found by audit; the repo's own faultproxy had been reproducing it
    under the name "the real defect we hit on first contact".
    """

    def _no_content_response(self, **extra):
        response = {"choices": [{"message": {"content": None},
                                 "finish_reason": "length"}]}
        response.update(extra)
        return response

    def test_a_choice_with_no_text_is_marked(self):
        from jig.backends.openai_compat import _content
        from jig.errors import BackendError
        with self.assertRaises(BackendError) as caught:
            _content(self._no_content_response())
        self.assertTrue(getattr(caught.exception, "empty_content", False),
                        "the backend did not mark this as a bad draw, so run_node "
                        "will abort the run instead of spending a rung")

    def test_a_response_with_no_choices_at_all_is_marked(self):
        from jig.backends.openai_compat import _content
        from jig.errors import BackendError
        with self.assertRaises(BackendError) as caught:
            _content({"choices": []})
        self.assertTrue(getattr(caught.exception, "empty_content", False))

    def test_the_reasoning_model_diagnostic_survives_the_marking(self):
        from jig.backends.openai_compat import _content
        from jig.errors import BackendError
        response = self._no_content_response(
            usage={"completion_tokens_details": {"reasoning_tokens": 31}})
        with self.assertRaises(BackendError) as caught:
            _content(response)
        self.assertIn("reasoning", str(caught.exception))
        self.assertTrue(getattr(caught.exception, "empty_content", False))

    def test_it_spends_a_rung_and_takes_on_fail(self):
        from jig.backends.openai_compat import _content
        from jig.errors import BackendError
        from jig.graph import run
        from jig.pack import Edge, Node, Pack

        calls = []

        class AlwaysEmpty(object):
            """Answers 200, every time, with nothing in it — exactly what the shipped
            backend turns into an `empty_content` error."""

            def generate(self, prompt, **kwargs):
                calls.append(prompt)
                _content({"choices": [{"message": {"content": None},
                                       "finish_reason": "length"}]})

        model = AlwaysEmpty()

        pack = Pack(
            name="t", version=1, entry="a",
            nodes={
                "a": Node(name="a", type="generate", prompt="say k", retries=1,
                          on_fail="rescue", max_tokens=16,
                          grammar={"type": "object",
                                   "properties": {"k": {"enum": ["a"]}},
                                   "required": ["k"]}),
                "done": Node(name="done", type="end", output=["k"]),
                "rescue": Node(name="rescue", type="end", output=[]),
            },
            edges=[Edge(source="a", target="done")],
            evalset=[], manifest={}, path=None, model=None,
        )
        result = run(pack, model, {}, max_steps=8)
        self.assertEqual("rescue", result.end_node)
        self.assertEqual(2, len(calls), "the ladder should have drawn twice")


class TheCliNeverShowsARawTracebackForAValueJsonCannotHold(unittest.TestCase):
    """A tool may return a date; JSON may not. The user must be told, not shown a stack.

    `Tool._checked` validates the KEYS a tool returns and not the value types, which is a
    deliberate choice — a tool's values are the host's business. But the value then travels
    to the store or to stdout, where `json` refuses it with a `TypeError`, and `cli.main`
    named PackError, JigError, ValidationError and ValueError while omitting TypeError. So
    a `datetime.date` in a tool's return produced a multi-frame traceback through jig
    internals. A NaN took the ValueError branch and printed cleanly, which made the split
    arbitrary as well as ugly. Found by audit.
    """

    def _run(self, tool):
        import subprocess, sys, pathlib, tempfile, textwrap, os
        root = pathlib.Path(__file__).resolve().parent.parent
        work = pathlib.Path(tempfile.mkdtemp(prefix="jig-cli-json-"))
        pack = work / "p"
        (pack / "prompts").mkdir(parents=True)
        (pack / "grammars").mkdir()
        (pack / "fakes").mkdir()
        (pack / "fakes" / "script.json").write_text('["{}"]')
        (pack / "graph.yaml").write_text(textwrap.dedent("""\
            nodes:
              t:
                type: tool
                tool: %s
              done:
                type: end
                output: [y]
            edges:
              - from: t
                to: done
            """ % tool))
        (pack / "manifest.yaml").write_text(
            "name: p\nversion: 1\nentry: t\ninputs: [x]\n")
        (work / "tools.py").write_text(textwrap.dedent("""\
            import datetime
            from jig.tools import ToolRegistry
            registry = ToolRegistry()

            @registry.register("datetool", reads=["x"], writes=["y"])
            def datetool(x):
                return {"y": datetime.date(2026, 8, 25)}

            @registry.register("nantool", reads=["x"], writes=["y"])
            def nantool(x):
                return {"y": float("nan")}
            """))
        proc = subprocess.run(
            [sys.executable, "-m", "jig", "run", str(pack),
             "--tools", str(work / "tools.py"),
             "--model", "fake:fakes/script.json", "--input", '{"x": 1}'],
            capture_output=True, text=True, cwd=str(root),
            env=dict(os.environ, PYTHONPATH=str(root)))
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        return proc

    def test_a_date_is_a_jig_error_not_a_traceback(self):
        proc = self._run("datetool")
        self.assertNotIn("Traceback", proc.stdout + proc.stderr)
        self.assertIn("jig:", proc.stderr)
        self.assertEqual(1, proc.returncode)

    def test_the_message_says_what_to_return_instead(self):
        proc = self._run("datetool")
        self.assertIn("JSON-shaped", proc.stderr)

    def test_a_nan_never_reaches_stdout_as_json(self):
        proc = self._run("nantool")
        self.assertNotIn("NaN", proc.stdout)
        self.assertEqual(1, proc.returncode)


class CompilerLintReachesTheReader(unittest.TestCase):
    """`check_script`'s diagnosis must not be computed and then thrown away.

    The compiler already works out exactly which spec key a plan cannot honour — a case
    declaring `rescued: true` when the plan has no `on_fail`, for instance — and stored it
    on the attempt. Nothing read it: not `Attempt.__str__`, not `CompileResult.summary`,
    not `_feedback`. The user saw "11/12 cases" and had to rediscover what the compiler
    had already established, and the planner was re-run without being told. Found by audit.
    """

    NOTE = ("case 'unreadable' declares rescued: true, which needs a node to fail its "
            "whole ladder; the plan says nothing about on_fail")

    def test_the_note_is_printed_with_the_score(self):
        from jig.build.compile import Attempt
        rendered = str(Attempt(number=1, passed=11, total=12, lint=[self.NOTE]))
        self.assertIn("11/12", rendered)
        self.assertIn("rescued", rendered)

    def test_an_attempt_with_no_lint_is_unchanged(self):
        from jig.build.compile import Attempt
        rendered = str(Attempt(number=2, passed=12, total=12))
        self.assertEqual("attempt 2: 12/12 cases", rendered)
