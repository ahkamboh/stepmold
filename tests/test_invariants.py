"""The load-bearing invariants, tested so that breaking them FAILS.

Every test here was written against a deliberately sabotaged copy of the module it
guards, and confirmed to fail, before the fix landed. That is the whole point: the
original suite named these invariants and could not catch their violation.

If you change one of these, you are changing a claim jig makes in docs/PLAN.md or
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
from jig.verify import Rejected, run_node


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
    """PLAN.md §3 / verify.py: a rejected generation must never be shown to the model.

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

    docs/PLAN.md §6 plans a pack registry and §7.2 describes scp-ing packs between
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


if __name__ == "__main__":
    unittest.main()
