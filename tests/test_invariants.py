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


if __name__ == "__main__":
    unittest.main()
