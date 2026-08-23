"""T9 — the command line, exercised the way a user does: as a subprocess."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
PACK = os.path.join(FIXTURES, "cli_pack")


def jig(*args):
    """Invoke `python3 -m jig ...` and return (exit code, stdout, stderr)."""
    completed = subprocess.run(
        [sys.executable, "-m", "jig"] + list(args),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return completed.returncode, completed.stdout, completed.stderr


class TestValidate(unittest.TestCase):
    def test_a_good_pack_validates(self):
        code, out, _ = jig("validate", PACK)
        self.assertEqual(code, 0)
        self.assertIn("cli_demo", out)
        self.assertIn("2 nodes", out)
        self.assertIn("2 evalset", out)

    def test_a_broken_pack_reports_the_reason_and_exits_one(self):
        code, out, err = jig("validate", os.path.join(FIXTURES, "bad_dangling_edge"))
        self.assertEqual(code, 1)
        self.assertIn("nowhere", err)
        self.assertEqual(out.strip(), "")

    def test_a_missing_pack_directory_exits_one(self):
        code, _, err = jig("validate", os.path.join(FIXTURES, "not_a_pack"))
        self.assertEqual(code, 1)
        self.assertIn("not_a_pack", err)


class TestRun(unittest.TestCase):
    def test_running_prints_the_output_as_json(self):
        code, out, err = jig("run", PACK, "--input", '{"ticket": "I was charged twice"}')
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), {"category": "billing"})

    def test_the_model_comes_from_the_manifest_when_not_given(self):
        code, out, _ = jig("run", PACK, "--input", '{"ticket": "the app crashes"}')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), {"category": "technical"})

    def test_an_explicit_model_overrides_the_manifest(self):
        code, out, _ = jig(
            "run", PACK,
            "--input", '{"ticket": "the app crashes"}',
            "--model", "fake:fakes/wrong.json",
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), {"category": "billing"})

    def test_malformed_input_json_is_a_usage_error(self):
        code, _, err = jig("run", PACK, "--input", "{not json")
        self.assertEqual(code, 1)
        self.assertIn("--input", err)

    def test_a_prompt_variable_the_input_lacks_fails_clearly(self):
        code, _, err = jig("run", PACK, "--input", "{}")
        self.assertEqual(code, 1)
        self.assertIn("ticket", err)

    def test_input_defaults_to_empty_when_omitted(self):
        code, _, err = jig("run", PACK)
        self.assertEqual(code, 1)
        self.assertIn("ticket", err)

    def test_an_unknown_model_scheme_is_reported(self):
        code, _, err = jig("run", PACK, "--input", "{}", "--model", "wat:nonsense")
        self.assertEqual(code, 1)
        self.assertIn("wat", err)

    def test_state_flag_prints_the_whole_state_not_just_the_projection(self):
        code, out, _ = jig(
            "run", PACK, "--input", '{"ticket": "I was charged twice"}', "--state"
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(out), {"ticket": "I was charged twice", "category": "billing"}
        )


class TestRunWithCheckpoints(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.database = os.path.join(self.directory, "runs.sqlite3")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_a_store_records_the_run(self):
        code, _, err = jig(
            "run", PACK,
            "--input", '{"ticket": "I was charged twice"}',
            "--run-id", "cli-1",
            "--store", self.database,
        )
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.isfile(self.database))

        sys.path.insert(0, ROOT)
        from jig.state import Store

        store = Store(self.database)
        self.assertEqual([c.node for c in store.history("cli-1")], ["classify", "done"])
        store.close()

    def test_resume_of_a_finished_run_reprints_its_output(self):
        jig("run", PACK, "--input", '{"ticket": "I was charged twice"}',
            "--run-id", "cli-2", "--store", self.database)
        code, out, err = jig("run", PACK, "--resume", "cli-2", "--store", self.database)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), {"category": "billing"})

    def test_resume_needs_a_store(self):
        code, _, err = jig("run", PACK, "--resume", "cli-2")
        self.assertEqual(code, 1)
        self.assertIn("--store", err)

    def test_resuming_an_unknown_run_exits_one(self):
        code, _, err = jig("run", PACK, "--resume", "ghost", "--store", self.database)
        self.assertEqual(code, 1)
        self.assertIn("ghost", err)


class TestEval(unittest.TestCase):
    def test_a_passing_pack_scores_and_exits_zero(self):
        code, out, err = jig("eval", PACK)
        self.assertEqual(code, 0, err)
        self.assertIn("2/2", out)

    def test_a_failing_pack_exits_one_and_names_the_case_and_node(self):
        code, out, _ = jig("eval", PACK, "--model", "fake:fakes/wrong.json")
        self.assertEqual(code, 1)
        self.assertIn("1/2", out)
        self.assertIn("technical case", out)
        self.assertIn("classify", out)

    def test_json_output_is_machine_readable(self):
        code, out, _ = jig("eval", PACK, "--json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["passed"], 2)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["cases"][0]["name"], "billing case")

    def test_json_output_on_failure_carries_the_attribution(self):
        code, out, _ = jig("eval", PACK, "--model", "fake:fakes/wrong.json", "--json")
        self.assertEqual(code, 1)
        report = json.loads(out)
        self.assertEqual(report["by_node"], {"classify": 1})

    def test_a_pack_without_an_evalset_exits_one(self):
        code, _, err = jig("eval", os.path.join(FIXTURES, "valid_pack"), "--model", "fake:x")
        self.assertEqual(code, 1)


class TestModelSpecs(unittest.TestCase):
    """Resolved in-process: constructing a backend opens no connection (see T11)."""

    def setUp(self):
        sys.path.insert(0, ROOT)
        from jig.cli import resolve_model
        from jig.pack import load_pack

        self.resolve = resolve_model
        self.pack = load_pack(PACK)

    def test_a_fake_spec_loads_the_packs_script(self):
        model = self.resolve("fake:fakes/script.json", self.pack)
        self.assertEqual(model.generate("please classify the charged ticket"),
                         '{"category": "billing"}')

    def test_the_manifest_spec_is_the_default(self):
        self.assertIsNotNone(self.resolve(None, self.pack))

    def test_an_openai_spec_builds_a_backend_without_calling_it(self):
        model = self.resolve("openai:http://localhost:8000#qwen3-8b", self.pack)
        self.assertEqual(model.url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(model.model, "qwen3-8b")
        self.assertEqual(model.grammar_mode, "response_format")

    def test_an_openai_spec_can_choose_the_grammar_mode(self):
        model = self.resolve("openai:http://host/v1#llama#json_schema", self.pack)
        self.assertEqual(model.grammar_mode, "json_schema")

    def test_an_openai_spec_without_a_model_name_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.resolve("openai:http://localhost:8000", self.pack)
        self.assertIn("model name", str(caught.exception))

    def test_a_pack_with_no_model_at_all_is_rejected(self):
        from jig.pack import load_pack

        pack = load_pack(os.path.join(FIXTURES, "valid_pack"))
        object.__setattr__(pack, "model", None)
        with self.assertRaises(ValueError) as caught:
            self.resolve(None, pack)
        self.assertIn("--model", str(caught.exception))


class TestUsage(unittest.TestCase):
    def test_no_arguments_prints_usage(self):
        code, _, err = jig()
        self.assertEqual(code, 2)
        self.assertIn("usage", err.lower())

    def test_help_lists_every_command(self):
        code, out, _ = jig("--help")
        self.assertEqual(code, 0)
        for command in ("run", "eval", "validate"):
            self.assertIn(command, out)

    def test_version_is_reported(self):
        code, out, _ = jig("--version")
        self.assertEqual(code, 0)
        self.assertIn(".", out)
