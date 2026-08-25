"""T13 — structured logging, and the two things it must never write.

Half of this file is ordinary: the events exist, they carry the fields an operator needs,
and the JSON mode parses. The other half is the interesting half, because a logging layer
is a new *exfiltration path* through a runtime whose central claim is that a rejected
generation never comes back out. Three negatives are tested as hard as the positives:

* a planted canary API key never appears in captured output at **any** level;
* rejected model text never appears at INFO, and the safe half still does;
* a megabyte of user-controlled input never becomes a megabyte of log line.

And one property that is neither: with no flags, stepmold configures nothing, emits nothing,
and costs nothing. `logging.lastResort` makes that non-obvious — a library with no
handler at all still prints WARNING and above to stderr — so it is tested rather than
assumed.

NOTHING HERE TOUCHES A NETWORK. The backend tests inject a fake opener, exactly as
tests/test_backend.py does.
"""

import email.message
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import logging
import unittest
import urllib.error

from stepmold import log
from stepmold.backends.openai_compat import OpenAICompatModel
from stepmold.errors import NodeFailed
from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack
from stepmold.state import Store
from stepmold.verify import Rejected, run_node

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A key shaped exactly like the real thing, planted so a leak is unambiguous. If this
# string turns up in a log line, stepmold wrote a credential to disk.
CANARY = "sk-stepmoldcanary-4Vb92LmQ0pTz7XkD3nR8"

# Model output that must never reach a default-level log line. Distinctive enough that a
# substring search is proof either way.
POISON = "REFUND-SCAM-9f3a-ssn-123-45-6789"

STR_SCHEMA = {
    "type": "object",
    "properties": {"v": {"type": "string"}},
    "required": ["v"],
    "additionalProperties": False,
}
ENUM_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": ["billing", "technical"]}},
    "required": ["category"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------------- fixtures


class Captured(unittest.TestCase):
    """A TestCase that turns stepmold's logging on into a buffer, and off again after.

    `configure` is the real entry point the CLI uses, so the formatters — where the
    redaction actually runs — are exercised rather than bypassed.
    """

    LEVEL = "debug"
    FORMAT = "text"

    def setUp(self):
        self.stream = io.StringIO()
        log.configure(level=self.LEVEL, fmt=self.FORMAT, stream=self.stream)
        self.addCleanup(log.reset)

    @property
    def text(self):
        return self.stream.getvalue()

    def lines(self):
        return [line for line in self.text.splitlines() if line.strip()]

    def events(self):
        """Every event name emitted, in order."""
        found = []
        for line in self.lines():
            if self.FORMAT == "json":
                found.append(json.loads(line)["event"])
            else:
                found.append(line.split()[3])
        return found

    def records(self):
        """The JSON-mode lines, parsed. Only meaningful with FORMAT = 'json'."""
        return [json.loads(line) for line in self.lines()]

    def field(self, event, name):
        for record in self.records():
            if record["event"] == event:
                return record.get(name)
        raise AssertionError("no %r event in:\n%s" % (event, self.text))


def write_pack(root, graph, schemas, prompts, manifest="name: p\nversion: 1\nentry: a\n"):
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
        with open(os.path.join(root, "prompts", name), "w") as handle:
            handle.write(body)
    return load_pack(root)


ONE_NODE_GRAPH = (
    "nodes:\n"
    "  a:\n"
    "    type: generate\n"
    "    output: r\n"
    "    retries: 2\n"
    "  z:\n"
    "    type: end\n"
    "edges:\n"
    "  - from: a\n"
    "    to: z\n"
)


def one_node_pack(root, prompt="Handle this: {ticket}\n"):
    return write_pack(root, ONE_NODE_GRAPH, {"a": STR_SCHEMA}, {"a.txt": prompt})


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class FakeHTTP:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


def http_error(code, body=b"nope", **headers):
    message = email.message.Message()
    for name, value in headers.items():
        message[name.replace("_", "-")] = value
    return urllib.error.HTTPError("http://x", code, "boom", message, io.BytesIO(body))


def completion(content, **usage):
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 412, "completion_tokens": 19,
                           "total_tokens": 431},
    }
    return json.dumps(body).encode()


def backend(http, **kwargs):
    options = {
        "base_url": "http://localhost:8000",
        "model": "qwen3-8b",
        "opener": http,
        "sleeper": lambda seconds: None,
    }
    options.update(kwargs)
    return OpenAICompatModel(**options)


# ------------------------------------------------------- a library that stays quiet


class TestTheLibraryDoesNotConfigureItsHost(unittest.TestCase):
    """Rule 1 of stepmold/log.py. Importing a library must change nothing about logging."""

    def test_the_package_logger_carries_a_null_handler(self):
        handlers = logging.getLogger("stepmold").handlers
        self.assertTrue(
            any(isinstance(handler, logging.NullHandler) for handler in handlers),
            "without a NullHandler, logging.lastResort prints stepmold's warnings to stderr",
        )

    def test_importing_stepmold_installs_nothing_on_the_root_logger(self):
        completed = subprocess.run(
            [sys.executable, "-c",
             "import logging, stepmold, stepmold.graph, stepmold.state, stepmold.backends.openai_compat;"
             "print(len(logging.getLogger().handlers))"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.stderr, "")
        self.assertEqual(completed.stdout.strip(), "0")

    def test_a_warning_with_nobody_listening_reaches_no_stream(self):
        """`logging.lastResort` is the trap this guards.

        A logger with no handler anywhere in its chain still prints WARNING and above to
        stderr, so an unconfigured stepmold emitting one rejection would put a line on the
        terminal of an application that never asked for it.
        """
        completed = subprocess.run(
            [sys.executable, "-c",
             "from stepmold.log import WARNING, event, get_logger;"
             "event(get_logger('graph'), WARNING, 'node.rejected', node='a')"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_configure_does_not_touch_the_root_logger(self):
        before = list(logging.getLogger().handlers)
        log.configure(level="debug", stream=io.StringIO())
        self.addCleanup(log.reset)
        self.assertEqual(logging.getLogger().handlers, before)

    def test_reset_removes_the_handler_configure_installed(self):
        log.configure(level="debug", stream=io.StringIO())
        log.reset()
        kinds = [type(handler) for handler in logging.getLogger("stepmold").handlers]
        self.assertEqual(kinds, [logging.NullHandler])

    def test_configure_twice_does_not_double_the_output(self):
        stream = io.StringIO()
        log.configure(level="info", stream=io.StringIO())
        log.configure(level="info", stream=stream)
        self.addCleanup(log.reset)
        log.event(log.get_logger("graph"), log.INFO, "run.start", run_id="x")
        self.assertEqual(len(stream.getvalue().splitlines()), 1)


class TestDisabledLoggingCostsNothing(unittest.TestCase):
    """Rule 3. The long-horizon suite walks 50-node packs; this must be free there."""

    def test_an_unconfigured_logger_is_not_enabled_for_debug(self):
        """This is the check every hot call site guards on. If it were True by default,
        every node would build a digest of state nobody asked for."""
        log.reset()
        self.assertFalse(log.get_logger("graph").isEnabledFor(log.DEBUG))
        self.assertFalse(log.get_logger("verify").isEnabledFor(log.INFO))

    def test_a_filtered_event_never_reaches_a_formatter(self):
        formatted = []

        class Counting(log.TextFormatter):
            def format(self, record):
                formatted.append(record)
                return log.TextFormatter.format(self, record)

        stream = io.StringIO()
        log.configure(level="warning", stream=stream)
        self.addCleanup(log.reset)
        logging.getLogger("stepmold").handlers[-1].setFormatter(Counting())

        logger = log.get_logger("graph")
        for _ in range(100):
            log.event(logger, log.DEBUG, "node.emit", node="a", prompt_bytes=1)
        self.assertEqual(formatted, [])

    def test_a_run_with_logging_off_emits_no_records_at_all(self):
        log.reset()
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = one_node_pack(root, prompt="go\n")

        seen = []

        class Collect(logging.Handler):
            def emit(self, record):
                seen.append(record)

        handler = Collect()
        logging.getLogger("stepmold").addHandler(handler)
        self.addCleanup(logging.getLogger("stepmold").removeHandler, handler)

        run(pack, FakeModel(['{"v": "ok"}']), {})
        self.assertEqual(seen, [])


# ------------------------------------------------------------------ the events exist


class TestRunEvents(Captured):
    FORMAT = "json"

    def setUp(self):
        Captured.setUp(self)
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.pack = one_node_pack(self.root, prompt="go\n")

    def test_a_run_logs_its_start_and_its_end(self):
        run(self.pack, FakeModel(['{"v": "ok"}']), {}, run_id="job-1")
        self.assertIn("run.start", self.events())
        self.assertIn("run.end", self.events())

    def test_run_start_names_the_pack_its_version_and_the_entry_node(self):
        run(self.pack, FakeModel(['{"v": "ok"}']), {}, run_id="job-1")
        self.assertEqual(self.field("run.start", "run_id"), "job-1")
        self.assertEqual(self.field("run.start", "pack"), "p")
        self.assertEqual(self.field("run.start", "version"), 1)
        self.assertEqual(self.field("run.start", "entry"), "a")

    def test_run_end_answers_what_happened_and_what_it_cost(self):
        run(self.pack, FakeModel(['{"v": "ok"}']), {}, run_id="job-1")
        self.assertEqual(self.field("run.end", "end_node"), "z")
        self.assertEqual(self.field("run.end", "steps"), 2)
        self.assertEqual(self.field("run.end", "generations"), 1)
        self.assertEqual(self.field("run.end", "failures"), 0)
        self.assertIsInstance(self.field("run.end", "duration_ms"), float)

    def test_every_node_reports_its_name_type_attempts_and_duration(self):
        run(self.pack, FakeModel(['{"v": "ok"}']), {})
        self.assertEqual(self.field("node.ok", "node"), "a")
        self.assertEqual(self.field("node.ok", "type"), "generate")
        self.assertEqual(self.field("node.ok", "attempts"), 1)
        self.assertIsInstance(self.field("node.ok", "duration_ms"), float)

    def test_a_rescued_node_reports_the_generations_it_actually_burned(self):
        """The run an operator wants to see *before* the next one fails."""
        model = FakeModel(["not json at all", '{"v": "ok"}'])
        run(self.pack, model, {})
        self.assertEqual(self.field("node.ok", "attempts"), 2)
        self.assertEqual(self.field("run.end", "generations"), 2)

    def test_the_retry_rung_is_logged_with_the_sampling_it_asked_for(self):
        run(self.pack, FakeModel(["not json at all", '{"v": "ok"}']), {})
        self.assertIn("node.retry", self.events())
        self.assertEqual(self.field("node.retry", "attempt"), 2)
        self.assertEqual(self.field("node.retry", "temperature"), 0.5)
        self.assertEqual(self.field("node.retry", "seed"), 1)

    def test_a_rejection_names_the_node_and_the_attempt(self):
        run(self.pack, FakeModel(["not json at all", '{"v": "ok"}']), {})
        self.assertEqual(self.field("node.rejected", "node"), "a")
        self.assertEqual(self.field("node.rejected", "attempt"), 1)

    def test_a_spent_ladder_logs_the_failure_and_the_edge_it_took(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        graph = (
            "nodes:\n"
            "  a:\n"
            "    type: generate\n"
            "    output: r\n"
            "    retries: 0\n"
            "    on_fail: rescue\n"
            "  rescue:\n"
            "    type: end\n"
            "  z:\n"
            "    type: end\n"
            "edges:\n"
            "  - from: a\n"
            "    to: z\n"
        )
        pack = write_pack(root, graph, {"a": STR_SCHEMA}, {"a.txt": "go\n"})
        run(pack, FakeModel(["nonsense"]), {})
        self.assertEqual(self.field("node.failed", "node"), "a")
        self.assertEqual(self.field("node.failed", "on_fail"), "rescue")
        self.assertEqual(self.field("edge.on_fail", "to"), "rescue")
        self.assertEqual(self.field("run.end", "end_node"), "rescue")

    def test_a_run_that_dies_records_which_node_it_died_on(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = one_node_pack(root, prompt="go\n")
        with self.assertRaises(NodeFailed):
            run(pack, FakeModel(["no", "no", "no"]), {})
        self.assertEqual(self.field("run.error", "node"), "a")
        self.assertEqual(self.field("run.error", "error"), "NodeFailed")


class TestCheckpointAndResumeEvents(Captured):
    FORMAT = "json"

    def setUp(self):
        Captured.setUp(self)
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.pack = one_node_pack(self.root, prompt="go\n")
        self.store = Store(os.path.join(self.root, "ck.sqlite"))
        self.addCleanup(self.store.close)

    def test_a_checkpointed_run_claims_its_id_and_records_every_write(self):
        run(self.pack, FakeModel(['{"v": "ok"}']), {}, run_id="job-2", store=self.store)
        self.assertEqual(self.field("run.claimed", "run_id"), "job-2")
        self.assertEqual(self.field("checkpoint.saved", "node"), "a")
        self.assertEqual(self.field("checkpoint.saved", "step"), 1)

    def test_a_resume_logs_the_lease_it_took_and_gave_back(self):
        from stepmold.state import resume

        run(self.pack, FakeModel(['{"v": "ok"}']), {}, run_id="job-3", store=self.store)
        resume(self.pack, FakeModel(['{"v": "unused"}']), "job-3", self.store)
        # A finished run replays rather than re-executing, so it takes no lease; an
        # unfinished one is the interesting case.
        self.assertIn("resume.replayed", self.events())

    def test_an_unfinished_resume_takes_a_lease(self):
        from stepmold.state import resume

        graph = (
            "nodes:\n"
            "  a:\n"
            "    type: generate\n"
            "    output: r\n"
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
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = write_pack(root, graph, {"a": STR_SCHEMA, "b": STR_SCHEMA},
                          {"a.txt": "go\n", "b.txt": "go\n"})
        store = Store(os.path.join(root, "ck.sqlite"))
        self.addCleanup(store.close)
        try:
            run(pack, FakeModel(['{"v": "1"}', "nonsense", "nonsense", "nonsense"]), {},
                run_id="job-4", store=store)
        except Exception:
            pass
        resume(pack, FakeModel(['{"v": "2"}']), "job-4", store)
        self.assertIn("resume.start", self.events())
        self.assertIn("lease.taken", self.events())
        self.assertIn("lease.released", self.events())


class TestBackendEvents(Captured):
    FORMAT = "json"

    def test_a_completion_logs_the_model_endpoint_status_tokens_and_duration(self):
        http = FakeHTTP(completion('{"v": "ok"}'))
        backend(http).generate("hi")
        self.assertEqual(self.field("backend.response", "model"), "qwen3-8b")
        self.assertEqual(
            self.field("backend.response", "endpoint"),
            "http://localhost:8000/v1/chat/completions",
        )
        self.assertEqual(self.field("backend.response", "status"), 200)
        self.assertEqual(self.field("backend.response", "prompt_tokens"), 412)
        self.assertEqual(self.field("backend.response", "completion_tokens"), 19)
        self.assertEqual(self.field("backend.response", "retries"), 0)
        self.assertIsInstance(self.field("backend.response", "duration_ms"), float)

    def test_a_reasoning_model_reports_its_reasoning_tokens(self):
        """The number ARCHITECTURE.md wants watched: output spend that bought no output."""
        http = FakeHTTP(completion(
            '{"v": "ok"}',
            prompt_tokens=100, completion_tokens=600,
            completion_tokens_details={"reasoning_tokens": 476},
        ))
        backend(http).generate("hi")
        self.assertEqual(self.field("backend.response", "reasoning_tokens"), 476)

    def test_a_missing_usage_block_leaves_the_field_present_and_null(self):
        """Stable field names, even against a server that reports nothing."""
        body = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()
        backend(FakeHTTP(body)).generate("hi")
        self.assertIsNone(self.field("backend.response", "prompt_tokens"))

    def test_a_retryable_status_logs_the_status_and_the_sleep_it_actually_took(self):
        failure = http_error(429, b'{"error": "slow down"}', Retry_After="7")
        self.addCleanup(failure.close)
        http = FakeHTTP(failure, completion('{"v": "ok"}'))
        backend(http).generate("hi")
        self.assertEqual(self.field("backend.http_error", "status"), 429)
        self.assertTrue(self.field("backend.http_error", "retryable"))
        self.assertEqual(self.field("backend.backoff", "retry_after"), 7.0)
        self.assertEqual(self.field("backend.backoff", "slept_s"), 7.0)
        self.assertEqual(self.field("backend.response", "retries"), 1)

    def test_an_exhausted_backend_logs_the_final_failure(self):
        failure = http_error(503, b"down")
        self.addCleanup(failure.close)
        with self.assertRaises(Exception):
            backend(FakeHTTP(failure), max_retries=1).generate("hi")
        self.assertEqual(self.field("backend.failed", "attempts"), 2)


# ------------------------------------------------------------------ the two formats


class TestTextFormat(Captured):
    FORMAT = "text"

    def test_a_line_carries_the_level_the_logger_the_event_and_its_fields(self):
        log.event(log.get_logger("graph"), log.INFO, "node.ok", node="classify",
                  attempts=2)
        line = self.lines()[0]
        self.assertIn("INFO", line)
        self.assertIn("stepmold.graph", line)
        self.assertIn("node.ok", line)
        self.assertIn("node=classify", line)
        self.assertIn("attempts=2", line)

    def test_a_value_containing_a_space_is_quoted_so_it_cannot_read_as_two_fields(self):
        log.event(log.get_logger("graph"), log.INFO, "node.failed",
                  reason="schema: category is not one of billing, technical")
        self.assertIn('reason="schema:', self.lines()[0])

    def test_a_missing_value_prints_as_a_dash_rather_than_vanishing(self):
        log.event(log.get_logger("graph"), log.INFO, "node.ok", output=None)
        self.assertIn("output=-", self.lines()[0])


class TestJsonFormat(Captured):
    FORMAT = "json"

    def test_every_line_is_one_json_object(self):
        log.event(log.get_logger("graph"), log.INFO, "run.start", run_id="x")
        log.event(log.get_logger("graph"), log.INFO, "run.end", run_id="x")
        parsed = [json.loads(line) for line in self.lines()]
        self.assertEqual([item["event"] for item in parsed], ["run.start", "run.end"])

    def test_the_four_reserved_keys_are_always_present(self):
        log.event(log.get_logger("verify"), log.WARNING, "node.rejected", node="a")
        record = self.records()[0]
        for key in ("ts", "level", "logger", "event"):
            self.assertIn(key, record)
        self.assertEqual(record["level"], "WARNING")
        self.assertEqual(record["logger"], "stepmold.verify")

    def test_fields_land_at_the_top_level_where_jq_can_reach_them(self):
        log.event(log.get_logger("graph"), log.INFO, "node.ok", node="a", attempts=3)
        self.assertEqual(self.records()[0]["node"], "a")
        self.assertEqual(self.records()[0]["attempts"], 3)

    def test_a_field_that_collides_with_a_reserved_key_is_kept_not_dropped(self):
        log.event(log.get_logger("graph"), log.INFO, "run.start", level="p1")
        record = self.records()[0]
        self.assertEqual(record["level"], "INFO")
        self.assertEqual(record["field_level"], "p1")


# ----------------------------------------------------- the negatives that matter


class TestNoCredentialEverReachesALogLine(Captured):
    """A canary key, planted in every place a key can be, at the loudest level there is.

    Redaction lives in the formatter rather than at the call sites on purpose: a rule
    enforced at the sink cannot be forgotten by the next person who adds an event.
    """

    LEVEL = "debug"

    def test_an_echoed_authorization_header_is_redacted_out_of_the_logged_body(self):
        echoed = ('{"error": "bad request", "sent": {"Authorization": "Bearer %s"}}'
                  % CANARY).encode()
        http = FakeHTTP(http_error(400, echoed))
        with self.assertRaises(Exception):
            backend(http, api_key=CANARY).generate("hi")
        self.assertNotIn(CANARY, self.text)
        self.assertIn("<redacted>", self.text)
        self.assertIn("bad request", self.text)  # the diagnosis survives the redaction

    def test_a_bare_key_shaped_string_in_an_error_body_is_redacted_too(self):
        http = FakeHTTP(http_error(400, ('{"error": "key %s revoked"}' % CANARY).encode()))
        with self.assertRaises(Exception):
            backend(http, api_key=None).generate("hi")
        self.assertNotIn(CANARY, self.text)
        self.assertIn("revoked", self.text)

    def test_a_key_in_a_transport_error_message_is_redacted(self):
        http = FakeHTTP(urllib.error.URLError("proxy rejected Bearer %s" % CANARY))
        with self.assertRaises(Exception):
            backend(http, api_key=CANARY, max_retries=0).generate("hi")
        self.assertNotIn(CANARY, self.text)

    def test_a_whole_successful_call_never_mentions_the_key_it_carried(self):
        http = FakeHTTP(completion('{"v": "ok"}'))
        backend(http, api_key=CANARY).generate("hi")
        self.assertNotIn(CANARY, self.text)
        self.assertIn("backend.response", self.events())

    def test_a_key_pasted_into_the_run_input_never_reaches_a_log_line(self):
        """The other direction: a credential in the caller's own data.

        Nothing logs state, so this passes by construction — which is exactly the claim
        worth pinning down, because the cheapest way to break it is to add
        `state=state` to one event.
        """
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = one_node_pack(root)
        run(pack, FakeModel(['{"v": "ok"}']), {"ticket": "my key is %s" % CANARY})
        self.assertNotIn(CANARY, self.text)

    def test_a_key_in_a_model_generation_never_reaches_a_log_line(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = one_node_pack(root, prompt="go\n")
        leaky = json.dumps({"v": CANARY, "extra": CANARY})  # extra -> schema rejection
        try:
            run(pack, FakeModel([leaky, leaky, leaky]), {})
        except Exception:
            pass
        self.assertNotIn(CANARY, self.text)

    def test_the_redactor_is_the_one_the_backend_already_used(self):
        """Reuse, not reinvention: the backend imports the filter back from stepmold.log."""
        from stepmold.backends import openai_compat

        self.assertIs(openai_compat._redact, log.redact)
        self.assertIs(openai_compat.KEY_SHAPED, log.KEY_SHAPED)


class TestRejectedModelTextStaysOutOfTheDefaultPath(unittest.TestCase):
    """stepmold's central invariant, extended to the new exfiltration path.

    `tests/test_invariants.py` proves a rejected generation never reaches the *model*.
    A log is not a model — bytes on disk cannot self-condition anything — but a log an
    operator ships to a collector is still somewhere the text should not go by default.
    So the split is by level: `Rejected.feedback` at INFO, `Rejected.detail` at DEBUG,
    and nothing in between.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.addCleanup(log.reset)

    def _run_with_rejections(self, level):
        stream = io.StringIO()
        log.configure(level=level, fmt="json", stream=stream)
        pack = write_pack(
            self.root, ONE_NODE_GRAPH, {"a": ENUM_SCHEMA}, {"a.txt": "go\n"}
        )
        bad = json.dumps({"category": POISON})
        run(pack, FakeModel([bad, bad, json.dumps({"category": "billing"})]), {})
        return stream.getvalue()

    def test_at_info_the_rejected_value_does_not_appear(self):
        self.assertNotIn(POISON, self._run_with_rejections("info"))

    def test_at_warning_the_rejected_value_does_not_appear(self):
        self.assertNotIn(POISON, self._run_with_rejections("warning"))

    def test_at_info_the_operator_is_still_told_what_was_wrong(self):
        """Sanitising must not become silence — that would make the log useless."""
        text = self._run_with_rejections("info")
        self.assertIn("node.rejected", text)
        self.assertIn("schema", text)
        self.assertIn("billing", text)  # the schema's own choices are the pack's words

    def test_at_debug_the_full_detail_is_available_to_the_operator(self):
        text = self._run_with_rejections("debug")
        self.assertIn("node.rejected.detail", text)
        self.assertIn(POISON, text)

    def test_a_node_that_spends_its_ladder_logs_the_safe_half_at_warning(self):
        stream = io.StringIO()
        log.configure(level="warning", fmt="json", stream=stream)
        graph = ONE_NODE_GRAPH.replace("    retries: 2\n", "    retries: 0\n")
        pack = write_pack(self.root, graph, {"a": ENUM_SCHEMA}, {"a.txt": "go\n"})
        with self.assertRaises(NodeFailed):
            run(pack, FakeModel([json.dumps({"category": POISON})]), {})
        text = stream.getvalue()
        self.assertIn("node.failed", text)
        self.assertNotIn(POISON, text)

    def test_node_failed_carries_both_halves_so_the_walker_can_choose(self):
        """The mechanism behind the test above, pinned at its source."""
        from stepmold.pack import Node

        node = Node(name="a", type="generate", prompt="go", grammar=ENUM_SCHEMA,
                    output="r", retries=0)
        with self.assertRaises(NodeFailed) as caught:
            run_node(node, {}, FakeModel([json.dumps({"category": POISON})]))
        self.assertIn(POISON, caught.exception.reason)
        self.assertNotIn(POISON, caught.exception.feedback)

    def test_a_node_failed_without_a_feedback_half_says_so_rather_than_leaking(self):
        from stepmold.graph import _safe_reason

        raw = NodeFailed("a", "output was not valid JSON: %s" % POISON, attempts=1)
        self.assertNotIn(POISON, _safe_reason(raw))

    def test_an_unparseable_generation_is_not_quoted_at_info(self):
        """The worst leak: free prose that never parsed, echoed whole."""
        stream = io.StringIO()
        log.configure(level="info", fmt="json", stream=stream)
        pack = write_pack(self.root, ONE_NODE_GRAPH, {"a": STR_SCHEMA},
                          {"a.txt": "go\n"})
        prose = "I think this is a %s from john.doe@example.com" % POISON
        run(pack, FakeModel([prose, prose, '{"v": "ok"}']), {})
        self.assertNotIn(POISON, stream.getvalue())


class TestNothingUserControlledArrivesUnbounded(Captured):
    """A one-megabyte ticket must not become a one-megabyte log line."""

    LEVEL = "debug"
    FORMAT = "text"

    def test_a_huge_rejection_detail_is_clipped(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = one_node_pack(root, prompt="go\n")
        huge = "x" * 1000000
        run(pack, FakeModel([huge, '{"v": "ok"}']), {})
        self.assertLess(len(self.text), 20000)
        self.assertNotIn("x" * 200, self.text)
        self.assertIn("...", self.text)  # something visibly got cut

    def test_the_formatter_bounds_a_field_nobody_clipped_at_the_call_site(self):
        """The backstop, tested on its own: redaction and clipping are properties of the
        sink, so an event added later cannot forget either one."""
        log.event(log.get_logger("graph"), log.INFO, "node.ok", note="q" * 100000)
        self.assertLess(len(self.text), 1000)
        self.assertIn("chars)", self.text)

    def test_a_huge_input_is_reported_by_size_not_by_content(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = one_node_pack(root)
        run(pack, FakeModel(['{"v": "ok"}']), {"ticket": "y" * 500000})
        self.assertLess(len(self.text), 20000)
        self.assertIn("state_bytes=", self.text)

    def test_clip_reports_the_length_it_threw_away(self):
        self.assertEqual(log.clip("abc", limit=10), "abc")
        clipped = log.clip("z" * 50, limit=10)
        self.assertTrue(clipped.startswith("z" * 10))
        self.assertIn("50", clipped)

    def test_clip_collapses_newlines_so_one_event_stays_one_line(self):
        self.assertEqual(log.clip("a\nb\r\nc"), "a b c")


class TestPromptsAndStateNeverAppearVerbatim(Captured):
    LEVEL = "debug"
    FORMAT = "json"

    def test_a_rendered_prompt_is_reported_as_a_byte_count(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = one_node_pack(root)
        run(pack, FakeModel(['{"v": "ok"}']), {"ticket": "SECRET-TICKET-BODY"})
        self.assertGreater(self.field("node.emit", "prompt_bytes"), 0)
        self.assertNotIn("SECRET-TICKET-BODY", self.text)

    def test_state_is_reported_as_a_digest_and_a_size(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pack = one_node_pack(root)
        run(pack, FakeModel(['{"v": "ok"}']), {"ticket": "SECRET-TICKET-BODY"})
        self.assertEqual(len(self.field("state.committed", "state_digest")), 12)
        self.assertGreater(self.field("state.committed", "state_bytes"), 0)

    def test_the_same_state_digests_the_same_and_a_different_one_does_not(self):
        self.assertEqual(log.digest({"a": 1}), log.digest({"a": 1}))
        self.assertNotEqual(log.digest({"a": 1}), log.digest({"a": 2}))


# --------------------------------------------------------------------------- the CLI


def stepmold(*argv):
    completed = subprocess.run(
        [sys.executable, "-m", "stepmold"] + list(argv), cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
    )
    return completed.returncode, completed.stdout, completed.stderr


TICKET = '{"ticket": "I was charged twice for order A-1001, $49.99 both times."}'
PACK = "examples/support_triage"


class TestTheCliFlags(unittest.TestCase):
    def test_by_default_a_run_prints_nothing_on_stderr(self):
        """The whole compatibility contract in one assertion."""
        code, out, err = stepmold("run", PACK, "--input", TICKET)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertTrue(out.startswith("{"))

    def test_log_level_off_is_the_same_as_no_flag(self):
        _, plain, _ = stepmold("run", PACK, "--input", TICKET)
        code, out, err = stepmold("run", PACK, "--input", TICKET, "--log-level", "off")
        self.assertEqual(out, plain)
        self.assertEqual(err, "")

    def test_log_level_info_writes_events_to_stderr_and_leaves_stdout_alone(self):
        _, plain, _ = stepmold("run", PACK, "--input", TICKET)
        code, out, err = stepmold("run", PACK, "--input", TICKET, "--log-level", "info")
        self.assertEqual(code, 0)
        self.assertEqual(out, plain, "logging must not touch the result on stdout")
        self.assertIn("run.start", err)
        self.assertIn("run.end", err)

    def test_log_format_json_puts_one_parseable_object_per_line_on_stderr(self):
        code, out, err = stepmold("run", PACK, "--input", TICKET,
                             "--log-level", "info", "--log-format", "json")
        self.assertEqual(code, 0)
        events = [json.loads(line)["event"] for line in err.splitlines() if line.strip()]
        self.assertIn("run.start", events)
        self.assertIn("node.ok", events)
        self.assertIn("run.end", events)

    def test_eval_takes_the_same_flags(self):
        code, out, err = stepmold("eval", PACK, "--log-level", "info")
        self.assertEqual(code, 0)
        self.assertIn("12/12", out)
        self.assertIn("run.end", err)

    def test_validate_takes_the_same_flags(self):
        code, out, err = stepmold("validate", PACK, "--log-level", "debug")
        self.assertEqual(code, 0)
        self.assertIn("support_triage", out)

    def test_an_unknown_level_is_a_usage_error_not_a_traceback(self):
        code, out, err = stepmold("run", PACK, "--input", TICKET, "--log-level", "chatty")
        self.assertEqual(code, 2)
        self.assertIn("chatty", err)


class TestLevelNames(unittest.TestCase):
    def test_the_names_the_cli_offers_all_resolve(self):
        for name in log.LEVELS:
            self.assertIsInstance(log.level_named(name), int)

    def test_off_silences_everything_including_critical(self):
        self.assertGreater(log.level_named("off"), logging.CRITICAL)

    def test_a_number_passes_through(self):
        self.assertEqual(log.level_named(logging.DEBUG), logging.DEBUG)

    def test_an_unknown_name_is_refused_with_the_known_ones(self):
        with self.assertRaises(ValueError) as caught:
            log.level_named("verbose")
        self.assertIn("verbose", str(caught.exception))


class TestRejectedKeepsItsTwoHalves(unittest.TestCase):
    """The type the whole level split rests on, pinned here as well as in verify."""

    def test_the_detail_and_the_feedback_are_not_the_same_string(self):
        exc = Rejected("output was not valid JSON: %s" % POISON,
                       feedback="output was not valid JSON")
        self.assertIn(POISON, exc.detail)
        self.assertNotIn(POISON, exc.feedback)


class TheSinkFiltersEverythingItEmits(unittest.TestCase):
    """Redaction must be a property of the sink, not a discipline at every call site.

    Each of these was a real defect found by an independent audit of the logging pass.
    They are grouped here because they share one cause: a filter that covered some of
    what reaches the formatter rather than all of it.
    """

    def test_a_credential_straddling_the_clip_boundary_is_still_redacted(self):
        """`redact(clip(x))` truncated a key until the pattern stopped matching it.

        The surviving prefix was then printed verbatim. Order matters: redact first.
        """
        from stepmold.log import _safe

        key = "csk-CANARYDONOTLOGME123456789"
        for separator in (" ", "=", ": ", '"', "Bearer "):
            for offset in range(150, 200):
                line = ("x" * offset) + separator + key
                self.assertNotIn(
                    "csk-CANARY", str(_safe(line)),
                    "credential leaked at offset %d after %r" % (offset, separator),
                )

    def test_a_plain_record_message_is_redacted_too(self):
        """A record logged through the ordinary stdlib API reaches the same formatter."""
        from stepmold.log import TextFormatter, JsonFormatter

        record = logging.LogRecord(
            "stepmold.probe", logging.WARNING, "", 1, "key is %s",
            ("csk-CANARYDONOTLOG12345",), None,
        )
        for formatter in (TextFormatter(), JsonFormatter()):
            rendered = formatter.format(record)
            self.assertNotIn("csk-CANARY", rendered)
            self.assertIn("redacted", rendered)

    def test_control_characters_cannot_reach_the_terminal(self):
        """A raw ESC in upstream text would execute as a terminal command."""
        from stepmold.log import TextFormatter

        record = logging.LogRecord("stepmold.probe", logging.WARNING, "", 1, "x", (), None)
        setattr(record, "stepmold_event", "probe")
        setattr(record, "stepmold_fields", {"b": "\x1b[2J\x1b[31mRED"})
        rendered = TextFormatter().format(record)
        self.assertNotIn("\x1b", rendered)
        self.assertIn("RED", rendered)


if __name__ == "__main__":
    unittest.main()
