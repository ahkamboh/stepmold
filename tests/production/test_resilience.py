"""Production resilience: jig driven end to end through every fault the proxy injects.

Everything else in `tests/` hands the runtime a `FakeModel` or a fake opener. This file
does not: it starts `tests/production/faultproxy.py` on a loopback socket, points a real
`OpenAICompatModel` at it, and walks a real graph through real HTTP. That is the only way
to reach the code paths that live *between* urllib and jig — a socket that dies mid-body,
a `Content-Length` that lies, a read that times out — because none of them can be
provoked by a fake opener that politely returns bytes.

Still offline by construction: the proxy answers by itself (no `upstream=`), binds
127.0.0.1 on an ephemeral port, and every model here is built with a `sleeper` that does
not sleep. Nothing reaches the network and nothing takes wall-clock time it does not need.

What the faults are for, and what each one *should* prove:

    ok, prose, garbage, badschema   200s. The node's own ladder owns these.
    429, 500, 503                   retryable statuses. The backend's ladder owns these.
    notjson, empty, reasoning       200s that are not answers. Nobody should retry them.
    truncated, reset, slow          the socket itself failing. jig handles these worst.

Five tests are marked `@unittest.expectedFailure`. Each one asserts what jig *should* do,
runs red today, and is named in a FINDING docstring next to it. When someone fixes the
defect the test becomes an unexpected success, which unittest reports as a failure — so
the fix cannot land without deleting the marker and the finding together. That is
deliberate: a defect that is merely commented about gets forgotten.
"""

import contextlib
import http.client
import io
import json
import os
import re
import sys
import traceback
import unittest
import urllib.error

from jig.backends.openai_compat import DEFAULT_OPENER, OpenAICompatModel
from jig.cli import main as cli_main
from jig.errors import BackendError, NodeFailed
from jig.graph import run
from jig.model import FakeModel, Model
from jig.pack import Edge, Node, Pack
from jig.state import Store, resume
from tests.production.faultproxy import FAULTS, FaultProxy


# A key shaped like a real one, so a grep for it in an error message is unambiguous. It
# is not a credential for anything: the proxy never checks Authorization.
FAKE_KEY = "sk-jig-FAKEKEY-DO-NOT-LEAK-1234567890"

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": ["billing", "technical"]}},
    "required": ["category"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

# The proxy's stand-in model answers an enum with its first member, so this is what a
# clean run through `ok` must commit.
SATISFYING = {"category": "billing"}

# Long enough that a client timeout of TIMEOUT fires first by a wide margin, short enough
# that the abandoned handler thread is gone before the module ends.
SLOW_TIMEOUT = 0.25

CLI_PACK = "tests/fixtures/cli_pack"


# ------------------------------------------------------------------------ one proxy

# `FaultProxy.stop()` costs half a second — `serve_forever` polls on a 0.5s interval, so
# `shutdown()` waits for the next tick. Starting one per test put 30 seconds of pure
# waiting into a suite that otherwise runs in under two, so the whole module shares a
# single proxy and re-scripts it per test. `script()` clears `calls` under the proxy's
# own lock, so each test still sees only its own requests.
PROXY = None


def setUpModule():
    global PROXY
    PROXY = FaultProxy().start()
    # A test that times out, or one served the `truncated`/`reset` faults, leaves the
    # handler writing to a socket the client already dropped. socketserver prints that
    # traceback to stderr, which buries the actual test output. Every other error still
    # gets printed.
    PROXY._server.handle_error = _quiet_handle_error


def tearDownModule():
    PROXY.stop()


def _quiet_handle_error(request, client_address):
    exception = sys.exc_info()[1]
    if isinstance(exception, (BrokenPipeError, ConnectionResetError)):
        return
    traceback.print_exc()


def use(fault):
    """Script the shared proxy to answer every call with `fault`, and hand it back.

    Scripting the *default* rather than a queue means a retry ladder cannot walk off the
    end of the script and accidentally succeed: a persistent fault stays persistent.
    """
    PROXY.script([], default=fault)
    return PROXY


def sequence(faults, default="ok"):
    PROXY.script(faults, default=default)
    return PROXY


# --------------------------------------------------------------------------- helpers


def model(**kwargs):
    """A real backend pointed at the shared proxy, with the clock removed.

    `sleeper` is a no-op so the retry ladder's 0.5s/1.0s backoff costs nothing here; the
    tests that care about the backoff record what it was *asked* to sleep instead.
    """
    options = {
        "base_url": PROXY.base_url,
        "model": "m",
        "api_key": FAKE_KEY,
        "sleeper": lambda seconds: None,
        "timeout": 5.0,
    }
    options.update(kwargs)
    return OpenAICompatModel(**options)


def generate(**kwargs):
    """One raw generation against the proxy, under the node's grammar."""
    return model(**kwargs).generate(
        "Classify: t", grammar={"kind": "json_schema", "schema": SCHEMA}, max_tokens=64
    )


def provoke(fault, **kwargs):
    """Run one generation through `fault`; return (text, exception, formatted traceback)."""
    use(fault)
    try:
        return generate(**kwargs), None, ""
    except BaseException as exc:  # noqa: BLE001 - catching everything is the point
        return None, exc, traceback.format_exc()


def recording_opener(seen):
    """The real opener, with every outgoing `Request` captured first.

    Needed because `FaultProxy._record` keeps only the User-Agent, and the highest-value
    assertion in this file is about a header it does not keep.
    """

    def opener(request, timeout=None):
        seen.append(request)
        return DEFAULT_OPENER.open(request, timeout=timeout)

    return opener


def classify_node(**kwargs):
    base = dict(
        name="classify",
        type="generate",
        prompt="Classify: {ticket}",
        grammar=SCHEMA,
        output="result",
        retries=2,
        on_fail="rescue",
    )
    base.update(kwargs)
    return Node(**base)


def one_node_pack(**kwargs):
    """classify -> done, with `rescue` as the declared failure edge."""
    nodes = [
        classify_node(**kwargs),
        Node(name="done", type="end", output=["result"]),
        Node(name="rescue", type="end"),
    ]
    return _pack(nodes, [Edge("classify", "done")])


def two_node_pack():
    """classify -> summarise -> done, so a fault can land on the *second* node."""
    nodes = [
        classify_node(retries=0, on_fail=None),
        Node(
            name="summarise",
            type="generate",
            prompt="Summarise: {ticket}",
            grammar=SUMMARY_SCHEMA,
            output="summary",
            retries=0,
        ),
        Node(name="done", type="end", output=["result", "summary"]),
    ]
    return _pack(nodes, [Edge("classify", "summarise"), Edge("summarise", "done")])


def _pack(nodes, edges):
    return Pack(
        path="<memory>",
        name="resilience",
        version=1,
        entry="classify",
        model=None,
        nodes={node.name: node for node in nodes},
        edges=edges,
        max_steps=20,
    )


# ------------------------------------------------------- the fault table (completeness)

# What jig actually does with each fault today, as measured: (verdict, HTTP calls made).
# This table is the map of the rest of the file. Every fault the proxy knows how to
# inject has a row, so nobody can add a fourteenth without deciding what the runtime is
# supposed to do with it.
#
#   "answers"  the call returns text; whatever is wrong is now the verifier's problem
#   "backend"  BackendError — jig's own type, with a message an operator can act on
#   "raw"      an exception from the standard library escapes jig entirely
BEHAVIOUR = {
    "ok": ("answers", 1),
    "prose": ("answers", 1),
    "garbage": ("answers", 1),
    "badschema": ("answers", 1),
    "429": ("backend", 3),
    "500": ("backend", 3),
    "503": ("backend", 3),
    "notjson": ("backend", 1),
    "empty": ("backend", 1),
    "reasoning": ("backend", 1),
    "truncated": ("raw", 1),
    "reset": ("raw", 1),
    "slow": ("raw", 1),
}


class TestEveryFaultIsCovered(unittest.TestCase):
    def test_the_table_names_every_fault_the_proxy_can_inject(self):
        self.assertEqual(sorted(BEHAVIOUR), sorted(FAULTS))

    def test_each_fault_behaves_the_way_the_table_says(self):
        """One sweep, so a change in any fault's handling shows up as a single clear diff."""
        measured = {}
        for fault in FAULTS:
            started = use(fault)
            try:
                generate(timeout=SLOW_TIMEOUT)
                verdict = "answers"
            except BackendError:
                verdict = "backend"
            except BaseException:  # noqa: BLE001
                verdict = "raw"
            measured[fault] = (verdict, len(started.calls))
        self.assertEqual(measured, BEHAVIOUR)


# ----------------------------------------------------------------------- the happy path


class TestOkOverRealSockets(unittest.TestCase):
    """`ok` is not a formality: it is the only proof the whole stack speaks HTTP at all."""

    def test_a_run_completes_and_commits_the_verified_output(self):
        started = use("ok")
        result = run(one_node_pack(), model(), {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "done"])
        self.assertEqual(result.output, {"result": SATISFYING})
        self.assertEqual(result.failures, [])
        self.assertEqual(len(started.calls), 1)

    def test_the_request_carries_the_schema_the_node_declared(self):
        started = use("ok")
        run(one_node_pack(), model(), {"ticket": "t"})
        call = started.calls[0]
        self.assertTrue(call["had_schema"])
        self.assertEqual(call["model"], "m")
        self.assertEqual(call["prompt"], "Classify: t")

    def test_the_user_agent_is_jigs_own(self):
        """Cerebras 403s "Python-urllib/*" before the request reaches a model.

        openai_compat.py sets an explicit User-Agent for exactly that reason. Asserting
        it on a live socket is the check that survives someone "simplifying" the header
        dict away.
        """
        started = use("ok")
        generate()
        agent = started.calls[0]["user_agent"]
        self.assertTrue(agent.startswith("jig/"), agent)
        self.assertNotIn("urllib", agent)

    def test_reasoning_reserve_is_added_to_the_nodes_budget_on_the_wire(self):
        """The pack budgets the answer; the backend adds this model's thinking room."""
        started = use("ok")
        generate(reasoning_reserve=256)
        self.assertEqual(started.calls[0]["max_tokens"], 64 + 256)


# ------------------------------------------------------------- retryable HTTP statuses


class TestRetryableStatuses(unittest.TestCase):
    """429/500/503 are the ladder's job, and the ladder does engage."""

    def test_a_retryable_status_is_tried_max_retries_plus_one_times(self):
        for fault in ("429", "500", "503"):
            with self.subTest(fault=fault):
                started = use(fault)
                with self.assertRaises(BackendError):
                    generate(max_retries=2)
                self.assertEqual(len(started.calls), 3)

    def test_max_retries_zero_means_exactly_one_attempt(self):
        started = use("500")
        with self.assertRaises(BackendError):
            generate(max_retries=0)
        self.assertEqual(len(started.calls), 1)

    def test_the_error_names_the_url_the_status_and_the_upstream_body(self):
        """_no_content_reason sets the standard: say what happened, not that it did."""
        started = use("503")
        with self.assertRaises(BackendError) as caught:
            generate()
        message = str(caught.exception)
        self.assertIn("503", message)
        self.assertIn(started.base_url, message)
        self.assertIn("upstream boom", message)

    def test_a_transient_failure_recovers_without_the_node_noticing(self):
        """Two 500s then an answer is one node attempt, not three."""
        started = sequence(["500", "500"], default="ok")
        result = run(one_node_pack(), model(), {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "done"])
        self.assertEqual(result.output, {"result": SATISFYING})
        self.assertEqual(len(started.calls), 3)

    def test_the_nodes_ladder_is_not_spent_on_a_backend_failure(self):
        """A persistent 500 costs 3 HTTP calls, not 3 node attempts x 3 retries.

        The two ladders must not multiply: `BackendError` is not `Rejected`, so
        `run_node` never re-samples on it.
        """
        started = use("500")
        with self.assertRaises(BackendError):
            run(one_node_pack(retries=2), model(max_retries=2), {"ticket": "t"})
        self.assertEqual(len(started.calls), 3)

    def test_a_non_retryable_status_is_not_retried(self):
        """404 is not in RETRY_STATUSES; hammering a missing model three times helps nobody."""
        opener = _StatusOpener(404, b'{"error": "no such model"}')
        with self.assertRaises(BackendError) as caught:
            model(opener=opener).generate("hi")
        self.assertEqual(opener.count, 1)
        self.assertIn("404", str(caught.exception))


class TestRateLimitBackoff(unittest.TestCase):
    """FINDING: `Retry-After` is parsed by nobody — `grep -r Retry-After jig/` is empty.

    The proxy answers 429 with `Retry-After: 1`. jig sleeps its own fixed 0.5s before the
    first retry — *sooner* than the provider asked — and 1.0s before the second. It is not
    a hot loop, which is the thing that gets an account banned, but it is a provider
    instruction being ignored, and a provider that says `Retry-After: 60` still gets three
    requests inside 1.5 seconds.
    """

    def test_the_server_does_send_a_retry_after_header(self):
        """Guard the premise: without this the test below would prove nothing."""
        opener = _CapturingOpener()
        use("429")
        with self.assertRaises(BackendError):
            generate(opener=opener, max_retries=0)
        self.assertEqual(opener.errors[0].headers.get("Retry-After"), "1")

    def test_the_backoff_is_fixed_exponential_and_ignores_retry_after(self):
        slept = []
        use("429")
        with self.assertRaises(BackendError):
            generate(sleeper=slept.append, max_retries=2)
        self.assertEqual(slept, [0.5, 1.0])

    def test_the_backoff_is_the_same_for_a_429_as_for_a_500(self):
        """Which is the defect stated a second way: the 429 path is not special at all."""
        rate_limited, server_error = [], []
        use("429")
        with self.assertRaises(BackendError):
            generate(sleeper=rate_limited.append, max_retries=2)
        use("500")
        with self.assertRaises(BackendError):
            generate(sleeper=server_error.append, max_retries=2)
        self.assertEqual(rate_limited, server_error)

    @unittest.expectedFailure
    def test_the_first_backoff_should_honour_retry_after(self):
        """What jig should do: wait at least as long as the provider asked."""
        slept = []
        use("429")
        with self.assertRaises(BackendError):
            generate(sleeper=slept.append, max_retries=2)
        self.assertGreaterEqual(slept[0], 1.0)


# ----------------------------------------------------- 200s that are not usable answers


class TestBodiesThatAreNotJson(unittest.TestCase):
    """`notjson` is a CDN's HTML error page returned with status 200."""

    def test_it_is_not_retried(self):
        """Correct: no number of retries turns an HTML page into a completion."""
        started = use("notjson")
        with self.assertRaises(BackendError):
            generate()
        self.assertEqual(len(started.calls), 1)

    def test_the_message_says_only_that_it_was_not_json(self):
        """FINDING (documented, not endorsed): the thinnest message jig produces.

        It names neither the endpoint nor a byte of the body, so an operator staring at
        "Expecting value: line 1 column 1 (char 0)" cannot tell a proxy's 502 page from a
        model that returned an empty string. Compare `_no_content_reason`, which names the
        cause AND the setting to change. What it SHOULD say: the URL, the content type,
        and a clipped prefix of the body.
        """
        started = use("notjson")
        with self.assertRaises(BackendError) as caught:
            generate()
        message = str(caught.exception)
        self.assertIn("not JSON", message)
        self.assertNotIn(started.base_url, message)
        self.assertNotIn("502 Bad Gateway", message)


class TestCompletionsWithNoContent(unittest.TestCase):
    """`empty` and `reasoning` are 200s whose `message.content` is null."""

    def test_a_reasoning_model_that_ran_out_of_room_is_diagnosed_by_name(self):
        use("reasoning")
        with self.assertRaises(BackendError) as caught:
            generate()
        message = str(caught.exception)
        self.assertIn("reasoning", message)
        self.assertIn("reasoning_reserve", message)
        self.assertIn("10", message)  # the reasoning_tokens the response reported

    def test_a_plain_length_stop_says_raise_max_tokens(self):
        use("empty")
        with self.assertRaises(BackendError) as caught:
            generate()
        self.assertIn("max_tokens", str(caught.exception))

    def test_neither_is_retried(self):
        for fault in ("empty", "reasoning"):
            with self.subTest(fault=fault):
                started = use(fault)
                with self.assertRaises(BackendError):
                    generate()
                self.assertEqual(len(started.calls), 1)

    def test_a_contentless_answer_aborts_the_run_past_its_on_fail_edge(self):
        """FINDING (documented, not endorsed): `on_fail` is not taken here.

        `graph.run` catches `NodeFailed` and `MissingVariable`; `BackendError` is neither,
        so it escapes past the failure edge the node declared. For a backend that is
        *down* that is defensible, and graph.py argues it deliberately. This is not that
        case: the endpoint answered 200, the model simply produced no text on this sample,
        and a re-sample is exactly the remedy the node's ladder exists for. As it stands,
        one flaky completion kills a long checkpointed workflow that declared a rescue
        path — and `reasoning` is the fault jig hit on its very first contact with a real
        endpoint, so it is not hypothetical.

        What SHOULD happen: a content-less 200 becomes a `Rejected`, the node re-samples,
        and a node that keeps getting nothing takes `on_fail` like any other spent ladder.
        """
        started = use("reasoning")
        with self.assertRaises(BackendError):
            run(one_node_pack(), model(), {"ticket": "t"})
        # One call: the node's three-rung ladder never got to spend a single rung.
        self.assertEqual(len(started.calls), 1)


# -------------------------------------------------------- 200s the verifier has to judge


class TestSchemaViolationsBelongToTheLadder(unittest.TestCase):
    """`badschema` is well-formed JSON that breaks the node's enum."""

    def test_the_ladder_spends_every_rung_then_takes_on_fail(self):
        started = use("badschema")
        result = run(one_node_pack(retries=2), model(), {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "rescue"])
        self.assertEqual(len(started.calls), 3)
        self.assertEqual(result.failures[0].attempts, 3)

    def test_the_rejected_value_never_lands_in_state(self):
        use("badschema")
        result = run(one_node_pack(), model(), {"ticket": "t"})
        self.assertNotIn("result", result.state)
        self.assertEqual(result.state, {"ticket": "t"})
        self.assertEqual(result.provenance, {})

    def test_the_failure_records_which_enum_member_was_violated(self):
        use("badschema")
        result = run(one_node_pack(), model(), {"ticket": "t"})
        reason = result.failures[0].reason
        self.assertIn("schema", reason)
        self.assertIn("definitely-not-in-the-enum", reason)

    def test_a_node_with_no_on_fail_raises_nodefailed_instead(self):
        use("badschema")
        with self.assertRaises(NodeFailed) as caught:
            run(one_node_pack(on_fail=None), model(), {"ticket": "t"})
        self.assertEqual(caught.exception.node, "classify")
        self.assertEqual(caught.exception.attempts, 3)

    def test_the_on_fail_hop_is_checkpointed_with_the_failure(self):
        """A run that was rescued must still say, on disk, what it was rescued from."""
        store = Store(":memory:")
        use("badschema")
        run(one_node_pack(), model(), {"ticket": "t"}, run_id="r", store=store)
        first = store.history("r")[0]
        self.assertEqual(first.node, "classify")
        self.assertEqual(first.next_node, "rescue")
        self.assertEqual(len(first.failures), 1)
        self.assertNotIn("result", first.state)


class TestForgivingExtraction(unittest.TestCase):
    """`prose` and `garbage` are what a model without grammar support actually returns."""

    def test_prose_wrapped_around_a_fenced_object_is_still_accepted(self):
        """extract_json's whole reason to exist, proved over the wire rather than against
        a hand-written string in a unit test."""
        started = use("prose")
        result = run(one_node_pack(), model(), {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "done"])
        self.assertEqual(result.output, {"result": SATISFYING})
        self.assertEqual(len(started.calls), 1)  # accepted first time, no rung spent

    def test_the_chatter_around_the_object_is_not_committed(self):
        use("prose")
        raw = generate()
        use("prose")
        result = run(one_node_pack(), model(), {"ticket": "t"})
        self.assertIn("Hope that helps!", raw)  # the proxy really did wrap it
        self.assertEqual(result.state["result"], SATISFYING)

    def test_a_refusal_with_no_object_in_it_is_rejected_and_routed(self):
        use("garbage")
        result = run(one_node_pack(), model(), {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "rescue"])
        self.assertIn("not valid JSON", result.failures[0].reason)

    def test_garbage_then_an_answer_recovers_inside_the_ladder(self):
        started = sequence(["garbage", "garbage"], default="ok")
        result = run(one_node_pack(), model(), {"ticket": "t"})
        self.assertEqual(result.path, ["classify", "done"])
        self.assertEqual(len(started.calls), 3)

    def test_the_refusal_text_is_never_quoted_back_over_the_wire(self):
        """The anti-self-conditioning invariant, checked on the bytes the server saw.

        tests/test_invariants.py proves this against a FakeModel. This proves the same
        thing about the prompts that actually left the process, which is the only place a
        leak would matter.
        """
        started = use("garbage")
        run(one_node_pack(), model(), {"ticket": "t"})
        prompts = [call["prompt"] for call in started.calls]
        self.assertEqual(len(prompts), 3)
        for prompt in prompts:
            self.assertNotIn("I'm sorry", prompt)
        # ...but the retry prompts must still say what was wrong.
        self.assertIn("not valid JSON", prompts[1])
        self.assertIn("not valid JSON", prompts[2])

    def test_a_rejected_enum_value_is_never_quoted_back_over_the_wire(self):
        """The same invariant for the other rejection path: a schema violation."""
        started = use("badschema")
        run(one_node_pack(), model(), {"ticket": "t"})
        for call in started.calls:
            self.assertNotIn("definitely-not-in-the-enum", call["prompt"])
        self.assertIn("schema:", started.calls[1]["prompt"])


# --------------------------------------------------------------- the socket itself dying


class TestSocketLevelFaults(unittest.TestCase):
    """FINDING: three faults escape jig as raw standard-library exceptions.

    `_post` turns `HTTPError` and `URLError` into `BackendError` and retries the retryable
    ones. It never sees these three, because urllib only wraps what
    `http.client.HTTPConnection.request()` raises — anything raised by `getresponse()` or
    by `response.read()` comes out untouched:

        reset      http.client.RemoteDisconnected
        truncated  http.client.IncompleteRead
        slow       builtins.TimeoutError  (socket.timeout, raised on read)

    Three consequences, each tested below: the caller gets an exception that is not a
    `JigError`, the retry ladder never engages for the most obviously transient failures
    jig can meet, and `jig run` dies with a traceback instead of a diagnosis.
    """

    def test_a_dead_socket_surfaces_as_a_raw_http_client_error(self):
        use("reset")
        with self.assertRaises(http.client.RemoteDisconnected):
            generate()

    def test_a_truncated_body_surfaces_as_a_raw_incomplete_read(self):
        use("truncated")
        with self.assertRaises(http.client.IncompleteRead):
            generate()

    def test_a_read_timeout_surfaces_as_a_raw_timeout_error(self):
        use("slow")
        with self.assertRaises(TimeoutError):
            generate(timeout=SLOW_TIMEOUT)

    def test_none_of_the_three_is_a_jig_error(self):
        """The contract every caller was given: what a run raises is a `JigError`."""
        from jig.errors import JigError

        for fault in ("reset", "truncated", "slow"):
            with self.subTest(fault=fault):
                _, exc, _ = provoke(fault, timeout=SLOW_TIMEOUT)
                self.assertIsNotNone(exc)
                self.assertNotIsInstance(exc, JigError)

    def test_none_of_the_three_is_retried(self):
        for fault in ("reset", "truncated", "slow"):
            with self.subTest(fault=fault):
                started = use(fault)
                try:
                    generate(timeout=SLOW_TIMEOUT, max_retries=2)
                except BaseException:  # noqa: BLE001
                    pass
                self.assertEqual(len(started.calls), 1)

    def test_the_cli_dies_with_a_traceback_instead_of_a_message(self):
        """`cli.main` catches PackError/JigError/ValidationError/ValueError. None of these
        is any of them, so the exception escapes `main` entirely: no `jig: ` line on
        stderr, no exit code 1, just a traceback out of the top of the process."""
        for fault, expected in (("reset", http.client.RemoteDisconnected),
                                ("truncated", http.client.IncompleteRead)):
            with self.subTest(fault=fault):
                started = use(fault)
                with self.assertRaises(expected):
                    cli_main([
                        "run", CLI_PACK,
                        "--input", '{"ticket": "charged twice"}',
                        "--model", "openai:%s#m" % started.base_url,
                    ])

    def test_the_cli_does_report_a_backend_error_cleanly(self):
        """The contrast that makes the test above a defect and not a style opinion."""
        started = use("500")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli_main([
                "run", CLI_PACK,
                "--input", '{"ticket": "charged twice"}',
                "--model", "openai:%s#m" % started.base_url,
            ])
        self.assertEqual(code, 1)
        self.assertTrue(stderr.getvalue().startswith("jig: "), stderr.getvalue())
        self.assertIn("BackendError", stderr.getvalue())

    @unittest.expectedFailure
    def test_a_dead_socket_should_be_a_backend_error(self):
        use("reset")
        with self.assertRaises(BackendError):
            generate()

    @unittest.expectedFailure
    def test_a_dead_socket_should_be_retried_like_any_other_transient_failure(self):
        """A connection closed with no response is the textbook retryable failure — and
        `503`, which is strictly less transient, already gets three tries."""
        started = use("reset")
        try:
            generate(max_retries=2)
        except BaseException:  # noqa: BLE001
            pass
        self.assertEqual(len(started.calls), 3)

    @unittest.expectedFailure
    def test_a_read_timeout_should_be_a_backend_error_naming_the_timeout(self):
        use("slow")
        with self.assertRaises(BackendError) as caught:
            generate(timeout=SLOW_TIMEOUT)
        self.assertIn(str(SLOW_TIMEOUT), str(caught.exception))

    def test_the_timeout_at_least_bounds_the_wait(self):
        """Whatever it raises, it does raise: a slow endpoint does not hang the run."""
        started = use("slow")
        with self.assertRaises(Exception):
            generate(timeout=SLOW_TIMEOUT)
        self.assertEqual(len(started.calls), 1)


# ----------------------------------------------------------------- state after a failure


class TestStateSurvivesAFailedNode(unittest.TestCase):
    """A half-failed run must not commit a partial node, and must be resumable."""

    def test_a_backend_failure_commits_nothing_for_the_node_it_hit(self):
        store = Store(":memory:")
        sequence(["ok"], default="500")
        with self.assertRaises(BackendError):
            run(two_node_pack(), model(), {"ticket": "t"}, run_id="r", store=store)
        history = store.history("r")
        self.assertEqual([cp.node for cp in history], ["classify"])
        self.assertEqual(history[-1].next_node, "summarise")
        self.assertNotIn("summary", history[-1].state)
        self.assertEqual(history[-1].state["result"], SATISFYING)

    def test_the_run_resumes_at_the_node_that_died_once_the_backend_recovers(self):
        store = Store(":memory:")
        pack = two_node_pack()
        started = sequence(["ok"], default="500")
        with self.assertRaises(BackendError):
            run(pack, model(), {"ticket": "t"}, run_id="r", store=store)
        self.assertEqual(len(started.calls), 4)  # one ok, then three 500s

        use("ok")
        result = resume(pack, model(), "r", store)
        self.assertEqual(result.output,
                         {"result": SATISFYING, "summary": {"summary": "x"}})
        # Exactly one generation on resume: `classify` was NOT re-run.
        self.assertEqual(len(started.calls), 1)

    def test_a_failed_first_node_leaves_no_checkpoint_at_all(self):
        store = Store(":memory:")
        use("500")
        with self.assertRaises(BackendError):
            run(two_node_pack(), model(), {"ticket": "t"}, run_id="r", store=store)
        self.assertEqual(store.history("r"), [])
        self.assertIsNone(store.latest("r"))

    def test_a_socket_death_leaves_the_same_consistent_state_as_a_500(self):
        """The raw-exception defect must not also corrupt the checkpoint chain."""
        store = Store(":memory:")
        sequence(["ok"], default="reset")
        with self.assertRaises(http.client.RemoteDisconnected):
            run(two_node_pack(), model(), {"ticket": "t"}, run_id="r", store=store)
        history = store.history("r")
        self.assertEqual([cp.node for cp in history], ["classify"])
        self.assertNotIn("summary", history[-1].state)


# ------------------------------------------------------------------ the key must not leak


class TestTheApiKeyNeverLeaks(unittest.TestCase):
    """The highest-value check here. A key in a log line is a key in an incident.

    Every fault is provoked with a recognisable key, and every surface an operator or a
    crash reporter could touch — `str`, `repr`, `args`, the formatted traceback — is
    grepped for it.
    """

    def test_the_key_really_is_on_the_wire(self):
        """Guard the premise: a sweep for a key that was never sent proves nothing."""
        seen = []
        use("ok")
        generate(opener=recording_opener(seen))
        self.assertEqual(seen[0].get_header("Authorization"), "Bearer %s" % FAKE_KEY)

    def test_no_error_from_any_fault_mentions_the_key(self):
        leaks = []
        for fault in FAULTS:
            _, exc, formatted = provoke(fault, timeout=SLOW_TIMEOUT)
            if exc is None:
                continue
            surfaces = [str(exc), repr(exc), repr(exc.args), formatted]
            if any(FAKE_KEY in surface for surface in surfaces):
                leaks.append(fault)
        self.assertEqual(leaks, [])

    def test_no_error_mentions_the_key_when_it_came_from_the_environment(self):
        """A key jig read out of JIG_API_KEY is the same secret as one passed in."""
        previous = os.environ.get("JIG_API_KEY")
        os.environ["JIG_API_KEY"] = FAKE_KEY
        try:
            use("500")
            with self.assertRaises(BackendError) as caught:
                model(api_key=None).generate("hi")
            self.assertNotIn(FAKE_KEY, str(caught.exception))
        finally:
            if previous is None:
                del os.environ["JIG_API_KEY"]
            else:
                os.environ["JIG_API_KEY"] = previous

    def test_the_key_is_never_put_in_the_request_body(self):
        """It belongs in a header. A body is what every proxy in between logs."""
        started = use("ok")
        generate()
        self.assertNotIn(FAKE_KEY, json.dumps(started.calls))

    def test_a_failed_run_records_no_key_in_its_checkpoints(self):
        store = Store(":memory:")
        use("badschema")
        run(one_node_pack(), model(), {"ticket": "t"}, run_id="r", store=store)
        dumped = json.dumps([cp.__dict__ for cp in store.history("r")], default=str)
        self.assertNotIn(FAKE_KEY, dumped)

    def test_nothing_that_looks_like_a_bearer_token_survives_a_run_failure(self):
        """A belt-and-braces sweep: no `Bearer <anything>` in any message we can provoke.

        Written as a pattern rather than the literal key, so a future rename of the header
        or a partially-masked key still trips it.
        """
        pattern = re.compile(r"Bearer\s+\S+")
        for fault in ("429", "500", "503", "notjson", "empty", "reasoning"):
            with self.subTest(fault=fault):
                _, exc, formatted = provoke(fault)
                self.assertIsNotNone(exc)
                self.assertIsNone(pattern.search(str(exc)))
                self.assertIsNone(pattern.search(formatted))

    def test_the_dataclass_repr_prints_the_key_in_full(self):
        """FINDING: `repr(OpenAICompatModel(...))` contains the API key verbatim.

        `OpenAICompatModel` is a plain `@dataclass`, so its generated `__repr__` prints
        every field, `api_key` included. Nothing in jig calls `repr` on a model today,
        which is the only reason this has not burned anyone yet — but a model is an
        ordinary object that ends up in `logging.debug("%r", model)`, in a failing
        assertion's diff, in a `pdb` frame dump, and in any crash reporter that walks
        locals. One `field(repr=False)` fixes it.

        This test asserts the defect so it cannot be fixed silently; the next test is the
        one that should pass.
        """
        printed = repr(model())
        self.assertIn(FAKE_KEY, printed)

    @unittest.expectedFailure
    def test_the_repr_should_redact_the_key(self):
        self.assertNotIn(FAKE_KEY, repr(model()))

    def test_an_upstream_error_body_is_copied_into_the_exception_verbatim(self):
        """FINDING (lower severity, same blast radius): `_read_error` splices 200 bytes of
        the upstream's error body into the message with no filtering.

        Gateways that echo the offending request back in a debug body — several do — put
        the caller's `Authorization` header in that body, and jig then puts it into an
        exception that gets logged. jig is not the leaker here, it is the amplifier, and a
        redaction pass over the body before it is interpolated costs one regex.
        """
        echoed = ('{"error": "bad request", "request_headers": '
                  '{"Authorization": "Bearer %s"}}' % FAKE_KEY).encode()
        opener = _StatusOpener(400, echoed)
        with self.assertRaises(BackendError) as caught:
            model(opener=opener).generate("hi")
        self.assertIn(FAKE_KEY, str(caught.exception))


# --------------------------------------------------------------------- the model contract


class TestTheBackendStillSatisfiesTheProtocol(unittest.TestCase):
    def test_a_live_backend_is_a_model(self):
        self.assertIsInstance(model(), Model)

    def test_a_run_over_http_produces_the_same_shape_as_one_over_a_fake(self):
        """The point of the `Model` protocol: swapping the transport changes nothing."""
        use("ok")
        live = run(one_node_pack(), model(), {"ticket": "t"})
        fake = run(one_node_pack(), FakeModel([json.dumps(SATISFYING)]), {"ticket": "t"})
        self.assertEqual(live.output, fake.output)
        self.assertEqual(live.path, fake.path)
        self.assertEqual(live.provenance, fake.provenance)


# --------------------------------------------------- openers for the two responses the
# proxy cannot produce: a non-retryable status, and an upstream body that echoes the
# caller's own headers back at it.


class _StatusOpener:
    """Answers every request with one `HTTPError`, counting the attempts."""

    def __init__(self, status, body):
        self.status = status
        self.body = body
        self.count = 0

    def __call__(self, request, timeout=None):
        self.count += 1
        raise urllib.error.HTTPError(
            request.full_url, self.status, "boom", {}, io.BytesIO(self.body)
        )


class _CapturingOpener:
    """The real opener, keeping the `HTTPError` so its response headers can be read."""

    def __init__(self):
        self.errors = []

    def __call__(self, request, timeout=None):
        try:
            return DEFAULT_OPENER.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            self.errors.append(exc)
            raise


if __name__ == "__main__":
    unittest.main()
