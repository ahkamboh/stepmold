"""The four grammar modes, driven over a real socket.

`jig.backends.openai_compat.GRAMMAR_MODES` names four ways to ask an OpenAI-compatible
server for constrained output. Exactly one of them — `response_format` — has ever been
executed against a live server. The other three were written from a spec and read back
by a mocked test that asserted the same dict the author had just typed. That is the
shape of the two defects first contact with a real endpoint already produced (the
`User-Agent` 403 and the reasoning-budget `content: null`), so this file assumes there
are more and goes looking.

What is different here from `tests/test_backend.py`:

* every request crosses a real loopback socket into `tests/production/faultproxy.py`,
  so `http.client` is in the loop and payload framing is real;
* the proxy answers *as the server family the mode targets would*: it finds the schema
  wherever that mode put it (in the payload, or appended to the prompt) and produces
  output shaped by it. That turns "the dict has the right keys" into "a server that
  reads this payload can actually satisfy the node";
* `json_object` and `none` return **unconstrained** text, which is the whole point of
  those modes existing. The retry ladder is the only thing standing between them and a
  corrupt commit, so it is measured, not assumed.

NO NETWORK. The proxy is constructed with no upstream, so it answers by itself and
never opens an outbound connection.

Tests whose name or comment says DEFECT document behaviour that is wrong. They assert
what jig does today so the suite stays honest; fixing the defect is expected to fail
them, and the comment says what the assertion should become. Three of those defects are
now fixed and their tests assert the fix instead: `extra_body` can no longer overwrite
the grammar, `strict: true` is claimed only for a schema that satisfies it, and a socket
that dies mid-response is a retryable `BackendError`. The one left is the latent
aliasing defect in `ThePackIsNeverEditedByBuildingAPayload`.
"""

import copy
import http.client
import json
import os
import unittest

from jig.backends.openai_compat import GRAMMAR_MODES, OpenAICompatModel
from jig.errors import BackendError, JigError, NodeFailed
from jig.grammar import check_schema, schema_to_grammar, validate_against
from jig.graph import run
from jig.pack import Edge, Node, Pack
from jig.verify import run_node

from .faultproxy import FaultProxy

# The exact wording `build_payload` uses to smuggle the schema into the prompt in
# `json_object` mode. Hard-coded on purpose: it is the only channel that mode has for a
# node's contract, so a silent reword is a behaviour change and should break a test.
SCHEMA_PREAMBLE = "\n\nReply with JSON matching this schema:\n"

# A closed schema whose only field is an enum: every kind of rejection the ladder knows
# (unparseable, wrong type, out-of-enum, missing required) is reachable from it.
ENUM_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": ["billing", "technical"]}},
    "required": ["category"],
    "additionalProperties": False,
}

# The same schema jig's own invariant suite uses (tests/test_invariants.py OPEN_SCHEMA):
# no `additionalProperties`, and a property that is not required. `check_schema` accepts
# it. OpenAI-family strict structured output does not. See TheStrictFlagIsUnconditional.
OPEN_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}, "note": {"type": "string"}},
    "required": ["category"],
}

CONSTRAINED_MODES = ("response_format", "json_schema", "json_object")


# --------------------------------------------------------------------------- the proxy


class Recorder(FaultProxy):
    """A `FaultProxy` that keeps the whole payload and can read a mode's schema.

    Two additions, both needed to say anything true about the unconstrained modes:

    * **the full payload per call.** `FaultProxy._record` keeps a summary; asserting a
      wire format needs the bytes that actually arrived.
    * **schema discovery per mode.** The stock proxy looks for a schema in
      `response_format.json_schema.schema` or a top-level `json_schema`. `json_object`
      mode puts it in the *prompt* instead, so to the stock proxy that mode looks
      identical to a request with no schema at all — every answer would be `{}` and
      every test would "prove" the mode is broken. A server told the schema in prose
      can honour it, so this one does, by parsing the block `build_payload` appended.
      `none` mode is left alone: it genuinely sends the schema nowhere, and pretending
      otherwise would hide the most interesting result in this file.

    `strict_server` emulates the schema validation an OpenAI-family server performs
    before it will accept `strict: true`.
    """

    strict_server = False

    def _record(self, fault, payload, headers):
        FaultProxy._record(self, fault, payload, headers)
        with self._lock:
            self.calls[-1]["payload"] = copy.deepcopy(payload)
            self.calls[-1]["headers"] = dict(headers)

    def _respond(self, handler, fault, payload):
        if self.strict_server:
            # Only a payload that actually *claims* strictness is held to the strict
            # rules — which is the point: `json_schema` mode sends the same schema with
            # no such claim, so the same server accepts it.
            envelope = (payload.get("response_format") or {}).get("json_schema") or {}
            problems = _strict_violations(envelope.get("schema")) if envelope.get(
                "strict") else []
            if problems:
                return self._send(handler, 400, {"error": {
                    "message": "Invalid schema for response_format 'jig_node': %s"
                               % "; ".join(problems)}})
        return FaultProxy._respond(self, handler, fault, _with_prompt_schema(payload))

    # ------------------------------------------------------------------ test helpers

    @property
    def payload(self):
        return self.calls[-1]["payload"]

    def prompt(self, index=-1):
        return (self.calls[index]["payload"].get("messages") or [{}])[0].get("content", "")

    def ask(self, mode, prompt="Classify: help", schema=ENUM_SCHEMA, faults=("ok",),
            **kwargs):
        """One `generate` in `mode`, returning the model's text (or raising)."""
        self.script(list(faults))
        return self.client(mode, **kwargs).generate(
            prompt, grammar=schema_to_grammar(schema) if schema else None, max_tokens=32
        )

    def client(self, mode, **kwargs):
        options = {
            "base_url": self.base_url,
            "model": "small-local-model",
            "api_key": "test-key-not-a-secret",
            "grammar_mode": mode,
            "max_retries": 0,
            # No wall-clock anywhere in this file: the backend's backoff is stubbed out.
            "sleeper": lambda seconds: None,
        }
        options.update(kwargs)
        return OpenAICompatModel(**options)


def _with_prompt_schema(payload):
    """Give the proxy the schema `json_object` mode appended to the prompt.

    Returned as a patched copy so the recorded payload stays exactly what arrived.
    """
    if _wire_schema(payload) is not None:
        return payload
    content = (payload.get("messages") or [{}])[0].get("content", "")
    if SCHEMA_PREAMBLE not in content:
        return payload
    try:
        found = json.loads(content.rsplit(SCHEMA_PREAMBLE, 1)[1])
    except ValueError:
        return payload
    if not isinstance(found, dict):
        return payload
    patched = dict(payload)
    patched["json_schema"] = found
    return patched


def _wire_schema(payload):
    """The schema this payload carries as a *field*, or None."""
    fmt = payload.get("response_format")
    if isinstance(fmt, dict):
        inner = fmt.get("json_schema")
        if isinstance(inner, dict) and isinstance(inner.get("schema"), dict):
            return inner["schema"]
    if isinstance(payload.get("json_schema"), dict):
        return payload["json_schema"]
    return None


def _prompt_schema(content):
    """The schema `json_object` mode appended to a prompt, or None."""
    if SCHEMA_PREAMBLE not in content:
        return None
    try:
        return json.loads(content.rsplit(SCHEMA_PREAMBLE, 1)[1])
    except ValueError:
        return None


def _strict_violations(schema, path=""):
    """What an OpenAI-family server objects to before it will honour `strict: true`.

    Two documented requirements, both structural: every object must set
    `additionalProperties: false`, and every declared property must appear in
    `required`. A schema that breaks either is rejected with HTTP 400 *before* any
    token is generated — the whole node dies, not just the sample.
    """
    if schema is None or not isinstance(schema, dict):
        return []
    where = path or "<root>"
    problems = []
    declared = schema.get("type")
    types = [declared] if isinstance(declared, str) else list(declared or [])
    if "object" in types:
        if schema.get("additionalProperties") is not False:
            problems.append("%s: 'additionalProperties' must be false" % where)
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        for name in sorted(set(properties) - required):
            problems.append("%s: %r must be in 'required'" % (where, name))
        for name, sub in sorted(properties.items()):
            problems.extend(_strict_violations(sub, "%s.%s" % (path, name) if path else name))
    if "array" in types and isinstance(schema.get("items"), dict):
        problems.extend(_strict_violations(schema["items"], path + "[]"))
    return problems


PROXY = None


def setUpModule():
    """One proxy for the whole module.

    `ThreadingHTTPServer.shutdown()` waits out `serve_forever`'s poll interval, so a
    proxy per test costs half a second per test. One start, one stop, and every test
    re-scripts it in `setUp`.
    """
    global PROXY
    PROXY = Recorder().start()


def tearDownModule():
    if PROXY is not None:
        PROXY.stop()


class ProxyTest(unittest.TestCase):
    """Base class: hand every test a clean, non-strict proxy."""

    def setUp(self):
        self.proxy = PROXY
        self.proxy.strict_server = False
        self.proxy.script(["ok"])

    def tearDown(self):
        self.proxy.strict_server = False


# ------------------------------------------------------------------ 1. the wire format


class TheWirePayloadMatchesTheServerFamily(ProxyTest):
    """Each mode names a server family. Assert what that family actually reads.

    The mapping is the module docstring of `jig/backends/openai_compat.py`:

        response_format  vLLM, SGLang, OpenAI  -> response_format.json_schema.schema
        json_schema      llama.cpp-server      -> a top-level "json_schema" field
        json_object      loose JSON mode       -> response_format {"type":"json_object"}
                                                  + the schema appended to the prompt
        none             -> nothing

    A mode that puts the schema in the wrong place fails open: the server ignores the
    field it does not know, returns unconstrained text, and nothing anywhere reports a
    problem. That is the failure `jig/grammar.py` calls "a constraint you think you have
    and don't", so it is asserted field by field rather than by shape.
    """

    def test_response_format_mode_nests_the_schema_where_vllm_reads_it(self):
        self.proxy.ask("response_format")
        payload = self.proxy.payload
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], ENUM_SCHEMA)
        self.assertEqual(payload["response_format"]["json_schema"]["name"], "jig_node")
        self.assertIs(payload["response_format"]["json_schema"]["strict"], True)
        # The llama.cpp spelling must not also be present: two constraints, one of which
        # the server silently ignores, is how a mode gets "verified" against the wrong one.
        self.assertNotIn("json_schema", payload)

    def test_json_schema_mode_puts_the_schema_at_the_top_level_for_llama_cpp(self):
        self.proxy.ask("json_schema")
        payload = self.proxy.payload
        self.assertEqual(payload["json_schema"], ENUM_SCHEMA)
        self.assertNotIn("response_format", payload)
        # llama.cpp-server takes the schema bare — not wrapped in a name/strict envelope.
        self.assertNotIn("strict", payload)

    def test_json_object_mode_asks_for_loose_json_and_describes_the_schema_in_words(self):
        self.proxy.ask("json_object")
        payload = self.proxy.payload
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("json_schema", payload)
        content = self.proxy.prompt()
        self.assertTrue(content.startswith("Classify: help"))
        self.assertEqual(_prompt_schema(content), ENUM_SCHEMA)

    def test_none_mode_sends_no_constraint_by_any_channel(self):
        self.proxy.ask("none")
        payload = self.proxy.payload
        self.assertNotIn("response_format", payload)
        self.assertNotIn("json_schema", payload)
        self.assertNotIn("grammar", payload)
        self.assertEqual(self.proxy.prompt(), "Classify: help")

    def test_no_grammar_means_no_constraint_in_any_mode(self):
        for mode in GRAMMAR_MODES:
            self.proxy.script(["ok"])
            self.proxy.client(mode).generate("hello", grammar=None, max_tokens=16)
            payload = self.proxy.payload
            self.assertNotIn("response_format", payload, mode)
            self.assertNotIn("json_schema", payload, mode)
            self.assertEqual(self.proxy.prompt(), "hello", mode)

    def test_every_mode_that_is_not_none_puts_a_constraint_on_the_wire(self):
        """The guard for the next mode somebody adds to GRAMMAR_MODES.

        `build_payload` dispatches on the mode name with an if/elif chain and no else.
        A mode added to the tuple but not to the chain is accepted at construction and
        then quietly sends an unconstrained request forever. This test names the whole
        set, so adding a mode without a branch fails here instead of in production.
        """
        carried = set()
        for mode in GRAMMAR_MODES:
            self.proxy.script(["ok"])
            self.proxy.client(mode).generate(
                "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16
            )
            payload = self.proxy.payload
            if _wire_schema(payload) is not None or _prompt_schema(
                self.proxy.prompt()
            ) is not None:
                carried.add(mode)
        self.assertEqual(carried, set(GRAMMAR_MODES) - {"none"})
        self.assertEqual(carried, set(CONSTRAINED_MODES))

    def test_the_common_fields_are_identical_across_modes(self):
        """The grammar mode changes the constraint and nothing else."""
        common = []
        for mode in GRAMMAR_MODES:
            self.proxy.script(["ok"])
            self.proxy.client(mode, temperature=0.3).generate(
                "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=48
            )
            payload = self.proxy.payload
            common.append((payload["model"], payload["max_tokens"], payload["temperature"],
                           len(payload["messages"]), payload["messages"][0]["role"]))
        self.assertEqual(
            common, [("small-local-model", 48, 0.3, 1, "user")] * len(GRAMMAR_MODES)
        )

    def test_the_user_agent_survives_the_socket_in_every_mode(self):
        """The 403 that first contact found is a header, not a payload — check it here."""
        for mode in GRAMMAR_MODES:
            self.proxy.ask(mode)
            agent = self.proxy.calls[-1]["user_agent"]
            self.assertTrue(agent.startswith("jig/"), mode)
            self.assertNotIn("python-urllib", agent.lower(), mode)


class TheSchemaOnTheWireIsTheSchemaJigVerifies(ProxyTest):
    """Whatever the mode, the server is told the same contract the verifier enforces.

    A mode that reshapes the schema on the way out (drops a keyword the server does not
    like, say) would leave the model constrained by one contract and judged by another —
    every generation legal at the server and rejected at the verifier.
    """

    def test_each_mode_transmits_the_node_schema_unaltered(self):
        for mode in CONSTRAINED_MODES:
            self.proxy.ask(mode, schema=ENUM_SCHEMA)
            payload = self.proxy.payload
            found = _wire_schema(payload)
            if found is None:
                found = _prompt_schema(self.proxy.prompt())
            self.assertEqual(found, ENUM_SCHEMA, mode)

    def test_a_richer_schema_survives_the_json_object_round_trip(self):
        """The prompt channel is text, so it is the one that can lose fidelity."""
        rich = {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": ["p0", "p1"],
                             "description": "how urgent, with a {brace} in the text"},
                "order_id": {"type": ["string", "null"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["priority", "order_id", "tags"],
            "additionalProperties": False,
        }
        check_schema(rich)  # jig accepts it, so a pack can ship it
        self.proxy.ask("json_object", schema=rich)
        self.assertEqual(_prompt_schema(self.proxy.prompt()), rich)

    def test_braces_in_the_appended_schema_are_never_re_rendered(self):
        """The schema is appended *after* template rendering, so its braces are inert.

        `jig.render` substitutes `{name}` from state. The schema is JSON: it is nothing
        but braces, and a `description` may hold a literal `{placeholder}`. Appending it
        inside the renderer would either explode on a missing variable or splice run
        state into the schema. It is appended in the backend instead, downstream of the
        renderer — this is the test that says so.
        """
        schema = {
            "type": "object",
            "properties": {"category": {"type": "string",
                                        "description": "not {ticket}, a literal brace"}},
            "required": ["category"],
            "additionalProperties": False,
        }
        node = _node(grammar=schema, prompt="Classify: {ticket}", retries=0)
        self.proxy.script(["ok"])
        run_node(node, {"ticket": "SECRET-STATE"}, self.proxy.client("json_object"))
        content = self.proxy.prompt()
        self.assertEqual(_prompt_schema(content), schema)
        # The rendered half saw state; the appended half did not.
        self.assertIn("Classify: SECRET-STATE", content)
        self.assertIn("{ticket}", content.rsplit(SCHEMA_PREAMBLE, 1)[1])


# ------------------------------------------------------------- 2. construction guards


class AnUnknownGrammarModeIsRefused(ProxyTest):
    """Refused at construction, before a socket exists — not at the first request.

    A mode name is a deployment decision, usually a CLI flag typed once. Discovering a
    typo on the first generation means discovering it after a pack has been loaded and a
    prompt rendered; discovering it at construction means the process never starts.
    """

    def test_construction_rejects_it_and_names_every_known_mode(self):
        with self.assertRaises(ValueError) as caught:
            self.proxy.client("telepathy")
        message = str(caught.exception)
        self.assertIn("telepathy", message)
        for name in GRAMMAR_MODES:
            self.assertIn(name, message)

    def test_no_request_is_made_while_finding_out(self):
        before = len(self.proxy.calls)
        with self.assertRaises(ValueError):
            self.proxy.client("telepathy")
        self.assertEqual(len(self.proxy.calls), before)

    def test_case_and_spelling_variants_are_all_refused(self):
        for name in ("Response_Format", "RESPONSE_FORMAT", "response-format",
                     "json object", "json_schema ", "", None, "grammar"):
            with self.assertRaises(ValueError):
                self.proxy.client(name)

    def test_the_cli_spec_path_refuses_it_too(self):
        """`--model openai:URL#name#mode` is where a typo actually gets typed."""
        from jig.cli import resolve_model

        with self.assertRaises(ValueError) as caught:
            resolve_model("openai:%s#m#telepathy" % self.proxy.base_url, pack=None)
        self.assertIn("telepathy", str(caught.exception))

    def test_the_cli_spec_path_accepts_every_declared_mode(self):
        from jig.cli import resolve_model

        for mode in GRAMMAR_MODES:
            model = resolve_model("openai:%s#m#%s" % (self.proxy.base_url, mode),
                                  pack=None)
            self.assertEqual(model.grammar_mode, mode)


# ------------------------------------------- 3. the unconstrained modes and the ladder


def _node(**kwargs):
    base = dict(name="classify", type="generate", prompt="Classify: {ticket}",
                grammar=ENUM_SCHEMA, output="result", retries=2)
    base.update(kwargs)
    return Node(**base)


def _survives(proxy, mode, faults, node=None, **model_kwargs):
    """Run one node against a scripted server. True if a verified value was committed."""
    proxy.script(list(faults), default=faults[-1])
    try:
        run_node(node or _node(), {"ticket": "help"}, proxy.client(mode, **model_kwargs))
        return True
    except (NodeFailed, BackendError):
        return False


class TheLadderIsTheOnlyThingHoldingUpTheUnconstrainedModes(ProxyTest):
    """`json_object` and `none` return whatever the model felt like. Measure the rescue.

    With `response_format` or `json_schema` the server is doing the work and the ladder
    is insurance. With these two the server is doing nothing, so every claim in
    `jig/verify.py` — parse, validate, re-sample with feedback — is load-bearing on the
    first attempt, every time.

    The proxy answers as a server that honours whatever schema the mode gave it, which
    is why `json_object` can succeed at all: its schema travels in the prompt.
    """

    def test_prose_around_the_json_is_rescued_in_every_mode(self):
        """The single most common unconstrained failure: a chat model being chatty."""
        for mode in GRAMMAR_MODES:
            self.proxy.script(["prose"])
            text = self.proxy.client(mode).generate(
                "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=32
            )
            self.assertIn("Sure!", text)  # the server really did wrap it in prose
            self.assertEqual(_first_verified(text), {"category": "billing"}
                             if mode in CONSTRAINED_MODES else {})

    def test_a_fenced_block_still_commits_in_json_object_mode(self):
        self.assertTrue(_survives(self.proxy, "json_object", ["prose"], _node(retries=0)))

    def test_flat_refusal_prose_exhausts_the_ladder_rather_than_committing(self):
        node = _node(retries=2)
        self.assertFalse(_survives(self.proxy, "json_object", ["garbage"] * 4, node))
        # Three generations: the first plus two re-samples. Not four, not one.
        self.assertEqual(len(self.proxy.calls), 3)

    def test_the_ladder_rescues_a_late_recovery_in_every_mode(self):
        """Two bad samples then a good one is exactly what retries=2 buys."""
        for mode in CONSTRAINED_MODES:
            self.assertTrue(
                _survives(self.proxy, mode, ["garbage", "badschema", "ok"]), mode
            )
            self.assertEqual(len(self.proxy.calls), 3, mode)

    def test_a_recovery_one_sample_too_late_is_not_rescued(self):
        for mode in CONSTRAINED_MODES:
            self.assertFalse(
                _survives(self.proxy, mode, ["garbage", "garbage", "garbage", "ok"]), mode
            )

    def test_a_wider_ladder_rescues_what_the_default_cannot(self):
        node = _node(retries=4)
        self.assertTrue(_survives(self.proxy, "json_object",
                                  ["garbage", "garbage", "garbage", "badschema", "ok"],
                                  node))
        self.assertEqual(len(self.proxy.calls), 5)

    def test_nothing_unverified_is_ever_returned_by_the_ladder(self):
        """Whatever survives satisfies the node's schema, in every mode."""
        for mode in GRAMMAR_MODES:
            for faults in (["ok"], ["prose"], ["badschema", "ok"], ["garbage", "prose"]):
                self.proxy.script(list(faults), default=faults[-1])
                try:
                    value = run_node(_node(), {"ticket": "help"},
                                     self.proxy.client(mode))
                except NodeFailed:
                    continue
                validate_against(ENUM_SCHEMA, value)  # raises if the guarantee is broken


def _first_verified(text):
    """What `jig.verify.extract_json` would pull out of a generation."""
    from jig.verify import extract_json

    return extract_json(text)


class TheSurvivalRateOfEachMode(ProxyTest):
    """Measured, not assumed: which modes survive which server, and how often.

    The matrix is a fixed set of three-sample fault scripts against a server that
    honours whatever schema the mode handed it. `response_format`, `json_schema` and
    `json_object` all reach the model with the contract, so they behave alike.
    `none` reaches it with nothing.
    """

    SCRIPTS = (
        ("ok", "ok", "ok"),
        ("prose", "prose", "prose"),
        ("badschema", "ok", "ok"),
        ("badschema", "badschema", "ok"),
        ("badschema", "badschema", "badschema"),
        ("garbage", "ok", "ok"),
        ("garbage", "garbage", "prose"),
        ("garbage", "garbage", "garbage"),
    )

    def _matrix(self):
        table = {}
        for mode in GRAMMAR_MODES:
            table[mode] = tuple(
                _survives(self.proxy, mode, list(script)) for script in self.SCRIPTS
            )
        return table

    def test_the_three_constrained_modes_survive_identically(self):
        table = self._matrix()
        expected = (True, True, True, True, False, True, True, False)
        for mode in CONSTRAINED_MODES:
            self.assertEqual(table[mode], expected, mode)

    def test_none_mode_survives_nothing(self):
        """DEFECT (design gap): `none` hands the schema to no one, by any channel.

        `json_object` mode already carries the fallback — append the schema to the
        prompt — and `none` does not use it, so a `none`-mode node's only hope is that
        the pack's prompt text happens to spell out the JSON shape. Against a server
        that knows only what jig told it, the survival rate is zero *for every script*,
        including the three where the server was willing to cooperate.

        SHOULD BE: `none` means "no *server-side* constraint", not "no contract" —
        the prompt-appended schema belongs here too, and this row should match the
        other three. Asserting the zero row so the gap cannot close unnoticed.
        """
        self.assertEqual(self._matrix()["none"], (False,) * len(self.SCRIPTS))

    def test_a_run_that_survives_is_always_a_verified_run(self):
        """Survival never means "committed something wrong" — count both halves."""
        survived = 0
        for mode in GRAMMAR_MODES:
            for script in self.SCRIPTS:
                self.proxy.script(list(script), default=script[-1])
                try:
                    value = run_node(_node(), {"ticket": "help"},
                                     self.proxy.client(mode))
                except NodeFailed:
                    continue
                survived += 1
                validate_against(ENUM_SCHEMA, value)
        self.assertEqual(survived, 18)  # 6 per constrained mode, 0 for `none`


class TheRunRoutesRatherThanCrashes(ProxyTest):
    """A node the ladder cannot rescue takes its `on_fail` edge — in every mode.

    This is the run-level half of the measurement: not "did the node produce a value"
    but "did the workflow reach an end node with its state intact".
    """

    def _pack(self, **node_kwargs):
        classify = _node(on_fail="needs_human", **node_kwargs)
        return Pack(
            path="<memory>", name="grammar-modes", version=1, entry="classify",
            model=None,
            nodes={
                "classify": classify,
                "done": Node(name="done", type="end", output=["result"]),
                "needs_human": Node(name="needs_human", type="end", output=["ticket"]),
            },
            edges=[Edge("classify", "done")],
        )

    def test_an_unrescuable_node_lands_on_the_failure_edge_in_every_mode(self):
        for mode in GRAMMAR_MODES:
            self.proxy.script(["garbage"] * 4, default="garbage")
            result = run(self._pack(), self.proxy.client(mode), {"ticket": "help"})
            self.assertEqual(result.end_node, "needs_human", mode)
            self.assertEqual([f.node for f in result.failures], ["classify"], mode)
            self.assertEqual(result.failures[0].attempts, 3, mode)
            self.assertNotIn("result", result.state, mode)

    def test_a_rescued_node_reaches_the_happy_end_in_the_constrained_modes(self):
        for mode in CONSTRAINED_MODES:
            self.proxy.script(["garbage", "prose"], default="ok")
            result = run(self._pack(), self.proxy.client(mode), {"ticket": "help"})
            self.assertEqual(result.end_node, "done", mode)
            self.assertEqual(result.output, {"result": {"category": "billing"}}, mode)
            self.assertEqual(result.failures, [], mode)

    def test_the_rejected_text_never_reaches_the_committed_state(self):
        """verify.py rule 1, checked over a real socket rather than a FakeModel."""
        self.proxy.script(["garbage", "garbage", "prose"], default="ok")
        result = run(self._pack(), self.proxy.client("json_object"), {"ticket": "help"})
        self.assertEqual(result.state["result"], {"category": "billing"})
        self.assertNotIn("I'm sorry", json.dumps(result.state))

    def test_the_failure_reason_is_kept_for_the_operator(self):
        self.proxy.script(["badschema"] * 4, default="badschema")
        result = run(self._pack(), self.proxy.client("none"), {"ticket": "help"})
        self.assertEqual(result.end_node, "needs_human")
        self.assertIn("schema:", result.failures[0].reason)


# ---------------------------------------------------- 4. two stages, and the feedback


class TheAppendedSchemaSurvivesThinkThenEmit(ProxyTest):
    """`json_object` mode plus `two_stage: true` — two calls, one of them constrained.

    The think stage is meant to be unconstrained (that is the entire point of splitting
    it out, docs/PLAN.md Bug 2). If the mode's schema block leaked into the think
    prompt, the think stage would start formatting instead of reasoning and the split
    would buy nothing.
    """

    NODE = dict(two_stage=True, think_max_tokens=64, max_tokens=48, retries=2)

    def test_the_think_call_carries_no_constraint_by_either_channel(self):
        self.proxy.script(["ok", "ok"])
        run_node(_node(**self.NODE), {"ticket": "help"},
                 self.proxy.client("json_object"))
        think = self.proxy.calls[0]["payload"]
        self.assertNotIn("response_format", think)
        self.assertNotIn("json_schema", think)
        self.assertIsNone(_prompt_schema(self.proxy.prompt(0)))
        self.assertIn("notes only", self.proxy.prompt(0))

    def test_the_emit_call_carries_both_halves(self):
        self.proxy.script(["ok", "ok"])
        run_node(_node(**self.NODE), {"ticket": "help"},
                 self.proxy.client("json_object"))
        emit = self.proxy.calls[1]["payload"]
        self.assertEqual(emit["response_format"], {"type": "json_object"})
        self.assertEqual(_prompt_schema(self.proxy.prompt(1)), ENUM_SCHEMA)

    def test_each_stage_gets_its_own_token_budget(self):
        self.proxy.script(["ok", "ok"])
        run_node(_node(**self.NODE), {"ticket": "help"},
                 self.proxy.client("json_object"))
        self.assertEqual([call["max_tokens"] for call in self.proxy.calls], [64, 48])

    def test_the_schema_is_still_there_after_the_ladder_appends_a_correction(self):
        """Every rung keeps the contract — including the ones that also carry feedback."""
        self.proxy.script(["ok", "garbage", "garbage", "garbage"], default="garbage")
        with self.assertRaises(NodeFailed):
            run_node(_node(**self.NODE), {"ticket": "help"},
                     self.proxy.client("json_object"))
        self.assertEqual(len(self.proxy.calls), 4)  # one think, three emits
        for index in (1, 2, 3):
            self.assertEqual(_prompt_schema(self.proxy.prompt(index)), ENUM_SCHEMA, index)

    def test_the_correction_is_present_from_the_second_emit_onwards(self):
        self.proxy.script(["ok", "badschema", "badschema", "badschema"],
                          default="badschema")
        with self.assertRaises(NodeFailed):
            run_node(_node(**self.NODE), {"ticket": "help"},
                     self.proxy.client("json_object"))
        self.assertNotIn("previous answer was rejected", self.proxy.prompt(1))
        for index in (2, 3):
            self.assertIn("previous answer was rejected", self.proxy.prompt(index))

    def test_the_correction_never_quotes_the_rejected_value(self):
        """verify.py rule 2 over a real socket, in the mode where the prompt is the API.

        `badschema` returns `"definitely-not-in-the-enum"`. The retry prompt may say the
        value was not one of the allowed choices; it may not repeat what the model said.
        """
        self.proxy.script(["ok", "badschema", "badschema", "badschema"],
                          default="badschema")
        with self.assertRaises(NodeFailed):
            run_node(_node(**self.NODE), {"ticket": "help"},
                     self.proxy.client("json_object"))
        for index in (2, 3):
            self.assertNotIn("definitely-not-in-the-enum", self.proxy.prompt(index))
            self.assertIn("not one of", self.proxy.prompt(index))

    def test_the_think_stage_is_paid_for_once_across_the_whole_ladder(self):
        """The scratchpad is reused; a re-sample re-rolls only the cheap half."""
        self.proxy.script(["ok", "garbage", "garbage", "garbage"], default="garbage")
        with self.assertRaises(NodeFailed):
            run_node(_node(**self.NODE), {"ticket": "help"},
                     self.proxy.client("json_object"))
        thinking = [call for call in self.proxy.calls if call["max_tokens"] == 64]
        self.assertEqual(len(thinking), 1)

    def test_the_schema_block_is_appended_after_the_volatile_correction(self):
        """DEFECT (minor, prompt ordering): the most stable block is placed last.

        `jig/codegen.py` states the rule explicitly — "Volatile content goes last. That
        is a prefix-cache decision" — and orders the emit prompt stable-first: template,
        then scratchpad, then correction. `build_payload` then appends the schema, the
        single most stable block in the whole prompt, *after* all of it. Every retry
        therefore re-sends the schema at a different offset behind changing text.

        SHOULD BE: the schema is a property of the node, not of the attempt, so it
        belongs immediately after the rendered template and before the scratchpad and
        the correction. Asserting today's order so a fix has to come here and say so.
        """
        self.proxy.script(["ok", "badschema", "badschema"], default="badschema")
        with self.assertRaises(NodeFailed):
            run_node(_node(retries=1, **{k: v for k, v in self.NODE.items()
                                         if k != "retries"}),
                     {"ticket": "help"}, self.proxy.client("json_object"))
        retry = self.proxy.prompt(2)
        self.assertLess(retry.index("previous answer was rejected"),
                        retry.index(SCHEMA_PREAMBLE.strip()))


# ---------------------------------------------------- 5. what the payload does to state


class ThePackIsNeverEditedByBuildingAPayload(ProxyTest):
    """tests/test_invariants.py claims a run never edits its pack. Check the backend.

    A pack's grammar dict is one object shared by every call to that node, for the life
    of the process. A backend that normalised it in place — sorted a `required` list,
    injected `additionalProperties: false` to satisfy strict mode — would corrupt every
    later call and every later run, and the corruption would be invisible: the pack on
    disk is still right.
    """

    def test_a_hundred_calls_leave_the_node_schema_byte_identical(self):
        for mode in GRAMMAR_MODES:
            schema = copy.deepcopy(ENUM_SCHEMA)
            node = _node(grammar=schema, retries=0)
            before = json.dumps(schema, sort_keys=True)
            model = self.proxy.client(mode)
            for _ in range(25):
                self.proxy.script(["ok"])
                try:
                    run_node(node, {"ticket": "help"}, model)
                except NodeFailed:
                    pass
            self.assertEqual(json.dumps(schema, sort_keys=True), before, mode)

    def test_two_runs_of_the_same_node_send_identical_payloads(self):
        """Drift between call one and call two is how in-place editing shows up."""
        for mode in GRAMMAR_MODES:
            node = _node(retries=0)
            sent = []
            model = self.proxy.client(mode)
            for _ in range(3):
                self.proxy.script(["ok"])
                try:
                    run_node(node, {"ticket": "help"}, model)
                except NodeFailed:
                    pass
                sent.append(json.dumps(self.proxy.payload, sort_keys=True))
            self.assertEqual(len(set(sent)), 1, mode)

    def test_the_payload_aliases_the_callers_schema_rather_than_copying_it(self):
        """DEFECT (latent): the schema goes into the payload by reference.

        Nothing in `build_payload` mutates it today, so the invariant above holds — but
        only because `jig.grammar.schema_to_grammar` deep-copies the schema one layer
        up, on every call. The backend's other documented entry point does not go
        through that copy: `generate(prompt, grammar=<bare schema>)` is supported and
        tested (tests/test_backend.py), and there the payload aliases the caller's own
        dict. Any future normalisation step in this function — the obvious one being
        injecting `additionalProperties: false` for strict mode, see below — would edit
        a pack's grammar in place through that alias.

        SHOULD BE: `build_payload` takes its own copy of the schema, and this assertion
        becomes `assertIsNot`.
        """
        model = self.proxy.client("response_format")
        payload = model.build_payload("p", ENUM_SCHEMA, 32)
        self.assertIs(payload["response_format"]["json_schema"]["schema"], ENUM_SCHEMA)

        model = self.proxy.client("json_schema")
        payload = model.build_payload("p", ENUM_SCHEMA, 32)
        self.assertIs(payload["json_schema"], ENUM_SCHEMA)

    def test_mutating_a_built_payload_reaches_back_into_the_callers_schema(self):
        """The consequence of the alias above, demonstrated rather than described."""
        schema = copy.deepcopy(ENUM_SCHEMA)
        payload = self.proxy.client("response_format").build_payload("p", schema, 32)
        payload["response_format"]["json_schema"]["schema"]["properties"]["leaked"] = {
            "type": "string"
        }
        self.assertIn("leaked", schema["properties"])  # DEFECT: should still be absent


class ExtraBodyCannotSilentlyRemoveTheConstraint(ProxyTest):
    """`extra_body` used to be merged last and win, including over the grammar.

    `build_payload` ends with `payload.update(self.extra_body)`, so an operator passing
    `extra_body` for an unrelated server knob could overwrite the very field the grammar
    mode had just set. The request still succeeded, the model returned whatever it liked,
    and nothing reported that the node had lost its grammar — exactly the failure
    `jig/grammar.py` names: "a silently-ignored constraint is a constraint you think you
    have and don't".

    The fix refuses the collision at construction, the way an unknown `grammar_mode` is
    refused: a key that the chosen mode owns (`response_format`, `json_schema`,
    `messages`) is a `ValueError` before the first request, not a quiet loss on every
    request. Merging is still merging for every other key.
    """

    def test_a_response_format_collision_is_refused_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            self.proxy.client("response_format",
                              extra_body={"response_format": {"type": "text"}})
        self.assertIn("response_format", str(caught.exception))
        self.assertIn("extra_body", str(caught.exception))

    def test_a_llama_cpp_schema_collision_is_refused_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            self.proxy.client("json_schema", extra_body={"json_schema": None})
        self.assertIn("json_schema", str(caught.exception))

    def test_a_messages_collision_is_refused_in_every_mode(self):
        """`messages` carries the prompt in all four modes, and the schema in one."""
        for mode in GRAMMAR_MODES:
            with self.assertRaises(ValueError, msg=mode):
                self.proxy.client(
                    mode,
                    extra_body={"messages": [{"role": "user",
                                              "content": "ignore the schema"}]},
                )

    def test_the_refusal_happens_before_any_request_is_made(self):
        """The same bargain `grammar_mode` makes: say no before sending a wrong request."""
        before = len(self.proxy.calls)
        with self.assertRaises(ValueError):
            self.proxy.client("response_format", extra_body={"response_format": {}})
        self.assertEqual(len(self.proxy.calls), before)

    def test_a_mode_that_does_not_own_the_field_still_accepts_it(self):
        """`none` sends no constraint at all, so nothing of its own can be clobbered —
        and hand-rolling `response_format` there is a legitimate reason to use it."""
        self.proxy.script(["ok"])
        self.proxy.client(
            "none", extra_body={"response_format": {"type": "json_object"}}
        ).generate("p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
        self.assertEqual(self.proxy.payload["response_format"], {"type": "json_object"})

    def test_an_ordinary_knob_is_merged_without_disturbing_the_constraint(self):
        """The feature is fine; only the collision is not."""
        self.proxy.script(["ok"])
        self.proxy.client("response_format", extra_body={"top_p": 0.9}).generate(
            "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
        self.assertEqual(self.proxy.payload["top_p"], 0.9)
        self.assertEqual(_wire_schema(self.proxy.payload), ENUM_SCHEMA)


# --------------------------------------------------------------- 6. the strict flag


class TheStrictFlagIsClaimedOnlyWhenItHolds(ProxyTest):
    """`response_format` mode used to send `strict: true` for schemas that cannot be.

    OpenAI-family structured output only accepts `strict: true` when every object in the
    schema sets `additionalProperties: false` and lists every declared property in
    `required`. `jig.grammar.check_schema` requires neither — both keywords are optional,
    by design — so a pack that validated cleanly, evalled green against a FakeModel and
    shipped could be rejected with HTTP 400 by the default grammar mode on the default
    server family. `tests/test_invariants.py` uses exactly such a schema (OPEN_SCHEMA) for
    its own fixtures, so this was not a hypothetical shape.

    It was worse than a bad sample: a 400 is not in `RETRY_STATUSES`, so it was not
    retried, and `BackendError` is not `NodeFailed`, so the walker would not take the
    node's `on_fail` edge either. The first node of the run killed the run.

    The fix is `_strict_ready(schema)`: claim strictness only for a schema that satisfies
    the rules. The alternative — normalising the schema into strict shape — was rejected
    because closing an object and marking every property required *changes the contract
    the pack declared*: an optional field becomes one the model must emit, and jig would
    then be verifying against one schema while the server enforced another. Nothing is
    lost by not claiming it. The schema still goes on the wire in the same place, servers
    that constrain decoding still use it, and `jig.verify` checks every output against
    the pack's own schema either way.
    """

    def test_jig_accepts_a_schema_that_strict_mode_forbids(self):
        check_schema(OPEN_SCHEMA)  # jig is happy
        self.assertEqual(
            _strict_violations(OPEN_SCHEMA),
            ["<root>: 'additionalProperties' must be false",
             "<root>: 'note' must be in 'required'"],
        )

    def test_strict_is_not_asserted_for_it_but_the_schema_is_still_sent(self):
        """Not claiming strictness is not the same as dropping the constraint."""
        self.proxy.ask("response_format", schema=OPEN_SCHEMA)
        envelope = self.proxy.payload["response_format"]["json_schema"]
        self.assertIs(envelope["strict"], False)
        self.assertEqual(envelope["schema"], OPEN_SCHEMA)
        self.assertEqual(envelope["name"], "jig_node")

    def test_strict_is_still_asserted_for_a_schema_that_satisfies_it(self):
        """ENUM_SCHEMA is closed and fully required, so the claim is true of it."""
        self.proxy.ask("response_format", schema=ENUM_SCHEMA)
        envelope = self.proxy.payload["response_format"]["json_schema"]
        self.assertIs(envelope["strict"], True)

    def test_a_nested_object_that_is_open_disqualifies_the_whole_schema(self):
        """Strict is checked to the leaves; a server checks it that way too."""
        nested = {
            "type": "object",
            "properties": {"inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            }},
            "required": ["inner"],
            "additionalProperties": False,
        }
        self.proxy.ask("response_format", schema=nested)
        envelope = self.proxy.payload["response_format"]["json_schema"]
        self.assertIs(envelope["strict"], False)
        self.assertNotEqual(_strict_violations(nested), [])

    def test_a_strict_enforcing_server_now_accepts_the_request(self):
        """The 400 that killed the run: gone, because the claim is gone."""
        self.proxy.strict_server = True
        self.proxy.script(["ok"])
        text = self.proxy.client("response_format").generate(
            "p", grammar=schema_to_grammar(OPEN_SCHEMA), max_tokens=16)
        self.assertEqual(json.loads(text), {"category": "x"})
        self.assertEqual(self.proxy.calls[-1]["fault"], "ok")

    def test_the_node_completes_instead_of_dying_on_its_first_sample(self):
        self.proxy.strict_server = True
        self.proxy.script(["ok"] * 4)
        before = len(self.proxy.calls)
        committed = run_node(_node(grammar=OPEN_SCHEMA, retries=2), {"ticket": "help"},
                             self.proxy.client("response_format"))
        self.assertEqual(committed, {"category": "x"})
        self.assertEqual(len(self.proxy.calls) - before, 1)

    def test_a_whole_run_over_an_open_schema_reaches_its_happy_end(self):
        """The shape of the original defect: the first node used to kill the run."""
        self.proxy.strict_server = True
        self.proxy.script(["ok"] * 4)
        pack = Pack(
            path="<memory>", name="strict", version=1, entry="classify", model=None,
            nodes={
                "classify": _node(grammar=OPEN_SCHEMA, on_fail="needs_human"),
                "done": Node(name="done", type="end", output=["result"]),
                "needs_human": Node(name="needs_human", type="end", output=["ticket"]),
            },
            edges=[Edge("classify", "done")],
        )
        result = run(pack, self.proxy.client("response_format"), {"ticket": "help"})
        self.assertEqual(result.path, ["classify", "done"])

    def test_a_server_that_does_enforce_strict_still_refuses_a_false_claim(self):
        """Guard the premise: the proxy's strict server is not a no-op.

        Without this, every assertion above would pass just as well against a server that
        never checked anything.
        """
        self.proxy.strict_server = True
        self.proxy.script(["ok"])
        model = self.proxy.client("response_format")
        payload = model.build_payload("p", OPEN_SCHEMA, 16)
        payload["response_format"]["json_schema"]["strict"] = True  # lie on purpose
        with self.assertRaises(BackendError) as caught:
            model._post(payload)
        self.assertIn("400", str(caught.exception))
        self.assertIn("additionalProperties", str(caught.exception))

    def test_the_other_three_modes_are_unaffected_by_the_same_schema(self):
        """Only `response_format` claims strictness, so only it can be refused for it."""
        self.proxy.strict_server = True
        for mode in ("json_schema", "json_object", "none"):
            self.proxy.script(["ok"])
            self.proxy.client(mode).generate(
                "p", grammar=schema_to_grammar(OPEN_SCHEMA), max_tokens=16)
            self.assertEqual(self.proxy.calls[-1]["fault"], "ok", mode)

    def test_the_shipped_example_pack_is_still_strict_safe(self):
        """The example is the thing people copy — keep it legal on a real server."""
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(here, "..", "..", "examples", "support_triage", "grammars")
        names = sorted(name for name in os.listdir(root) if name.endswith(".json"))
        self.assertTrue(names)
        for name in names:
            with open(os.path.join(root, name)) as handle:
                schema = json.load(handle)
            self.assertEqual(_strict_violations(schema), [], name)
            self.assertIs(
                self.proxy.client("response_format").build_payload(
                    "p", schema, 16)["response_format"]["json_schema"]["strict"],
                True,
                name,
            )


# ----------------------------------------------------- 7. the socket under the modes


class ADeadSocketIsAWrappedRetryableBackendError(ProxyTest):
    """A connection that dies mid-response used to be neither retried nor wrapped.

    `_post` caught `urllib.error.HTTPError` and `urllib.error.URLError`. Neither covers a
    socket that dies *after* the response line: `http.client` raises `IncompleteRead`
    when a body is shorter than its `Content-Length`, and `RemoteDisconnected` when the
    peer closes before writing anything. Both come out of `response.read()` /
    `getresponse()` inside the `with self.opener(...)` block, and both walked straight
    past the handler — so they were not `JigError`s (the CLI reported them as an
    unhandled traceback rather than a diagnosed failure) and they were not retried at
    all, although a truncated body or a disconnect from an idle-timing-out proxy is the
    textbook transient, the exact thing `max_retries` exists for.

    `_post` now catches `OSError` and `http.client.HTTPException` around both the opener
    call and the read. Asserted for all four modes because the grammar mode makes no
    difference: it is the same socket.
    """

    def test_a_truncated_body_is_a_backend_error_in_every_mode(self):
        for mode in GRAMMAR_MODES:
            self.proxy.script(["truncated"], default="truncated")
            with self.assertRaises(BackendError, msg=mode) as caught:
                self.proxy.client(mode).generate(
                    "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
            self.assertIn("truncated", str(caught.exception), mode)

    def test_a_dropped_connection_is_a_backend_error_in_every_mode(self):
        for mode in GRAMMAR_MODES:
            self.proxy.script(["reset"], default="reset")
            with self.assertRaises(BackendError, msg=mode) as caught:
                self.proxy.client(mode).generate(
                    "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
            self.assertIn(self.proxy.base_url, str(caught.exception), mode)

    def test_both_are_jig_errors_so_the_cli_can_diagnose_them(self):
        for fault in ("truncated", "reset"):
            self.proxy.script([fault], default=fault)
            try:
                self.proxy.client("response_format").generate("p", None, 16)
            except BackendError as exc:
                self.assertIsInstance(exc, JigError)
            else:
                self.fail("expected a BackendError for fault %r" % fault)

    def test_both_are_retried_because_both_are_transient(self):
        """Fail once, then take the "ok" that is right there — one node attempt, not a
        dead run."""
        for fault in ("truncated", "reset"):
            self.proxy.script([fault, "ok", "ok", "ok"], default="ok")
            before = len(self.proxy.calls)
            self.proxy.client("response_format", max_retries=3).generate(
                "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
            self.assertEqual(len(self.proxy.calls) - before, 2, fault)

    def test_a_persistent_socket_death_still_gives_up_after_the_ladder(self):
        """Retrying a permanently dead endpoint forever is the other way to be wrong."""
        self.proxy.script([], default="reset")
        before = len(self.proxy.calls)
        with self.assertRaises(BackendError):
            self.proxy.client("response_format", max_retries=2).generate(
                "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
        self.assertEqual(len(self.proxy.calls) - before, 3)

    def test_a_5xx_by_contrast_is_wrapped_and_retried_in_every_mode(self):
        """The contrast that made the two above look like an oversight, not a policy."""
        for mode in GRAMMAR_MODES:
            self.proxy.script(["503", "ok"], default="ok")
            before = len(self.proxy.calls)
            self.proxy.client(mode, max_retries=2).generate(
                "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
            self.assertEqual(len(self.proxy.calls) - before, 2, mode)

    def test_a_gateway_html_page_is_wrapped_in_every_mode(self):
        """A CDN error page arrives as HTTP 200 — this one was always handled. Keep it."""
        for mode in GRAMMAR_MODES:
            self.proxy.script(["notjson"], default="notjson")
            with self.assertRaises(BackendError) as caught:
                self.proxy.client(mode).generate(
                    "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
            self.assertIn("not JSON", str(caught.exception), mode)


class TheReasoningBudgetDiagnosisSurvivesEveryMode(ProxyTest):
    """The defect first contact found, re-checked on the three unexercised modes.

    A reasoning model that spends its ceiling thinking returns `content: null`. The
    backend turns that into a diagnosis naming `reasoning_reserve`. That diagnosis lives
    in `_content`, downstream of the mode switch, so it should be identical for all
    four — but "should be" is what this file is for.
    """

    def test_content_null_after_thinking_names_the_fix_in_every_mode(self):
        for mode in GRAMMAR_MODES:
            self.proxy.script(["reasoning"], default="reasoning")
            with self.assertRaises(BackendError) as caught:
                self.proxy.client(mode).generate(
                    "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
            self.assertIn("reasoning_reserve", str(caught.exception), mode)

    def test_a_plain_length_stop_says_max_tokens_instead(self):
        for mode in GRAMMAR_MODES:
            self.proxy.script(["empty"], default="empty")
            with self.assertRaises(BackendError) as caught:
                self.proxy.client(mode).generate(
                    "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=16)
            self.assertIn("max_tokens", str(caught.exception), mode)

    def test_the_reserve_is_added_to_the_wire_budget_in_every_mode(self):
        for mode in GRAMMAR_MODES:
            self.proxy.script(["ok"])
            self.proxy.client(mode, reasoning_reserve=256).generate(
                "p", grammar=schema_to_grammar(ENUM_SCHEMA), max_tokens=48)
            self.assertEqual(self.proxy.calls[-1]["max_tokens"], 304, mode)


if __name__ == "__main__":
    unittest.main()
