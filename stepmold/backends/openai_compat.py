"""An OpenAI-compatible `/v1/chat/completions` client, in `urllib`.

This is the adapter that points stepmold at a real small model. The same wire format is
spoken by llama.cpp-server, vLLM and SGLang, which is exactly the point: ARCHITECTURE.md §2's
"one base model, many packs" only works if swapping the server is a URL change.

Written but **never executed against a live server in this repo** — the test suite mocks
the HTTP layer and makes no network call, per TASKS.md T11. Treat it as unverified
against real servers until someone runs it against one.

Grammar handling differs per backend, so it is a flag rather than a guess:

    response_format   vLLM, SGLang, OpenAI  -> response_format.json_schema.schema
    json_schema       llama.cpp-server      -> a top-level "json_schema" field
    json_object       any server with only  -> response_format {"type": "json_object"},
                      loose JSON mode          schema appended to the prompt as text
    none              no server-side constraint at all

Even with `none`, stepmold still validates every output before committing it (see
`stepmold.verify`) — constrained decoding is an optimisation here, not the safety net.
"""

import datetime
import email.utils
import http.client
import json
import socket
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..errors import BackendError
from ..log import (BEARER, DEBUG, ERROR, INFO, KEY_SHAPED, WARNING, event,
                   get_logger, redact)

__all__ = [
    "DEFAULT_OPENER",
    "GRAMMAR_MODES",
    "NoCrossOriginRedirect",
    "OpenAICompatModel",
]

_log = get_logger("backend")

GRAMMAR_MODES = ("response_format", "json_schema", "json_object", "none")
API_KEY_VARIABLES = ("STEPMOLD_API_KEY", "OPENAI_API_KEY")
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
DEFAULT_PORTS = {"http": 80, "https": 443}

# The payload fields each grammar mode writes itself. `extra_body` is merged last, so a
# key in here coming from a caller would delete the constraint the mode just applied and
# say nothing about it — "a constraint you think you have and don't", which is the exact
# failure stepmold/grammar.py exists to prevent. Refused at construction instead.
GRAMMAR_FIELDS = {
    "response_format": ("messages", "response_format"),
    "json_schema": ("messages", "json_schema"),
    "json_object": ("messages", "response_format"),
    "none": ("messages",),
}

# A provider that answers `Retry-After: 3600` is telling the truth, but a run must not
# disappear into an hour-long sleep inside one node. Honour the instruction up to here.
RETRY_AFTER_CEILING = 60.0

# Credentials, as they appear in text somebody else wrote: a header a gateway echoed back
# at us, or a key pasted into an upstream error message. The patterns and the filter moved
# to `stepmold.log` when logging arrived — the same text now goes to two places, and one filter
# in front of both is the only version of this that stays true. Re-exported here because
# this module is where the redaction rule was written and where readers look for it.
_redact = redact


class NoCrossOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only while it stays on the origin the operator chose.

    urllib's stock handler copies the request headers onto the next hop, so the
    `Authorization: Bearer ...` this client attaches is delivered to whatever host the
    endpoint names in a `Location` — verified here for a 302, which urllib follows as a
    GET carrying the header. The endpoint is not necessarily trusted with the operator's
    key: `base_url` can come from a shared config or a proxy someone stood up. Chat
    completions needs no cross-origin hop, so refuse it by name rather than follow it
    silently and return the other host's reply as if it were the model's.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        here, there = _origin(req.full_url), _origin(newurl)
        if here != there:
            # Raised, not returned as None: returning None makes urllib hand the caller
            # the redirect response as if it were an answer, which reads like a
            # malformed reply instead of a refusal.
            raise BackendError(
                "refusing to follow an HTTP %s redirect from %s to %s: the request "
                "carries your API key, and a redirect must not move it to another "
                "origin" % (code, _text(here), _text(there))
            )
        return urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl
        )


def _origin(url):
    """(scheme, host, port) — what a redirect must keep identical."""
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:  # a Location with a nonsense port is not an origin we can match
        port = None
    return scheme, (parts.hostname or "").lower(), port or DEFAULT_PORTS.get(scheme)


def _text(origin):
    scheme, host, port = origin
    return "%s://%s:%s" % (scheme, host, port)


# Built once at import; constructing an opener performs no IO.
DEFAULT_OPENER = urllib.request.build_opener(NoCrossOriginRedirect)


@dataclass
class OpenAICompatModel:
    """A `Model` backed by an OpenAI-compatible chat completions endpoint.

    `opener` and `sleeper` exist so the HTTP layer can be replaced in a test. Nothing
    else in stepmold should ever need them. The default opener is not bare `urlopen`: it
    refuses cross-origin redirects so the API key cannot be walked off the chosen host
    (see `NoCrossOriginRedirect`).
    """

    base_url: str
    model: str
    # repr=False so the generated __repr__ cannot print it. The explicit __repr__ below
    # takes over anyway, but a model is exactly the object that lands in
    # logging.debug("%r", model), in a failed assertion's diff and in a crash reporter
    # walking locals, so the credential is hidden at the field as well as at the method.
    api_key: Optional[str] = field(default=None, repr=False)
    grammar_mode: str = "response_format"
    user_agent: str = "stepmold/%s" % __import__("stepmold").__version__
    reasoning_reserve: int = 0
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 2
    schema_name: str = "stepmold_node"
    extra_body: Dict[str, Any] = field(default_factory=dict)
    opener: Callable = DEFAULT_OPENER.open
    sleeper: Callable = time.sleep

    def __post_init__(self):
        if self.grammar_mode not in GRAMMAR_MODES:
            raise ValueError(
                "unknown grammar_mode %r (known: %s)"
                % (self.grammar_mode, ", ".join(GRAMMAR_MODES))
            )
        _refuse_grammar_collisions(self.grammar_mode, self.extra_body)
        self.url = _completions_url(self.base_url)
        if self.api_key is None:
            self.api_key = _api_key_from_environment()

    def __repr__(self):
        """Identify the model without printing the credential it carries.

        A dataclass repr prints every field, and a model is an ordinary object: it ends
        up in logs, tracebacks and pdb frames. The fingerprint is enough to tell "the
        wrong key is loaded" from "no key is loaded" and far too little to authenticate
        with. `opener` and `sleeper` are left out for the same reason a repr exists at
        all — they are noise at the point where somebody is reading one.
        """
        return "OpenAICompatModel(base_url=%r, model=%r, grammar_mode=%r, api_key=%s)" % (
            self.base_url, self.model, self.grammar_mode, _fingerprint(self.api_key)
        )

    # ------------------------------------------------------------------ Model protocol

    def generate(self, prompt, grammar=None, max_tokens=512, sampling=None):
        """Complete `prompt`, constrained by `grammar` in whichever way this server wants."""
        payload = self.build_payload(prompt, grammar, max_tokens, sampling=sampling)
        return _content(self._post(payload))

    # ------------------------------------------------------------------------ internals

    def build_payload(self, prompt, grammar, max_tokens, sampling=None):
        schema = _schema_of(grammar)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # A pack's max_tokens budgets the ANSWER, because a pack is portable and
            # cannot know which model will run it. A reasoning model bills its private
            # chain of thought against the same ceiling, so the model-specific headroom
            # is added here, in the backend that knows what it is talking to. Without
            # it, a node with a tight budget spends the whole allowance thinking and
            # returns finish_reason=length with content=None.
            "max_tokens": max_tokens + self.reasoning_reserve,
            "temperature": self.temperature,
        }
        # An independent draw is the whole point of a re-sample and of the agreement gate.
        # Without honouring this hint every extra draw is a byte-identical request, the
        # draws agree by construction, and the gate reports full confidence on a single
        # sample — which is worse than no gate, because it is confidence that was never
        # earned. `seed` matters as much as temperature: a server pinned to greedy
        # decoding is deterministic in its stream but still varies on seed.
        if sampling is not None:
            payload["temperature"] = sampling.temperature
            if sampling.seed is not None:
                payload["seed"] = sampling.seed
        if schema is not None:
            if self.grammar_mode == "response_format":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.schema_name,
                        "schema": schema,
                        # Claimed only when it is true of this schema — see
                        # `_strict_ready`. Claiming it unconditionally gets the whole
                        # request refused with a 400 that no ladder can absorb.
                        "strict": _strict_ready(schema),
                    },
                }
            elif self.grammar_mode == "json_schema":
                payload["json_schema"] = schema
            elif self.grammar_mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
                payload["messages"][0]["content"] = (
                    prompt
                    + "\n\nReply with JSON matching this schema:\n"
                    + json.dumps(schema, sort_keys=True)
                )
        payload.update(self.extra_body)
        return payload

    def _post(self, payload):
        body = json.dumps(payload).encode("utf-8")
        # An explicit User-Agent is not cosmetic. urllib defaults to "Python-urllib/X.Y",
        # which Cloudflare-fronted inference providers (Cerebras among them) reject with
        # HTTP 403 error 1010 — a browser-signature ban, before the request ever reaches
        # the model. Found by running this adapter against a real endpoint for the first
        # time; no mocked test could have caught it.
        headers = {"Content-Type": "application/json", "User-Agent": self.user_agent}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key

        last = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.url, data=body, headers=headers, method="POST"
            )
            asked_for = 0.0
            # One clock read per HTTP attempt, next to a call that is about to block on a
            # GPU for a fifth of a second. The cost of always taking it is not worth the
            # branch it would take to avoid it.
            started = time.monotonic()
            event(_log, DEBUG, "backend.request", model=self.model, endpoint=self.url,
                  attempt=attempt + 1, of=self.max_retries + 1,
                  grammar_mode=self.grammar_mode,
                  max_tokens=payload.get("max_tokens"), body_bytes=len(body))
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    # The read stays inside the try: a body shorter than its
                    # Content-Length fails here, not at the opener.
                    decoded = _decode(response.read(), self.url, _content_type(response))
                    status = getattr(response, "status", None) or 200
            except urllib.error.HTTPError as exc:
                detail = _read_error(exc, self.api_key)
                last = BackendError(
                    "%s returned HTTP %s: %s" % (self.url, exc.code, detail)
                )
                retryable = exc.code in RETRY_STATUSES
                # `detail` has already been through `_read_error`, which strips the key
                # we sent and anything else key-shaped out of somebody else's error body.
                event(_log, WARNING, "backend.http_error", model=self.model,
                      endpoint=self.url, status=exc.code, attempt=attempt + 1,
                      retryable=retryable, duration_ms=_ms(started), detail=detail)
                if not retryable:
                    raise last
                asked_for = _retry_after(exc)
            except urllib.error.URLError as exc:
                last = BackendError(
                    "could not reach %s: %s" % (self.url, _redact(str(exc.reason)))
                )
                event(_log, WARNING, "backend.transport_error", model=self.model,
                      endpoint=self.url, attempt=attempt + 1, error="URLError",
                      duration_ms=_ms(started), detail=_redact(str(exc.reason)))
            except (OSError, http.client.HTTPException) as exc:
                # urllib wraps only what HTTPConnection.request() raises. Everything from
                # getresponse() and response.read() — a peer that hangs up, a truncated
                # body, a read that outlives the timeout — used to escape stepmold raw: not a
                # StepmoldError, never retried, and a bare traceback out of `stepmold run`. These
                # are the most transient failures stepmold can meet, so they ride the same
                # ladder as a 503.
                last = _transport_error(self.url, self.timeout, exc)
                event(_log, WARNING, "backend.transport_error", model=self.model,
                      endpoint=self.url, attempt=attempt + 1,
                      error=type(exc).__name__, duration_ms=_ms(started),
                      detail=_redact(str(exc)))
            else:
                # An `else` rather than a `return` inside the `try`, so the accounting
                # below runs on a request that actually completed without widening what
                # the except clauses above are watching.
                if _log.isEnabledFor(INFO):
                    # `_usage` walks the response, so it is built only when something is
                    # listening. This is the line that answers "what did that cost".
                    event(_log, INFO, "backend.response", model=self.model,
                          endpoint=self.url, status=status, attempt=attempt + 1,
                          retries=attempt, duration_ms=_ms(started), **_usage(decoded))
                return decoded
            if attempt < self.max_retries:
                # The provider's instruction wins when it is longer than our own backoff;
                # our backoff wins when the provider said nothing.
                delay = max(0.5 * (2 ** attempt), asked_for)
                # The slept seconds, not the schedule: a run that spent nine seconds
                # inside one node spent them here, and only the actual number says so.
                event(_log, INFO, "backend.backoff", endpoint=self.url,
                      attempt=attempt + 1, retry_after=asked_for, slept_s=round(delay, 3))
                self.sleeper(delay)
        event(_log, ERROR, "backend.failed", model=self.model, endpoint=self.url,
              attempts=self.max_retries + 1, detail=str(last))
        raise last


def _completions_url(base_url):
    """Accept a host, a `/v1`, or the full endpoint — all three are what people paste."""
    trimmed = (base_url or "").rstrip("/")
    if not trimmed:
        raise ValueError("base_url is required, e.g. http://localhost:8000")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return trimmed + "/chat/completions"
    return trimmed + "/v1/chat/completions"


def _api_key_from_environment():
    for name in API_KEY_VARIABLES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _fingerprint(key):
    """A key you can recognise but not use."""
    if not key:
        return "None"
    if len(key) < 12:
        # Too short for a prefix and a suffix to leave anything unguessable behind.
        return "'<set>'"
    return "'%s...%s'" % (key[:3], key[-3:])


def _refuse_grammar_collisions(grammar_mode, extra_body):
    """Refuse an `extra_body` key that the chosen grammar mode owns.

    `extra_body` is merged last so an operator can reach a server knob stepmold has no field
    for. Merged last also means it wins, and the fields the grammar mode writes are
    exactly the ones that must not be quietly lost: the request still succeeds, the model
    answers unconstrained, and nothing anywhere says the node lost its grammar. Refusing
    at construction is the same bargain `grammar_mode` itself already makes — say no
    before the first request rather than send a wrong one forever.
    """
    owned = GRAMMAR_FIELDS[grammar_mode]
    clashing = sorted(key for key in (extra_body or {}) if key in owned)
    if clashing:
        raise ValueError(
            "extra_body may not set %s: grammar_mode %r builds those fields itself, and "
            "extra_body is merged last, so the constraint would be overwritten with no "
            "error at all. Drop the key, or choose the grammar_mode that sends what you "
            "want (known: %s)."
            % (", ".join(repr(key) for key in clashing), grammar_mode,
               ", ".join(GRAMMAR_MODES))
        )


def _strict_ready(schema):
    """True when `schema` satisfies OpenAI-family strict structured output.

    Strict mode is not a superset of JSON Schema: every object must set
    `additionalProperties: false` and name every declared property in `required`.
    `stepmold.grammar.check_schema` requires neither — optional properties are a deliberate
    part of stepmold's subset — so a pack that validates, evals green against a FakeModel and
    ships could still be refused with an HTTP 400 that no ladder absorbs (400 is not
    retryable, and `BackendError` is not `NodeFailed`, so `on_fail` never fires either).

    Claiming strictness only when it holds, rather than normalising the schema into it,
    is the choice made here for two reasons: closing an object and marking every property
    required changes the contract the pack declared (an optional field becomes one the
    model must emit), and doing it safely would mean copying the pack's grammar dict on
    every call to avoid editing it in place. Nothing is lost by not claiming it — the
    schema still goes on the wire, servers that constrain decoding still use it, and
    `stepmold.verify` checks the output against the pack's schema either way.
    """
    if not isinstance(schema, dict):
        return True
    declared = schema.get("type")
    types = [declared] if isinstance(declared, str) else list(declared or [])
    if "object" in types:
        if schema.get("additionalProperties") is not False:
            return False
        properties = schema.get("properties") or {}
        if set(properties) - set(schema.get("required") or []):
            return False
        if not all(_strict_ready(sub) for sub in properties.values()):
            return False
    if "array" in types and isinstance(schema.get("items"), dict):
        return _strict_ready(schema["items"])
    return True


def _retry_after(exc):
    """How long the provider asked us to wait, in seconds, or 0.0 if it did not.

    A 429 backing off on stepmold's own schedule is a provider instruction being ignored: the
    first retry goes out at 0.5s when the header said 1s, and a `Retry-After: 60` still
    gets three requests inside 1.5 seconds — which is how an account gets banned rather
    than rate limited.
    """
    headers = getattr(exc, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    seconds = _delta_seconds(value.strip()) if value else None
    if seconds is None:
        return 0.0
    return max(0.0, min(seconds, RETRY_AFTER_CEILING))


def _delta_seconds(value):
    """Both spellings RFC 7231 allows: `120`, and `Wed, 21 Oct 2015 07:28:00 GMT`."""
    try:
        return float(int(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:  # a date with no zone is UTC, per the spec
        when = when.replace(tzinfo=datetime.timezone.utc)
    return (when - datetime.datetime.now(datetime.timezone.utc)).total_seconds()


def _transport_error(url, timeout, exc):
    """Name a socket-level failure the way an HTTP status is named.

    "RemoteDisconnected" on its own tells an operator nothing about which endpoint died
    or what to change; `_no_content_reason` sets the standard for the rest of this file.
    """
    # socket.timeout only became an alias of TimeoutError in Python 3.10; on 3.9 it is a
    # plain OSError, so checking TimeoutError alone let a read timeout fall through to the
    # generic message and lose the one number the operator needs — the timeout itself.
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return BackendError(
            "%s did not answer within the %ss timeout. Raise the backend's `timeout` if "
            "the model is simply slow to start emitting tokens." % (url, timeout)
        )
    if isinstance(exc, http.client.IncompleteRead):
        # `expected` counts the bytes still OUTSTANDING, not the whole body, so the
        # Content-Length the server promised is the sum of the two.
        arrived = len(exc.partial)
        promised = "%d" % (arrived + exc.expected) if exc.expected is not None else "more"
        return BackendError(
            "%s sent a truncated response body: %d bytes arrived of the %s it promised "
            "in Content-Length, then the connection closed."
            % (url, arrived, promised)
        )
    return BackendError(
        "the connection to %s failed before a complete response arrived (%s: %s)"
        % (url, type(exc).__name__, _redact(str(exc)) or "no detail")
    )


def _content_type(response):
    """The response's Content-Type, if this response object has headers at all."""
    headers = getattr(response, "headers", None)
    return headers.get("Content-Type") if headers is not None else None


def _schema_of(grammar):
    """Unwrap `stepmold.grammar.schema_to_grammar`'s struct, or take a bare schema."""
    if grammar is None:
        return None
    if isinstance(grammar, dict) and "schema" in grammar:
        return grammar["schema"]
    return grammar


def _decode(raw, url=None, content_type=None):
    """Parse a completion body, and say enough about a body that is not one.

    The old message was `"not JSON (Expecting value: line 1 column 1 (char 0))"`, which
    names neither the endpoint nor a byte of what arrived — an operator cannot tell a
    proxy's HTML 502 page from a model that returned an empty string. Both are HTTP 200
    here, and only the body says which.
    """
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    try:
        return json.loads(text)
    except ValueError as exc:
        raise BackendError(
            "%s returned a body that is not JSON (%s). Content-Type was %s; the body "
            "began: %s"
            % (url or "the backend", exc, content_type or "not stated",
               _clip(_redact(text)))
        )


def _clip(text, limit=200):
    stripped = (text or "").strip()
    if not stripped:
        return "<empty>"
    return repr(stripped[:limit] + ("..." if len(stripped) > limit else ""))


def _content(response):
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise _empty(
            "backend returned no choices: %s" % _clip(_redact(json.dumps(response)))
        )
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = choice.get("text")
    if not isinstance(content, str):
        raise _empty(_no_content_reason(response, choice, message))
    return content


def _empty(detail):
    """A 200 that carried no text: a bad draw, not an endpoint that is down.

    `verify.run_node` reads `empty_content` and turns such an error into a rejection that
    spends one rung of the ladder, so a node with an `on_fail` edge diverts instead of
    aborting the run. `verify.EmptyCompletion` documents that contract and names this
    module as marking its errors — which this module did not actually do, so the shipped
    backend aborted on the first content-less answer while the docs promised a retry.
    Found by audit. The diagnostic is kept verbatim: it is what an operator needs, and
    `_no_content_reason` is the part that says a reasoning model ate its own budget.
    """
    error = BackendError(detail)
    error.empty_content = True
    return error


def _no_content_reason(response, choice, message):
    """Say *why* a choice carried no text, not merely that it did not.

    The common cause on a reasoning model is a budget the private chain of thought ate
    before any answer was emitted: finish_reason "length", a populated `reasoning`
    field, and `content` null. Diagnosing that from "no text content" costs an hour, so
    name it and say what to change.
    """
    finish = choice.get("finish_reason")
    usage = response.get("usage") or {}
    detail = usage.get("completion_tokens_details") or {}
    thought = detail.get("reasoning_tokens")
    reasoning = message.get("reasoning")
    if finish == "length" and (thought or reasoning):
        return (
            "the model spent its whole token budget reasoning and emitted no answer "
            "(finish_reason=length, %s reasoning tokens, content=null). This is a "
            "reasoning model: its private chain of thought is billed against the same "
            "ceiling as the answer. Raise the backend's reasoning_reserve (or the "
            "node's max_tokens) and run again."
            % (thought if thought is not None else "some")
        )
    if finish == "length":
        return (
            "the model hit its token ceiling before emitting content "
            "(finish_reason=length). Raise the node's max_tokens."
        )
    if reasoning:
        return (
            "the backend returned reasoning but no content (finish_reason=%r). Some "
            "reasoning models only fill `message.reasoning` when they stop early."
            % finish
        )
    return "backend returned a choice with no text content (finish_reason=%r)" % finish


def _read_error(exc, api_key=None):
    """The upstream's own words about the failure, with the credentials taken out.

    An error body is written by somebody else and read by our logs, so it is filtered
    before it is quoted: the key we sent (a gateway may echo the request back), plus
    anything else bearer- or key-shaped that was already in there.
    """
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        return exc.reason if getattr(exc, "reason", None) else "<no body>"
    if api_key:
        body = body.replace(api_key, "<redacted>")
    return _clip(_redact(body))


def _ms(started):
    """Milliseconds since `started`, at a resolution a log line can hold."""
    return round((time.monotonic() - started) * 1000.0, 1)


def _usage(response):
    """What this call cost, from the provider's own accounting.

    Every field is optional on the wire — llama.cpp-server omits the reasoning breakdown
    that Cerebras fills in — so a missing number is reported as None rather than guessed
    at or dropped. A stable set of field names is what makes a day of these lines
    summable; a field that vanishes when it is zero is not.
    """
    usage = response.get("usage") or {} if isinstance(response, dict) else {}
    detail = usage.get("completion_tokens_details") or {}
    choices = response.get("choices") if isinstance(response, dict) else None
    first = choices[0] if choices else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": detail.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
    }
