"""An OpenAI-compatible `/v1/chat/completions` client, in `urllib`.

This is the adapter that points jig at a real small model. The same wire format is
spoken by llama.cpp-server, vLLM and SGLang, which is exactly the point: PLAN.md §2's
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

Even with `none`, jig still validates every output before committing it (see
`jig.verify`) — constrained decoding is an optimisation here, not the safety net.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..errors import BackendError

__all__ = [
    "DEFAULT_OPENER",
    "GRAMMAR_MODES",
    "NoCrossOriginRedirect",
    "OpenAICompatModel",
]

GRAMMAR_MODES = ("response_format", "json_schema", "json_object", "none")
API_KEY_VARIABLES = ("JIG_API_KEY", "OPENAI_API_KEY")
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
DEFAULT_PORTS = {"http": 80, "https": 443}


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
    else in jig should ever need them. The default opener is not bare `urlopen`: it
    refuses cross-origin redirects so the API key cannot be walked off the chosen host
    (see `NoCrossOriginRedirect`).
    """

    base_url: str
    model: str
    api_key: Optional[str] = None
    grammar_mode: str = "response_format"
    user_agent: str = "jig/%s" % __import__("jig").__version__
    reasoning_reserve: int = 0
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 2
    schema_name: str = "jig_node"
    extra_body: Dict[str, Any] = field(default_factory=dict)
    opener: Callable = DEFAULT_OPENER.open
    sleeper: Callable = time.sleep

    def __post_init__(self):
        if self.grammar_mode not in GRAMMAR_MODES:
            raise ValueError(
                "unknown grammar_mode %r (known: %s)"
                % (self.grammar_mode, ", ".join(GRAMMAR_MODES))
            )
        self.url = _completions_url(self.base_url)
        if self.api_key is None:
            self.api_key = _api_key_from_environment()

    # ------------------------------------------------------------------ Model protocol

    def generate(self, prompt, grammar=None, max_tokens=512):
        """Complete `prompt`, constrained by `grammar` in whichever way this server wants."""
        payload = self.build_payload(prompt, grammar, max_tokens)
        return _content(self._post(payload))

    # ------------------------------------------------------------------------ internals

    def build_payload(self, prompt, grammar, max_tokens):
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
        if schema is not None:
            if self.grammar_mode == "response_format":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.schema_name,
                        "schema": schema,
                        "strict": True,
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
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    return _decode(response.read())
            except urllib.error.HTTPError as exc:
                detail = _read_error(exc)
                last = BackendError(
                    "%s returned HTTP %s: %s" % (self.url, exc.code, detail)
                )
                if exc.code not in RETRY_STATUSES:
                    raise last
            except urllib.error.URLError as exc:
                last = BackendError("could not reach %s: %s" % (self.url, exc.reason))
            if attempt < self.max_retries:
                self.sleeper(0.5 * (2 ** attempt))
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


def _schema_of(grammar):
    """Unwrap `jig.grammar.schema_to_grammar`'s struct, or take a bare schema."""
    if grammar is None:
        return None
    if isinstance(grammar, dict) and "schema" in grammar:
        return grammar["schema"]
    return grammar


def _decode(raw):
    try:
        return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except ValueError as exc:
        raise BackendError("backend returned something that is not JSON (%s)" % exc)


def _content(response):
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise BackendError("backend returned no choices: %s" % json.dumps(response)[:200])
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = choice.get("text")
    if not isinstance(content, str):
        raise BackendError(_no_content_reason(response, choice, message))
    return content


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


def _read_error(exc):
    try:
        return exc.read().decode("utf-8")[:200]
    except Exception:
        return exc.reason if getattr(exc, "reason", None) else "<no body>"
