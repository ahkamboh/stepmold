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
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..errors import BackendError

__all__ = ["GRAMMAR_MODES", "OpenAICompatModel"]

GRAMMAR_MODES = ("response_format", "json_schema", "json_object", "none")
API_KEY_VARIABLES = ("JIG_API_KEY", "OPENAI_API_KEY")
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)


@dataclass
class OpenAICompatModel:
    """A `Model` backed by an OpenAI-compatible chat completions endpoint.

    `opener` and `sleeper` exist so the HTTP layer can be replaced in a test. Nothing
    else in jig should ever need them.
    """

    base_url: str
    model: str
    api_key: Optional[str] = None
    grammar_mode: str = "response_format"
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 2
    schema_name: str = "jig_node"
    extra_body: Dict[str, Any] = field(default_factory=dict)
    opener: Callable = urllib.request.urlopen
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
            "max_tokens": max_tokens,
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
        headers = {"Content-Type": "application/json"}
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
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        content = choices[0].get("text")
    if not isinstance(content, str):
        raise BackendError("backend returned a choice with no text content")
    return content


def _read_error(exc):
    try:
        return exc.read().decode("utf-8")[:200]
    except Exception:
        return exc.reason if getattr(exc, "reason", None) else "<no body>"
