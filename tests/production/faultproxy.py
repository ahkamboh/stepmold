"""A fault-injecting proxy for an OpenAI-compatible endpoint.

Production is not the happy path. Rate limits, 5xx, dead sockets, truncated bodies and
models that ignore their grammar are the *normal* condition, and until today every one of
jig's failure paths had only ever met a mock that behaved exactly as the mock's author
expected.

This sits between jig and a real upstream (or answers by itself) and misbehaves on
purpose. Start it, point `--model openai:http://127.0.0.1:PORT/v1#...` at it, and watch
what the runtime actually does.

stdlib only, like everything else in jig.

    proxy = FaultProxy(upstream="https://api.cerebras.ai/v1", api_key=KEY)
    proxy.start()
    proxy.script(["429", "500", "ok"])       # first call 429s, second 500s, third works
    ...
    proxy.stop()

Faults, by name:

    ok            forward to upstream (or answer with a schema-shaped stub)
    429           rate limited, with Retry-After
    500           upstream error
    503           service unavailable
    truncated     valid prefix of a JSON body, then the socket closes
    notjson       200 with an HTML error page, which is what CDNs actually return
    empty         200, well-formed, but message.content is null
    reasoning     200 with reasoning filled and content null (a reasoning model that
                  spent its budget thinking — the real defect we hit on first contact)
    badschema     200, valid JSON, but the object violates the requested schema
    prose         200 whose content is chatty prose wrapped around the JSON
    slow          sleeps past a client timeout, then answers
    reset         closes the connection without writing anything
    garbage       200 whose content is not JSON at all
"""

import http.server
import json
import socket
import threading
import time
import urllib.error
import urllib.request


FAULTS = (
    "ok", "429", "500", "503", "truncated", "notjson", "empty", "reasoning",
    "badschema", "prose", "slow", "reset", "garbage",
)


class FaultProxy:
    """An OpenAI-compatible endpoint that can be told to misbehave."""

    def __init__(self, upstream=None, api_key=None, model="gpt-oss-120b",
                 slow_seconds=5.0, host="127.0.0.1"):
        self.upstream = upstream.rstrip("/") if upstream else None
        self.api_key = api_key
        self.model = model
        self.slow_seconds = slow_seconds
        self.host = host
        self._script = []
        self._default = "ok"
        self._lock = threading.Lock()
        self.calls = []           # one record per request the proxy received
        self._server = None
        self._thread = None

    # ---------------------------------------------------------------- lifecycle

    def start(self):
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass  # keep the test output readable

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except ValueError:
                    payload = {}
                fault = proxy._next_fault()
                proxy._record(fault, payload, dict(self.headers))
                proxy._respond(self, fault, payload)

        self._server = http.server.ThreadingHTTPServer((self.host, 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    @property
    def port(self):
        return self._server.server_address[1]

    @property
    def base_url(self):
        return "http://%s:%d/v1" % (self.host, self.port)

    # ------------------------------------------------------------------ control

    def script(self, faults, default="ok"):
        """Queue a fault per call. When the queue drains, `default` takes over."""
        for name in list(faults) + [default]:
            if name not in FAULTS:
                raise ValueError("unknown fault %r (known: %s)" % (name, ", ".join(FAULTS)))
        with self._lock:
            self._script = list(faults)
            self._default = default
            self.calls = []
        return self

    def _next_fault(self):
        with self._lock:
            if self._script:
                return self._script.pop(0)
            return self._default

    def _record(self, fault, payload, headers):
        with self._lock:
            self.calls.append({
                "fault": fault,
                "model": payload.get("model"),
                "max_tokens": payload.get("max_tokens"),
                "user_agent": headers.get("User-Agent"),
                "had_schema": "response_format" in payload or "json_schema" in payload,
                "prompt": (payload.get("messages") or [{}])[0].get("content", ""),
            })

    # ----------------------------------------------------------------- responses

    def _respond(self, handler, fault, payload):
        if fault == "reset":
            try:
                handler.connection.close()
            except Exception:
                pass
            return
        if fault == "slow":
            time.sleep(self.slow_seconds)
            fault = "ok"
        if fault == "429":
            return self._send(handler, 429, {"error": {"message": "rate limited"}},
                              extra_headers={"Retry-After": "1"})
        if fault in ("500", "503"):
            return self._send(handler, int(fault), {"error": {"message": "upstream boom"}})
        if fault == "notjson":
            body = b"<html><head><title>502 Bad Gateway</title></head></html>"
            handler.send_response(200)
            handler.send_header("Content-Type", "text/html")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return
        if fault == "truncated":
            body = b'{"choices": [{"message": {"content": "{\\"cat'
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body) + 200))  # lie, then stop
            handler.end_headers()
            handler.wfile.write(body)
            try:
                handler.connection.close()
            except Exception:
                pass
            return

        content, reasoning = self._content_for(fault, payload)
        return self._send(handler, 200, {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content,
                            "reasoning": reasoning},
                "finish_reason": "length" if fault in ("empty", "reasoning") else "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10,
                      "completion_tokens_details": {"reasoning_tokens":
                                                    10 if fault == "reasoning" else 0}},
        })

    def _content_for(self, fault, payload):
        if fault == "empty":
            return None, None
        if fault == "reasoning":
            return None, "I should think about this at length and then run out of room"
        if fault == "garbage":
            return "I'm sorry, I can't help with that.", None
        schema = _schema_of(payload)
        if fault == "badschema":
            return json.dumps(_violating(schema)), None
        if fault == "prose":
            return ("Sure! Here is the JSON you asked for:\n\n```json\n"
                    + json.dumps(_satisfying(schema)) + "\n```\nHope that helps!"), None
        # ok
        if self.upstream:
            forwarded = self._forward(payload)
            if forwarded is not None:
                return forwarded
        return json.dumps(_satisfying(schema)), None

    def _forward(self, payload):
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": "jig-faultproxy/1"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        request = urllib.request.Request(
            self.upstream + "/chat/completions", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        message = (data.get("choices") or [{}])[0].get("message") or {}
        return message.get("content"), message.get("reasoning")

    def _send(self, handler, status, obj, extra_headers=None):
        body = json.dumps(obj).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(body)


# ------------------------------------------------------------------- schema helpers

def _schema_of(payload):
    fmt = payload.get("response_format") or {}
    if isinstance(fmt, dict):
        inner = fmt.get("json_schema")
        if isinstance(inner, dict) and isinstance(inner.get("schema"), dict):
            return inner["schema"]
    if isinstance(payload.get("json_schema"), dict):
        return payload["json_schema"]
    return {"type": "object", "properties": {}, "required": []}


def _satisfying(schema):
    """The smallest object that satisfies `schema` — the proxy's stand-in for a model."""
    out = {}
    for name, spec in (schema.get("properties") or {}).items():
        if name not in (schema.get("required") or []):
            continue
        out[name] = _value_for(spec)
    return out


def _value_for(spec):
    if "enum" in spec and spec["enum"]:
        return spec["enum"][0]
    declared = spec.get("type")
    if isinstance(declared, list):
        declared = declared[0]
    return {"string": "x", "integer": 1, "number": 1.0,
            "boolean": False, "array": [], "object": {}, "null": None}.get(declared, "x")


def _violating(schema):
    """An object of the right shape whose values break the schema on purpose."""
    out = {}
    for name, spec in (schema.get("properties") or {}).items():
        if name not in (schema.get("required") or []):
            continue
        if "enum" in spec:
            out[name] = "definitely-not-in-the-enum"
        elif spec.get("type") == "string":
            out[name] = 12345
        else:
            out[name] = "wrong type on purpose"
    return out
