"""T11 — the OpenAI-compatible adapter, with the HTTP layer mocked.

NOTHING IN THIS FILE MAKES A NETWORK CALL. Every test injects a fake opener; the real
`urllib.request.urlopen` is only ever compared by identity, never invoked.
"""

import email.message
import io
import json
import unittest
import urllib.error
import urllib.request

from jig.errors import BackendError
from jig.grammar import schema_to_grammar
from jig.model import Model
from jig.backends.openai_compat import (
    DEFAULT_OPENER,
    GRAMMAR_MODES,
    NoCrossOriginRedirect,
    OpenAICompatModel,
)

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}},
    "required": ["category"],
}


class FakeResponse(io.BytesIO):
    """Just enough of an http response for `with opener(...) as response:`."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class FakeHTTP:
    """Records every request and replays a scripted list of outcomes."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes) or [_reply("ok")]
        self.requests = []
        self.timeouts = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)

    @property
    def payload(self):
        return json.loads(self.requests[-1].data.decode("utf-8"))

    @property
    def count(self):
        return len(self.requests)


def _reply(content):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()


def _http_error(code, body=b"nope"):
    return urllib.error.HTTPError("http://x", code, "boom", {}, io.BytesIO(body))


def model(http=None, **kwargs):
    options = {
        "base_url": "http://localhost:8000",
        "model": "qwen3-8b",
        "opener": http or FakeHTTP(),
        "sleeper": lambda seconds: None,
    }
    options.update(kwargs)
    return OpenAICompatModel(**options)


class TestNoNetwork(unittest.TestCase):
    def test_constructing_a_model_performs_no_io(self):
        http = FakeHTTP()
        model(http)
        self.assertEqual(http.count, 0)

    def test_the_default_opener_is_the_guarded_one_and_is_never_called_here(self):
        built = OpenAICompatModel(base_url="http://localhost:8000", model="m")
        # Bound methods compare equal but are not identical objects, so assertEqual.
        self.assertEqual(built.opener, DEFAULT_OPENER.open)


class TestProtocol(unittest.TestCase):
    def test_it_satisfies_the_model_protocol(self):
        self.assertIsInstance(model(), Model)

    def test_generate_returns_the_message_content(self):
        http = FakeHTTP(_reply('{"category": "billing"}'))
        self.assertEqual(model(http).generate("hi"), '{"category": "billing"}')

    def test_a_completion_style_choice_is_also_accepted(self):
        http = FakeHTTP(json.dumps({"choices": [{"text": "legacy"}]}).encode())
        self.assertEqual(model(http).generate("hi"), "legacy")


class TestUrlBuilding(unittest.TestCase):
    def test_a_bare_host_gets_the_full_path(self):
        self.assertEqual(
            model(base_url="http://localhost:8000").url,
            "http://localhost:8000/v1/chat/completions",
        )

    def test_a_trailing_slash_is_tolerated(self):
        self.assertEqual(
            model(base_url="http://localhost:8000/").url,
            "http://localhost:8000/v1/chat/completions",
        )

    def test_a_v1_base_is_completed(self):
        self.assertEqual(
            model(base_url="http://host/v1").url, "http://host/v1/chat/completions"
        )

    def test_a_full_endpoint_is_left_alone(self):
        self.assertEqual(
            model(base_url="http://host/v1/chat/completions").url,
            "http://host/v1/chat/completions",
        )

    def test_an_empty_base_url_is_rejected(self):
        with self.assertRaises(ValueError):
            model(base_url="")


class TestRequestShape(unittest.TestCase):
    def test_the_payload_carries_model_prompt_and_limits(self):
        http = FakeHTTP()
        model(http, temperature=0.2).generate("classify this", max_tokens=64)
        payload = http.payload
        self.assertEqual(payload["model"], "qwen3-8b")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "classify this"}])
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["temperature"], 0.2)

    def test_it_posts_json(self):
        http = FakeHTTP()
        model(http).generate("hi")
        request = http.requests[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Content-type"], "application/json")

    def test_an_api_key_becomes_a_bearer_header(self):
        http = FakeHTTP()
        model(http, api_key="secret").generate("hi")
        self.assertEqual(http.requests[0].headers["Authorization"], "Bearer secret")

    def test_no_api_key_means_no_auth_header(self):
        http = FakeHTTP()
        model(http, api_key="").generate("hi")
        self.assertNotIn("Authorization", http.requests[0].headers)

    def test_the_timeout_is_passed_to_the_opener(self):
        http = FakeHTTP()
        model(http, timeout=12.5).generate("hi")
        self.assertEqual(http.timeouts[0], 12.5)

    def test_extra_body_is_merged(self):
        http = FakeHTTP()
        model(http, extra_body={"top_p": 0.9}).generate("hi")
        self.assertEqual(http.payload["top_p"], 0.9)


class TestGrammarModes(unittest.TestCase):
    def test_response_format_is_the_default_and_carries_the_schema(self):
        http = FakeHTTP()
        model(http).generate("hi", grammar=schema_to_grammar(SCHEMA))
        response_format = http.payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["schema"], SCHEMA)
        self.assertTrue(response_format["json_schema"]["strict"])

    def test_json_schema_mode_puts_the_schema_at_the_top_level(self):
        http = FakeHTTP()
        model(http, grammar_mode="json_schema").generate("hi", grammar=schema_to_grammar(SCHEMA))
        self.assertEqual(http.payload["json_schema"], SCHEMA)
        self.assertNotIn("response_format", http.payload)

    def test_json_object_mode_falls_back_to_describing_the_schema(self):
        http = FakeHTTP()
        model(http, grammar_mode="json_object").generate("hi", grammar=schema_to_grammar(SCHEMA))
        self.assertEqual(http.payload["response_format"], {"type": "json_object"})
        self.assertIn("category", http.payload["messages"][0]["content"])

    def test_none_mode_sends_no_constraint(self):
        http = FakeHTTP()
        model(http, grammar_mode="none").generate("hi", grammar=schema_to_grammar(SCHEMA))
        self.assertNotIn("response_format", http.payload)
        self.assertNotIn("json_schema", http.payload)

    def test_a_bare_schema_is_accepted_as_well_as_the_wrapped_struct(self):
        http = FakeHTTP()
        model(http).generate("hi", grammar=SCHEMA)
        self.assertEqual(http.payload["response_format"]["json_schema"]["schema"], SCHEMA)

    def test_no_grammar_means_no_constraint_field(self):
        http = FakeHTTP()
        model(http).generate("hi")
        self.assertNotIn("response_format", http.payload)

    def test_an_unknown_mode_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            model(grammar_mode="telepathy")
        self.assertIn("telepathy", str(caught.exception))
        for name in GRAMMAR_MODES:
            self.assertIn(name, str(caught.exception))


class TestFailures(unittest.TestCase):
    def test_a_client_error_is_raised_immediately_with_the_body(self):
        http = FakeHTTP(_http_error(400, b"bad schema"))
        with self.assertRaises(BackendError) as caught:
            model(http).generate("hi")
        self.assertIn("400", str(caught.exception))
        self.assertIn("bad schema", str(caught.exception))
        self.assertEqual(http.count, 1)

    def test_a_server_error_is_retried_then_raised(self):
        http = FakeHTTP(_http_error(503))
        with self.assertRaises(BackendError):
            model(http, max_retries=2).generate("hi")
        self.assertEqual(http.count, 3)

    def test_a_retry_can_succeed(self):
        http = FakeHTTP(_http_error(503), _reply("recovered"))
        self.assertEqual(model(http).generate("hi"), "recovered")
        self.assertEqual(http.count, 2)

    def test_an_unreachable_host_is_reported(self):
        http = FakeHTTP(urllib.error.URLError("connection refused"))
        with self.assertRaises(BackendError) as caught:
            model(http, max_retries=0).generate("hi")
        self.assertIn("connection refused", str(caught.exception))

    def test_a_non_json_body_is_reported(self):
        http = FakeHTTP(b"<html>gateway</html>")
        with self.assertRaises(BackendError) as caught:
            model(http).generate("hi")
        self.assertIn("not JSON", str(caught.exception))

    def test_no_choices_is_reported(self):
        http = FakeHTTP(json.dumps({"choices": []}).encode())
        with self.assertRaises(BackendError):
            model(http).generate("hi")

    def test_a_choice_without_text_is_reported(self):
        http = FakeHTTP(json.dumps({"choices": [{"message": {}}]}).encode())
        with self.assertRaises(BackendError):
            model(http).generate("hi")

    def test_a_backend_error_is_a_jig_error_so_the_cli_reports_it(self):
        from jig.errors import JigError

        self.assertTrue(issubclass(BackendError, JigError))


class TestApiKeyFromEnvironment(unittest.TestCase):
    def setUp(self):
        import os

        self.saved = {name: os.environ.pop(name, None)
                      for name in ("JIG_API_KEY", "OPENAI_API_KEY")}

    def tearDown(self):
        import os

        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_jig_api_key_is_picked_up(self):
        import os

        os.environ["JIG_API_KEY"] = "from-env"
        self.assertEqual(model().api_key, "from-env")

    def test_openai_api_key_is_the_fallback(self):
        import os

        os.environ["OPENAI_API_KEY"] = "fallback"
        self.assertEqual(model().api_key, "fallback")

    def test_an_explicit_key_wins(self):
        import os

        os.environ["JIG_API_KEY"] = "from-env"
        self.assertEqual(model(api_key="explicit").api_key, "explicit")

    def test_no_key_anywhere_is_fine(self):
        self.assertIsNone(model().api_key)


class TestRedirectsCannotLeakTheKey(unittest.TestCase):
    """A redirect must never hand the Authorization header to another origin.

    urllib's stock `HTTPRedirectHandler` copies request headers onto the next hop with
    no same-origin check, so a `base_url` that answers 302 could walk the operator's API
    key — and the rendered prompt — to any host it names. Still no network here: the
    handler is driven directly, the way urllib would drive it.
    """

    ENDPOINT = "http://localhost:8000/v1/chat/completions"

    def redirect(self, handler, to_url, from_url=None):
        request = urllib.request.Request(
            from_url or self.ENDPOINT,
            data=b"{}",
            headers={"Authorization": "Bearer secret",
                     "Content-Type": "application/json"},
            method="POST",
        )
        return handler.redirect_request(
            request, io.BytesIO(b""), 302, "Found", email.message.Message(), to_url
        )

    def test_the_stdlib_handler_really_does_carry_the_key(self):
        """Why the guard exists — this is the leak, reproduced."""
        carried = self.redirect(
            urllib.request.HTTPRedirectHandler(), "http://elsewhere.example/collect"
        )
        self.assertEqual(carried.headers["Authorization"], "Bearer secret")

    def test_a_cross_origin_redirect_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            self.redirect(NoCrossOriginRedirect(), "http://elsewhere.example/collect")
        self.assertIn("elsewhere.example", str(caught.exception))

    def test_a_scheme_downgrade_is_refused(self):
        with self.assertRaises(BackendError):
            self.redirect(
                NoCrossOriginRedirect(),
                "http://api.example/v1/chat/completions",
                from_url="https://api.example/v1/chat/completions",
            )

    def test_another_port_on_the_same_host_is_refused(self):
        with self.assertRaises(BackendError):
            self.redirect(NoCrossOriginRedirect(), "http://localhost:9999/collect")

    def test_a_same_origin_redirect_is_still_followed(self):
        moved = self.redirect(
            NoCrossOriginRedirect(), "http://localhost:8000/v2/chat/completions"
        )
        self.assertEqual(moved.full_url, "http://localhost:8000/v2/chat/completions")

    def test_the_default_port_is_not_treated_as_a_different_origin(self):
        moved = self.redirect(
            NoCrossOriginRedirect(),
            "http://api.example:80/v1/chat/completions",
            from_url="http://api.example/v1/chat/completions",
        )
        self.assertEqual(moved.full_url, "http://api.example:80/v1/chat/completions")

    def test_the_default_opener_installs_the_guard_instead_of_the_stock_handler(self):
        handlers = DEFAULT_OPENER.handlers
        self.assertTrue(
            any(isinstance(handler, NoCrossOriginRedirect) for handler in handlers)
        )
        self.assertFalse(
            any(type(handler) is urllib.request.HTTPRedirectHandler
                for handler in handlers)
        )

    def test_a_refused_redirect_is_not_retried(self):
        """It is a refusal, not a flaky server: one attempt, then the error."""

        class Redirecting:
            def __init__(self):
                self.count = 0

            def __call__(self, request, timeout=None):
                self.count += 1
                raise BackendError("refusing to follow a redirect")

        http = Redirecting()
        with self.assertRaises(BackendError):
            model(http, max_retries=2).generate("hi")
        self.assertEqual(http.count, 1)


class TheRequestCarriesAnExplicitUserAgent(unittest.TestCase):
    """urllib's default UA gets 403'd by Cloudflare-fronted providers.

    Cerebras returns HTTP 403 error 1010 (browser-signature ban) for
    "Python-urllib/X.Y" before the request reaches the model. Found the first time this
    adapter spoke to a real server — every mocked test passed without it.
    """

    def test_a_user_agent_header_is_always_sent(self):
        from jig.backends.openai_compat import OpenAICompatModel

        seen = {}

        def opener(request, timeout=None):
            seen["headers"] = dict(request.headers)
            raise AssertionError("stop before the network")

        model = OpenAICompatModel(base_url="http://x", model="m", opener=opener)
        try:
            model.generate("hi", None, 10)
        except Exception:
            pass
        keys = {k.lower() for k in seen.get("headers", {})}
        self.assertIn("user-agent", keys)

    def test_the_user_agent_is_not_the_urllib_default(self):
        from jig.backends.openai_compat import OpenAICompatModel

        model = OpenAICompatModel(base_url="http://x", model="m")
        self.assertNotIn("python-urllib", model.user_agent.lower())
        self.assertTrue(model.user_agent.strip())


if __name__ == "__main__":
    unittest.main()
