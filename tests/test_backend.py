"""T11 — the OpenAI-compatible adapter, with the HTTP layer mocked.

NOTHING IN THIS FILE MAKES A NETWORK CALL. Every test injects a fake opener; the real
`urllib.request.urlopen` is only ever compared by identity, never invoked.
"""

import email.message
import email.utils
import http.client
import io
import json
import time
import unittest
import urllib.error
import urllib.request

from jig.errors import BackendError
from jig.grammar import schema_to_grammar
from jig.model import Model
from jig.backends.openai_compat import (
    DEFAULT_OPENER,
    GRAMMAR_MODES,
    RETRY_AFTER_CEILING,
    NoCrossOriginRedirect,
    OpenAICompatModel,
)

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}},
    "required": ["category"],
}

# The same schema, closed the way OpenAI-family strict structured output demands: every
# object `additionalProperties: false`, every declared property in `required`. jig's own
# grammar subset accepts both shapes, which is why `strict` cannot be a constant.
CLOSED_SCHEMA = dict(SCHEMA, additionalProperties=False)


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


def _http_error(code, body=b"nope", **headers):
    message = email.message.Message()
    for name, value in headers.items():
        message[name.replace("_", "-")] = value
    return urllib.error.HTTPError("http://x", code, "boom", message, io.BytesIO(body))


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

    def test_extra_body_may_not_collide_with_the_field_the_grammar_mode_owns(self):
        """It is merged last, so a collision would delete the constraint in silence."""
        for mode, key in (("response_format", "response_format"),
                          ("json_schema", "json_schema"),
                          ("json_object", "response_format"),
                          ("response_format", "messages")):
            with self.subTest(mode=mode, key=key):
                with self.assertRaises(ValueError) as caught:
                    model(grammar_mode=mode, extra_body={key: {}})
                self.assertIn(key, str(caught.exception))
                self.assertIn("extra_body", str(caught.exception))

    def test_the_collision_is_refused_before_any_request_is_made(self):
        http = FakeHTTP()
        with self.assertRaises(ValueError):
            model(http, extra_body={"response_format": {}})
        self.assertEqual(http.count, 0)

    def test_a_mode_that_owns_no_constraint_field_still_accepts_it(self):
        """`none` sends nothing of its own, so nothing of its own can be clobbered."""
        http = FakeHTTP()
        model(http, grammar_mode="none",
              extra_body={"response_format": {"type": "json_object"}}).generate("hi")
        self.assertEqual(http.payload["response_format"], {"type": "json_object"})


class TestGrammarModes(unittest.TestCase):
    def test_response_format_is_the_default_and_carries_the_schema(self):
        http = FakeHTTP()
        model(http).generate("hi", grammar=schema_to_grammar(SCHEMA))
        response_format = http.payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["schema"], SCHEMA)

    def test_strict_is_claimed_for_a_schema_that_satisfies_strict_mode(self):
        http = FakeHTTP()
        model(http).generate("hi", grammar=schema_to_grammar(CLOSED_SCHEMA))
        self.assertIs(http.payload["response_format"]["json_schema"]["strict"], True)

    def test_strict_is_not_claimed_for_a_schema_that_does_not(self):
        """SCHEMA has no `additionalProperties: false`, which strict mode requires.

        Claiming it anyway is an HTTP 400 from the server — not a bad sample: 400 is not
        retryable and `BackendError` is not `NodeFailed`, so neither ladder nor `on_fail`
        can absorb it, and the first node kills the run. The schema still goes on the
        wire; only the claim about it is dropped.
        """
        http = FakeHTTP()
        model(http).generate("hi", grammar=schema_to_grammar(SCHEMA))
        envelope = http.payload["response_format"]["json_schema"]
        self.assertIs(envelope["strict"], False)
        self.assertEqual(envelope["schema"], SCHEMA)

    def test_a_property_missing_from_required_is_enough_to_drop_the_claim(self):
        open_schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        http = FakeHTTP()
        model(http).generate("hi", grammar=schema_to_grammar(open_schema))
        self.assertIs(http.payload["response_format"]["json_schema"]["strict"], False)

    def test_a_nested_object_is_checked_too(self):
        nested = {
            "type": "object",
            "properties": {"inner": {"type": "object", "properties": {}, "required": []}},
            "required": ["inner"],
            "additionalProperties": False,
        }
        http = FakeHTTP()
        model(http).generate("hi", grammar=schema_to_grammar(nested))
        self.assertIs(http.payload["response_format"]["json_schema"]["strict"], False)
        nested["properties"]["inner"]["additionalProperties"] = False
        http = FakeHTTP()
        model(http).generate("hi", grammar=schema_to_grammar(nested))
        self.assertIs(http.payload["response_format"]["json_schema"]["strict"], True)

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

    def test_the_non_json_message_names_the_endpoint_and_quotes_the_body(self):
        """A CDN's HTML error page and an empty string are both HTTP 200 here.

        "Expecting value: line 1 column 1 (char 0)" cannot tell them apart, so the URL
        and a clipped prefix of what arrived go into the message.
        """
        http = FakeHTTP(b"<html><title>502 Bad Gateway</title></html>")
        with self.assertRaises(BackendError) as caught:
            model(http).generate("hi")
        message = str(caught.exception)
        self.assertIn("http://localhost:8000/v1/chat/completions", message)
        self.assertIn("502 Bad Gateway", message)

    def test_an_empty_body_says_it_was_empty_rather_than_quoting_nothing(self):
        http = FakeHTTP(b"")
        with self.assertRaises(BackendError) as caught:
            model(http).generate("hi")
        self.assertIn("<empty>", str(caught.exception))

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


class TestSocketLevelFailuresAreWrapped(unittest.TestCase):
    """urllib wraps only what `HTTPConnection.request()` raises.

    Everything from `getresponse()` and `response.read()` — a peer that hangs up, a body
    shorter than its `Content-Length`, a read that outlives the timeout — used to escape
    `_post` untouched: not a `JigError`, so `cli.main` could not report it, and not
    retried, although these are the most transient failures a backend can have.
    `tests/production/test_resilience.py` provokes them over a real socket; here they are
    injected directly, which is the only way to pin the retry count deterministically.
    """

    def test_a_dropped_connection_becomes_a_backend_error(self):
        fake = FakeHTTP(http.client.RemoteDisconnected("peer closed"))
        with self.assertRaises(BackendError) as caught:
            model(fake, max_retries=0).generate("hi")
        self.assertIn("http://localhost:8000/v1/chat/completions", str(caught.exception))
        self.assertIn("RemoteDisconnected", str(caught.exception))

    def test_a_truncated_body_becomes_a_backend_error_naming_the_shortfall(self):
        fake = FakeHTTP(http.client.IncompleteRead(b"12345", 95))
        with self.assertRaises(BackendError) as caught:
            model(fake, max_retries=0).generate("hi")
        message = str(caught.exception)
        self.assertIn("truncated", message)
        self.assertIn("5 bytes", message)
        self.assertIn("100", message)  # 5 read + 95 still expected

    def test_a_read_timeout_names_the_timeout_and_the_setting_to_raise(self):
        fake = FakeHTTP(TimeoutError("timed out"))
        with self.assertRaises(BackendError) as caught:
            model(fake, timeout=2.5, max_retries=0).generate("hi")
        message = str(caught.exception)
        self.assertIn("2.5", message)
        self.assertIn("timeout", message)

    def test_all_three_ride_the_same_ladder_as_a_503(self):
        for failure in (http.client.RemoteDisconnected("peer closed"),
                        http.client.IncompleteRead(b"1", 9),
                        TimeoutError("timed out")):
            with self.subTest(failure=type(failure).__name__):
                fake = FakeHTTP(failure)
                with self.assertRaises(BackendError):
                    model(fake, max_retries=2).generate("hi")
                self.assertEqual(fake.count, 3)

    def test_a_socket_failure_that_clears_is_invisible_to_the_caller(self):
        fake = FakeHTTP(http.client.RemoteDisconnected("peer closed"), _reply("recovered"))
        self.assertEqual(model(fake).generate("hi"), "recovered")
        self.assertEqual(fake.count, 2)


class TestRetryAfterIsHonoured(unittest.TestCase):
    """A `Retry-After` is an instruction, not a suggestion.

    Sleeping jig's own 0.5s when the provider said 1s sends the retry out *sooner* than
    it was asked for, and a provider that says `Retry-After: 60` still gets three
    requests inside 1.5 seconds — which is how an account gets banned rather than rate
    limited. The wait is now `max(own backoff, Retry-After)`, capped.
    """

    def _slept(self, error, **kwargs):
        slept = []
        with self.assertRaises(BackendError):
            model(FakeHTTP(error), sleeper=slept.append, max_retries=2, **kwargs).generate("hi")
        return slept

    def test_delta_seconds_lift_the_backoff_to_what_was_asked(self):
        self.assertEqual(self._slept(_http_error(429, Retry_After="4")), [4.0, 4.0])

    def test_a_shorter_retry_after_never_shortens_our_own_backoff(self):
        """The header is a floor, not a replacement: 0.5s/1.0s already exceeds it."""
        self.assertEqual(self._slept(_http_error(429, Retry_After="0")), [0.5, 1.0])

    def test_an_http_date_is_understood_as_well_as_a_number(self):
        when = email.utils.formatdate(time.time() + 30, usegmt=True)
        slept = self._slept(_http_error(503, Retry_After=when))
        self.assertGreater(slept[0], 20.0)
        self.assertLessEqual(slept[0], 30.0)

    def test_an_absurd_retry_after_is_capped_rather_than_obeyed(self):
        """One node must not disappear into an hour-long sleep inside a run."""
        self.assertEqual(self._slept(_http_error(429, Retry_After="3600")),
                         [RETRY_AFTER_CEILING, RETRY_AFTER_CEILING])

    def test_a_date_already_in_the_past_does_not_produce_a_negative_wait(self):
        when = email.utils.formatdate(time.time() - 600, usegmt=True)
        self.assertEqual(self._slept(_http_error(429, Retry_After=when)), [0.5, 1.0])

    def test_an_unparseable_header_falls_back_to_the_exponential_ladder(self):
        self.assertEqual(self._slept(_http_error(429, Retry_After="soon")), [0.5, 1.0])

    def test_no_header_at_all_is_the_ordinary_case(self):
        self.assertEqual(self._slept(_http_error(500)), [0.5, 1.0])


class TestNothingPrintsTheCredential(unittest.TestCase):
    """A key in a log line is a key in an incident.

    `OpenAICompatModel` was a plain `@dataclass`, so its generated `__repr__` printed
    `api_key` verbatim — and a model is an ordinary object that lands in
    `logging.debug("%r", model)`, in a failing assertion's diff and in a `pdb` frame
    dump. The second half is the amplification path: an upstream error body that echoes
    the caller's own `Authorization` header back, spliced into an exception that is then
    logged.
    """

    KEY = "sk-test-NOTAREALKEY-0123456789"

    def test_the_repr_does_not_contain_the_key(self):
        self.assertNotIn(self.KEY, repr(model(api_key=self.KEY)))

    def test_the_repr_still_says_which_endpoint_and_model_it_is(self):
        printed = repr(model(api_key=self.KEY))
        self.assertIn("http://localhost:8000", printed)
        self.assertIn("qwen3-8b", printed)

    def test_the_repr_fingerprints_the_key_so_the_wrong_one_is_recognisable(self):
        printed = repr(model(api_key=self.KEY))
        self.assertIn(self.KEY[-3:], printed)
        self.assertNotIn(self.KEY[3:-3], printed)
        self.assertNotIn("789", repr(model(api_key=None)))

    def test_an_echoed_bearer_header_is_redacted_out_of_an_error_body(self):
        body = ('{"error": "bad request", "sent": {"Authorization": "Bearer %s"}}'
                % self.KEY).encode()
        with self.assertRaises(BackendError) as caught:
            model(FakeHTTP(_http_error(400, body)), api_key=self.KEY).generate("hi")
        message = str(caught.exception)
        self.assertNotIn(self.KEY, message)
        self.assertIn("<redacted>", message)
        self.assertIn("bad request", message)  # the diagnosis must survive the redaction

    def test_a_bare_key_shaped_string_is_redacted_too(self):
        """Not every echo comes with a `Bearer` in front of it, and the key in the body
        is not necessarily the one this model is holding."""
        body = b'{"error": "key sk-someone-elses-0123456789 is revoked"}'
        with self.assertRaises(BackendError) as caught:
            model(FakeHTTP(_http_error(400, body)), api_key=None).generate("hi")
        self.assertNotIn("sk-someone-elses-0123456789", str(caught.exception))
        self.assertIn("revoked", str(caught.exception))


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
