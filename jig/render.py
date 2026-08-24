"""Prompt templating: `{var}` and `{a.b}`, substituted from run state.

Deliberately *not* `str.format`. Prompts routinely contain literal JSON (`{"a": 1}`) —
which `str.format` treats as a field — and `str.format` also resolves attributes and
indexes, which is more machinery than a prompt template should have. This does one thing:
look a name up in state, render it as text.

`{{` and `}}` are literal braces, so a prompt can show the model an example object.

Substitution is a single `re.sub` pass and the result is never re-scanned. That is the
load-bearing property, not an implementation detail: run state holds every value the
workflow has seen, so a second pass would let a ticket reading `{card_number}` print
another state key into the prompt — input reading state it was never shown. Tests in
tests/production/test_adversarial.py pin it.
"""

import json
import re

from .errors import MissingVariable

__all__ = ["render"]

_TOKEN = re.compile(
    r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}"
)


def render(template, state):
    """Return `template` with every `{name}` replaced by its value from `state`."""

    def substitute(match):
        if match.group(0) == "{{":
            return "{"
        if match.group(0) == "}}":
            return "}"
        return as_text(_lookup(match.group(1), state))

    return _TOKEN.sub(substitute, template)


def as_text(value):
    """Render a state value for inclusion in a prompt."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=False)


def _lookup(path, state):
    current = state
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise MissingVariable(
                "prompt needs {%s} but state has %s" % (_clip(path), _names(state))
            )
        current = current[part]
    return current


def _names(state):
    """The state keys, repr'd — because the confusing case is an invisible difference.

    A key holding a zero-width space renders identically to the real one, so a plain
    list says "prompt needs {ticket} but state has ticket" and the operator concludes
    jig is broken. repr writes such a character as an escape, so the difference shows.
    Keys are caller-supplied, so the list is clipped too: a run input can be as long as
    whoever pasted it.
    """
    if not state:
        return "nothing"
    return _clip(", ".join(repr(name) for name in sorted(state)), 200)


def _clip(text, limit=120):
    """Keep a message bounded. A megabyte of input must not become a megabyte of error."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."
