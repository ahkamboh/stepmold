"""Prompt templating: `{var}` and `{a.b}`, substituted from run state.

Deliberately *not* `str.format`. Prompts routinely contain literal JSON (`{"a": 1}`) —
which `str.format` treats as a field — and `str.format` also resolves attributes and
indexes, which is more machinery than a prompt template should have. This does one thing:
look a name up in state, render it as text.

`{{` and `}}` are literal braces, so a prompt can show the model an example object.
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
    for index, part in enumerate(path.split(".")):
        if not isinstance(current, dict) or part not in current:
            available = ", ".join(sorted(state)) if state else "nothing"
            raise MissingVariable(
                "prompt needs {%s} but state has %s" % (path, available)
            )
        current = current[part]
    return current
