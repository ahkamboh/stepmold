"""How one node's output gets generated: think -> emit, or just emit.

This is the fix for the constraint tax (docs/PLAN.md §2, Bug 2): forcing a model to
commit to a schema on its first token costs quality, and the known remedy is to separate
the reasoning from the formatting. So a `two_stage: true` node runs twice —

    1. `think`  unconstrained, token-capped, output kept in a scratchpad
    2. `emit`   constrained by the node's grammar, conditioned on that scratchpad

— and the scratchpad is **returned to the caller, never committed to state**. That is not
a detail: state is what downstream prompts see, and letting a model's own loose reasoning
back into its later context is the self-conditioning decay this whole design exists to
avoid. `Attempt` carries the scratchpad; `graph.commit` writes only `Attempt.text`.

Single-stage nodes skip straight to emit, so trivial nodes stay one call cheap.
"""

from dataclasses import dataclass
from typing import Optional

from .grammar import schema_to_grammar
from .render import render

__all__ = ["Attempt", "SCRATCHPAD", "generate_once", "think", "emit"]

SCRATCHPAD = "scratchpad"

DEFAULT_THINK_SUFFIX = (
    "\n\nThink this through in a few short sentences. "
    "Do not answer in JSON yet — notes only."
)
SCRATCHPAD_BLOCK = "\n\nYour notes from thinking this through:\n%s"
ERROR_BLOCK = (
    "\n\nYour previous answer was rejected: %s\n"
    "Return corrected output that satisfies the schema."
)


@dataclass
class Attempt:
    """One pass at a node's output. `text` is what gets verified and committed."""

    text: str
    scratchpad: Optional[str] = None
    calls: int = 0


def generate_once(node, state, model, error=None, scratchpad=None):
    """Produce one candidate output for `node`.

    `error` is the previous rejection, appended so a re-sample can correct itself
    (T6's ladder). `scratchpad` reuses an earlier think stage instead of paying for
    another one — a re-sample re-rolls the *emit*, which is the cheap half.
    """
    calls = 0
    if node.two_stage and scratchpad is None:
        scratchpad = think(node, state, model)
        calls += 1
    text = emit(node, state, model, scratchpad=scratchpad, error=error)
    return Attempt(text=text, scratchpad=scratchpad, calls=calls + 1)


def think(node, state, model):
    """The unconstrained first stage. Its output never leaves the scratchpad."""
    template = node.think_prompt or (node.prompt + DEFAULT_THINK_SUFFIX)
    # The default template is the emit prompt, which may itself hold a {scratchpad}
    # placeholder. There are no notes yet, so it renders empty rather than exploding.
    scope = dict(state)
    scope.setdefault(SCRATCHPAD, "")
    return model.generate(
        render(template, scope), grammar=None, max_tokens=node.think_max_tokens
    )


def emit(node, state, model, scratchpad=None, error=None):
    """The constrained stage: the only output that can ever be committed."""
    return model.generate(
        build_prompt(node, state, scratchpad=scratchpad, error=error),
        grammar=schema_to_grammar(node.grammar) if node.grammar else None,
        max_tokens=node.max_tokens,
    )


def build_prompt(node, state, scratchpad=None, error=None):
    """Assemble the emit prompt.

    Volatile content goes last. That is a prefix-cache decision, not a style one:
    PLAN.md §2 wants every prompt ordered most-stable-first so the KV cache hit rate is
    engineered rather than accidental. The node template is stable, the scratchpad
    changes every call, and a correction changes every retry.
    """
    placeholder = "{%s}" % SCRATCHPAD
    if scratchpad is not None and placeholder in node.prompt:
        prompt = render(node.prompt, dict(state, **{SCRATCHPAD: scratchpad}))
    else:
        prompt = render(node.prompt, state)
        if scratchpad is not None:
            prompt += SCRATCHPAD_BLOCK % scratchpad
    if error:
        prompt += ERROR_BLOCK % error
    return prompt
