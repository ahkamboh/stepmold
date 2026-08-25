"""How one node's output gets generated: think -> emit, or just emit.

This is the fix for the constraint tax (docs/ARCHITECTURE.md §2, Bug 2): forcing a model to
commit to a schema on its first token costs quality, and the known remedy is to separate
the reasoning from the formatting. So a `two_stage: true` node runs twice —

    1. `think`  unconstrained, token-capped, output kept in a scratchpad
    2. `emit`   constrained by the node's grammar, conditioned on that scratchpad

— and the scratchpad is **returned to the caller, never committed to state**. That is not
a detail: state is what downstream prompts see, and letting a model's own loose reasoning
back into its later context is the self-conditioning decay this whole design exists to
avoid. `Attempt` carries the scratchpad; `graph.commit` writes only `Attempt.text`.

Single-stage nodes skip straight to emit, so trivial nodes stay one call cheap.

This module is also where the `Model` protocol's *optional* per-call sampling hint is
spent (`Sampling`). The retry ladder decides what to ask for; `_generate` decides whether
the model in front of it can hear the question. A model whose `generate` does not declare
a `sampling` parameter is called exactly as before, which is what keeps every existing
`Model` — `FakeModel`, the production harness's `FlakyModel`, `OpenAICompatModel` — a
valid `Model` without a line of change. See `verify.run_node` for why a re-sample needs
the hint at all.
"""

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from .errors import BackendError
from .grammar import schema_to_grammar
from .log import DEBUG, event, get_logger
from .render import render

_log = get_logger("codegen")

__all__ = [
    "Attempt",
    "SCRATCHPAD",
    "Sampling",
    "accepts_sampling",
    "generate_once",
    "think",
    "emit",
]

SCRATCHPAD = "scratchpad"
SAMPLING = "sampling"

DEFAULT_THINK_SUFFIX = (
    "\n\nThink this through in a few short sentences. "
    "Do not answer in JSON yet — notes only."
)
SCRATCHPAD_BLOCK = "\n\nYour notes from thinking this through:\n%s"
ERROR_BLOCK = (
    "\n\nYour previous answer was rejected: %s\n"
    "Return corrected output that satisfies the schema."
)


@dataclass(frozen=True)
class Sampling:
    """What the ladder asks a backend for when it wants a *different* draw.

    A hint, not a command: it is the one thing this layer can change that makes a
    re-sample an independent draw rather than the identical request twice. Backends that
    cannot vary their sampling ignore it, and the ladder still has its feedback rung.

    `seed` is here because a server pinned to temperature 0 is deterministic in its
    *stream*, not in its arithmetic: llama.cpp, vLLM and SGLang all accept a per-request
    seed, and changing it is the only knob a greedy server has.
    """

    temperature: float
    seed: Optional[int] = None


@dataclass
class Attempt:
    """One pass at a node's output. `text` is what gets verified and committed."""

    text: str
    scratchpad: Optional[str] = None
    calls: int = 0


def generate_once(node, state, model, error=None, scratchpad=None, sampling=None):
    """Produce one candidate output for `node`.

    `error` is the previous rejection, appended so a re-sample can correct itself
    (T6's ladder). `scratchpad` reuses an earlier think stage instead of paying for
    another one; the caller decides whether reusing it is honest — see
    `verify.run_node`, which throws it away whenever the reasoning behind it is what got
    rejected. `sampling` is the ladder's request for a different draw, and it applies to
    *both* stages: re-thinking at the same temperature as the notes that were just
    discarded would reproduce them.
    """
    calls = 0
    if node.two_stage and scratchpad is None:
        scratchpad = think(node, state, model, sampling=sampling)
        calls += 1
    try:
        text = emit(node, state, model, scratchpad=scratchpad, error=error,
                    sampling=sampling)
    except BackendError as exc:
        # The notes survive the failure they were not part of. A backend that returned
        # nothing has said nothing about the reasoning, and a ladder that re-samples on
        # it should not pay for a second think stage — see `verify.run_node`.
        exc.scratchpad = scratchpad
        raise
    return Attempt(text=text, scratchpad=scratchpad, calls=calls + 1)


def think(node, state, model, sampling=None):
    """The unconstrained first stage. Its output never leaves the scratchpad."""
    template = node.think_prompt or (node.prompt + DEFAULT_THINK_SUFFIX)
    # The default template is the emit prompt, which may itself hold a {scratchpad}
    # placeholder. There are no notes yet, so it renders empty rather than exploding.
    scope = dict(state)
    scope.setdefault(SCRATCHPAD, "")
    prompt = render(template, scope)
    # The size of a prompt, never the prompt. A rendered prompt holds whatever the caller
    # put into state — a whole support ticket, a customer's name — and the point of
    # logging it would be to see it. Bytes tell an operator what they actually came for:
    # whether a prompt is the size they think it is, and where a token bill went.
    if _log.isEnabledFor(DEBUG):
        # Guarded at the call site, not just inside `event`. This fires once per
        # generation and the long-horizon suite makes hundreds of thousands of them, so
        # even building the keyword dict for an event nobody is listening to is a cost
        # worth not paying. The check itself is one bound-method call returning False.
        event(_log, DEBUG, "node.think", node=node.name, prompt_bytes=len(prompt),
              max_tokens=node.think_max_tokens)
    return _generate(model, prompt, None, node.think_max_tokens, sampling)


def emit(node, state, model, scratchpad=None, error=None, sampling=None):
    """The constrained stage: the only output that can ever be committed."""
    prompt = build_prompt(node, state, scratchpad=scratchpad, error=error)
    if _log.isEnabledFor(DEBUG):
        event(_log, DEBUG, "node.emit", node=node.name, prompt_bytes=len(prompt),
              grammar=bool(node.grammar), max_tokens=node.max_tokens,
              scratchpad_bytes=len(scratchpad) if scratchpad is not None else None,
              corrected=bool(error))
    return _generate(
        model,
        prompt,
        schema_to_grammar(node.grammar) if node.grammar else None,
        node.max_tokens,
        sampling,
    )


def _generate(model, prompt, grammar, max_tokens, sampling):
    """Call the model, passing the sampling hint only to a model that declares one.

    The hint is optional in the protocol, so it cannot be sent unconditionally: a model
    written against the three-argument signature would raise `TypeError` on a keyword it
    never asked for, and stepmold would have broken every existing backend to gain a knob most
    of them ignore anyway.
    """
    if sampling is None or not accepts_sampling(model):
        return model.generate(prompt, grammar=grammar, max_tokens=max_tokens)
    return model.generate(
        prompt, grammar=grammar, max_tokens=max_tokens, sampling=sampling
    )


def accepts_sampling(model):
    """True when `model.generate` declares the optional `sampling` keyword."""
    return _accepts_sampling(type(model))


@lru_cache(maxsize=None)
def _accepts_sampling(model_type):
    # Cached on the type, not the instance: the signature belongs to the class, and a
    # ladder asks this question once per generation.
    generate = getattr(model_type, "generate", None)
    if generate is None:
        return False
    try:
        parameters = inspect.signature(generate).parameters.values()
    except (TypeError, ValueError):  # a builtin or C-level callable has no signature
        return False
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == SAMPLING and parameter.kind in kinds:
            return True
    return False


def build_prompt(node, state, scratchpad=None, error=None):
    """Assemble the emit prompt.

    Volatile content goes last. That is a prefix-cache decision, not a style one:
    ARCHITECTURE.md §2 wants every prompt ordered most-stable-first so the KV cache hit rate is
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
