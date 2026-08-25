"""Stages 2 and 3 — decompose the task, then write one prompt per node.

These are the two stages that need a model, and they use the model the same way jig's
runtime does: a schema goes out with the request, the answer is validated against that
schema by hand, and a rejected answer is re-asked with *what was wrong* and never with
what it said (jig/verify.py, rule 2). A compiler that asked for unconstrained JSON and
hoped would be a poor advertisement for the thing it compiles.

What the planner is *allowed to say* matters as much as what is checked afterwards. It
proposes an ordered list of nodes and a list of endings; it never writes an edge. The
edges are derived here — the nodes in the order given, then the endings off the last
node with the conditional ones first, because `graph.yaml` takes the first matching edge.
So the classic planner failure of pointing an edge at a node that does not exist is not
rejected, it is unrepresentable. The failures that remain representable — a field written
twice, a field written by nobody, a node reading a field that is written after it — are
checked and re-asked, never quietly repaired.

Two checks are worth more than the rest, and both are arithmetic over the gold cases
rather than an opinion:

* **exactly once.** Every field the pack must produce is written by exactly one node.
  `assemble` enforces it again at the end; enforcing it here is what lets the planner
  fix its own answer while a model is still in the room.
* **the branch is replayed.** When the gold cases carry an `end`, the proposed endings
  are run against every case's expected output. A gate that routes a gold case to the
  wrong ending is rejected with the case that broke it. And when the cases show only one
  ending, exactly one ending is accepted — a branch nobody asked for is a rejection.
"""

import json
import re

from ..errors import MissingVariable
from ..grammar import ValidationError, schema_to_grammar, validate_against
from ..render import render
from ..verify import Rejected, extract_json
from .spec import BuildError, GraphPlan, NodePlan

__all__ = ["induce", "write_prompts", "ATTEMPTS", "MAX_WRITES_PER_NODE"]

# One attempt plus two re-asks, matching the runtime's default ladder (verify.run_node).
ATTEMPTS = 3

# The measured result this project rests on is that reliability comes from bounded steps,
# not a bigger model, so the cap is a hard rejection rather than advice in the prompt. It
# is the widest node in the example packs: `extract` and `emit` in support_triage each
# write three fields, and nothing writes four.
MAX_WRITES_PER_NODE = 3

# A prompt for a 7B node is a short one. The floor catches "Classify it." and the ceiling
# catches an essay; both are the shapes a model reaches for when it has not been told the
# job is small. The example packs run 250-600 characters.
MIN_PROMPT_CHARS = 60
MAX_PROMPT_CHARS = 1500

# An enum this small must be spelled out in the prompt: the grammar makes a wrong value
# unrepresentable, but only the prompt can say what the right one means. Above it, the
# prompt would be a wall of tokens, so a few worked examples are enough.
MAX_ENUM_TO_SPELL_OUT = 6
MIN_ENUM_TO_QUOTE = 3

# Reasoning belongs in a two_stage think pass, where it is thrown away before anything is
# committed — not in an emit prompt, where it costs tokens inside the constrained call and
# fights the grammar for the model's attention.
_COT_PHRASES = (
    "step by step",
    "step-by-step",
    "think through",
    "think about",
    "chain of thought",
    "reason about",
    "let's think",
    "explain your reasoning",
    "before answering",
)

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

# `scratchpad` is the one state name a pack may not use for its own data: it is where a
# two_stage node's notes are handed to the emit call (docs/pack-format.md).
_SCRATCHPAD = "scratchpad"

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "writes": {"type": "array", "items": {"type": "string"}},
                    "purpose": {"type": "string"},
                    "two_stage": {"type": "boolean"},
                    "reads": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "writes", "purpose", "two_stage", "reads"],
                "additionalProperties": False,
            },
        },
        # An ending's condition is a field name and one value, not a mapping: jig's
        # grammar subset pins named properties, and an open mapping of state-path to
        # value would be `{"type": "object"}` — a constraint you think you have and
        # don't. The pair is turned into `when: {field: value}` here.
        "endings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "when_field": {"type": "string"},
                    "when_equals": {"type": "string"},
                },
                "required": ["name", "when_field", "when_equals"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes", "endings"],
    "additionalProperties": False,
}

_PROMPT_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
    "additionalProperties": False,
}


class _Reject(ValueError):
    """One answer failed a check. Retryable — unlike `BuildError`, which is fatal."""


# ---------------------------------------------------------------------------
# induce
# ---------------------------------------------------------------------------


def induce(task, model):
    """Propose a decomposition of `task` as a `GraphPlan`: who writes what, in what order."""
    if not task.fields:
        # Nothing downstream can recover from this and no re-ask will invent a field, so
        # it fails here rather than after three model calls.
        raise BuildError(
            "%s: the examples declare no output fields, so there is nothing to "
            "decompose" % task.name
        )

    answer = _ask(
        model,
        _plan_prompt(task),
        _PLAN_SCHEMA,
        1024,
        lambda obj: _check_plan(task, obj),
        "planner",
    )
    return _build_plan(task, answer)


def _plan_prompt(task):
    lines = [
        "You are decomposing one workflow into a graph of small steps. Each step is a "
        "single call to a 7B model, constrained by a JSON grammar.",
        "",
        "Workflow: %s" % task.name,
    ]
    if task.description.strip():
        lines.append(task.description.strip())
    lines.append("")
    lines.append("Every run is given: %s" % (", ".join(task.inputs) or "nothing"))
    lines.append("")
    lines.append(
        "The fields the workflow must produce, as observed in %d gold examples:"
        % len(task.cases)
    )
    for spec in task.fields:
        lines.append("  " + _field_line(spec))

    sample = _sample_cases(task)
    if sample:
        lines.append("")
        lines.append("Two of the gold examples:")
        lines.extend("  " + line for line in sample)

    lines.append("")
    lines.extend(_endings_evidence(task))
    lines.append("")
    lines.append("Rules:")
    lines.append("  - Every field is written by exactly one node: none twice, none left out.")
    lines.append(
        "  - At most %d fields per node. Reliability comes from short steps, not from a "
        "bigger model, so prefer more, smaller nodes." % MAX_WRITES_PER_NODE
    )
    lines.append(
        "  - The nodes run in the order you list them. A node that needs a field an "
        "earlier node wrote must list it in reads. Run inputs may be read by any node."
    )
    lines.append(
        "  - two_stage costs a second model call and buys an unconstrained think pass. "
        "Set it only where the node has a real judgement to make, and say why in purpose."
    )
    lines.append("  - purpose is one imperative line: what this node decides.")
    lines.append("  - Every name is a lowercase identifier.")
    lines.append("")
    lines.append("Answer with a JSON object of this shape:")
    lines.append(
        '  {"nodes": [{"name": "classify", "writes": ["category"], '
        '"purpose": "Pick the ticket category.", "two_stage": false, "reads": []}], '
        '"endings": [{"name": "done", "when_field": "", "when_equals": ""}]}'
    )
    lines.append(
        "The last ending is the fallthrough: its when_field and when_equals are both "
        "empty strings. Any earlier ending is taken when that field holds that value."
    )
    return "\n".join(lines)


def _field_line(spec):
    parts = [spec.type]
    if spec.enum:
        parts.append("one of: %s" % ", ".join(_show(value) for value in spec.enum))
    if spec.optional:
        parts.append("sometimes null")
    line = "%s (%s)" % (spec.name, "; ".join(parts))
    shown = [_show(value) for value in spec.examples[:3] if value is not None]
    if shown and not spec.enum:
        line += " e.g. %s" % ", ".join(shown)
    return line


def _sample_cases(task):
    """Two gold cases, clipped. Concrete beats a field table for shape questions."""
    lines = []
    for case in task.cases[:2]:
        lines.append(
            "%s -> %s"
            % (
                _clip(json.dumps(case.get("input", {}), sort_keys=True), 200),
                _clip(json.dumps(case.get("expect", {}), sort_keys=True), 200),
            )
        )
    return lines


def _endings_evidence(task):
    ends = _observed_ends(task)
    if len(ends) > 1:
        counted = ", ".join(
            "%s (%d case%s)" % (name, count, "" if count == 1 else "s")
            for name, count in sorted(ends.items())
        )
        return [
            "The gold examples end in %d different places: %s." % (len(ends), counted),
            "Propose exactly those endings, and say which field value routes to each.",
        ]
    return [
        "Every gold example ends the same way. Propose exactly one ending and do not "
        "invent a branch."
    ]


def _observed_ends(task):
    """`{ending name: how many gold cases end there}` — the only evidence for a branch."""
    counts = {}
    for case in task.cases:
        end = case.get("end")
        if isinstance(end, str) and end:
            counts[end] = counts.get(end, 0) + 1
    return counts


def _check_plan(task, answer):
    names = _check_node_names(answer["nodes"])
    written = _check_writes(task, answer["nodes"])
    _check_reads(task, answer["nodes"], written)
    _check_two_stage(answer["nodes"])
    _check_endings(task, answer["endings"], names, written)


def _check_node_names(nodes):
    if not nodes:
        raise _Reject("the plan has no nodes")
    names = []
    for node in nodes:
        name = node["name"]
        if not _NAME.match(name):
            raise _Reject(
                "node name %r is not a lowercase identifier" % _clip(name, 40)
            )
        if name in names:
            raise _Reject("two nodes are both called %r" % name)
        if not node["purpose"].strip():
            raise _Reject("node %r has an empty purpose" % name)
        names.append(name)
    return names


def _check_writes(task, nodes):
    """Return `{field: node that writes it}`, or reject. This is the central check."""
    declared = [spec.name for spec in task.fields]
    written = {}
    for node in nodes:
        writes = node["writes"]
        if not writes:
            raise _Reject(
                "node %r writes no field; every node must write at least one, and a "
                "node that only reads belongs in the node that uses it" % node["name"]
            )
        if len(writes) > MAX_WRITES_PER_NODE:
            raise _Reject(
                "node %r writes %d fields (%s); split it — at most %d per node"
                % (node["name"], len(writes), ", ".join(writes), MAX_WRITES_PER_NODE)
            )
        for name in writes:
            if name not in declared:
                raise _Reject(
                    "node %r writes %r, which is not one of the task's fields (%s)"
                    % (node["name"], _clip(name, 40), ", ".join(declared))
                )
            if name in written:
                if written[name] == node["name"]:
                    raise _Reject(
                        "node %r lists %r twice; every field must be written by exactly "
                        "one node, once" % (node["name"], name)
                    )
                raise _Reject(
                    "%r is written twice, by %r and by %r; every field must be written "
                    "by exactly one node" % (name, written[name], node["name"])
                )
            written[name] = node["name"]
    missing = [name for name in declared if name not in written]
    if missing:
        raise _Reject(
            "no node writes %s; every field must be written by exactly one node"
            % ", ".join(repr(name) for name in missing)
        )
    return written


def _check_reads(task, nodes, written):
    position = {node["name"]: index for index, node in enumerate(nodes)}
    for index, node in enumerate(nodes):
        for name in node["reads"]:
            if name in task.inputs:
                continue
            if name not in written:
                raise _Reject(
                    "node %r reads %r, which is neither a run input (%s) nor a field "
                    "any node writes"
                    % (node["name"], _clip(name, 40), ", ".join(task.inputs) or "none")
                )
            source = written[name]
            if position[source] >= index:
                raise _Reject(
                    "node %r reads %r, but %r is written by %r, which does not run "
                    "before it; order the nodes so a field is written before it is read"
                    % (node["name"], name, name, source)
                )


def _check_two_stage(nodes):
    thinkers = [node["name"] for node in nodes if node["two_stage"]]
    # Every two_stage node doubles that node's calls. A plan where most nodes think is a
    # plan that has not decomposed the judgement out into its own step. A one-node plan
    # is exempt: there is no other step for the judgement to move to.
    if len(nodes) > 1 and len(thinkers) * 2 > len(nodes):
        raise _Reject(
            "%d of %d nodes are two_stage (%s); reserve it for the nodes with a real "
            "judgement to make"
            % (len(thinkers), len(nodes), ", ".join(thinkers))
        )


def _check_endings(task, endings, node_names, written):
    if not endings:
        raise _Reject("the plan has no endings; every run has to end somewhere")
    seen = []
    for ending in endings:
        name = ending["name"]
        if not _NAME.match(name):
            raise _Reject(
                "ending name %r is not a lowercase identifier" % _clip(name, 40)
            )
        if name in seen:
            raise _Reject("two endings are both called %r" % name)
        if name in node_names:
            raise _Reject(
                "%r is both a node and an ending; an end node writes nothing" % name
            )
        seen.append(name)

    fallthrough = endings[-1]
    if fallthrough["when_field"] or fallthrough["when_equals"]:
        raise _Reject(
            "the last ending (%r) must be the unconditional fallthrough, with an empty "
            "when_field and when_equals" % fallthrough["name"]
        )

    observed = _observed_ends(task)
    if observed:
        if sorted(observed) != sorted(seen):
            raise _Reject(
                "the gold examples end in %s; the plan proposes %s"
                % (", ".join(sorted(observed)), ", ".join(sorted(seen)))
            )
    elif len(endings) != 1:
        raise _Reject(
            "the gold examples show only one ending, so the graph needs exactly one; "
            "the plan proposes %d (%s)" % (len(endings), ", ".join(seen))
        )

    for ending in endings[:-1]:
        _check_condition(task, ending, written)
    _replay_endings(task, endings)


def _check_condition(task, ending, written):
    field = ending["when_field"]
    if not field or not ending["when_equals"]:
        raise _Reject(
            "ending %r is not the last one, so it needs both a when_field and a "
            "when_equals" % ending["name"]
        )
    if field not in written:
        raise _Reject(
            "ending %r is gated on %r, which no node writes"
            % (ending["name"], _clip(field, 40))
        )
    _condition_value(task, ending)


def _condition_value(task, ending):
    """The `when_equals` string as the typed value the gate compares against.

    `when` in graph.yaml is `==` against committed state, so `"true"` and `True` are not
    the same gate. The string the planner gave is coerced to the field's own type and
    rejected if it does not name a value the gold examples actually contain.
    """
    spec = task.field_named(ending["when_field"])
    text = ending["when_equals"]
    if spec.type == "boolean":
        if text not in ("true", "false"):
            raise _Reject(
                "ending %r gates on boolean %r, so when_equals must be \"true\" or "
                "\"false\", not %r" % (ending["name"], spec.name, _clip(text, 40))
            )
        value = text == "true"
    elif spec.type in ("integer", "number"):
        try:
            value = int(text) if spec.type == "integer" else float(text)
        except ValueError:
            raise _Reject(
                "ending %r gates on %s %r, but when_equals %r is not a number"
                % (ending["name"], spec.type, spec.name, _clip(text, 40))
            )
    else:
        value = text

    observed = _observed_values(task, spec.name)
    if observed and value not in observed:
        raise _Reject(
            "ending %r gates on %s == %s, a value no gold example has for that field"
            % (ending["name"], spec.name, _show(value))
        )
    return value


def _observed_values(task, field):
    """Every value the gold cases give `field`. A list, because a value may be unhashable."""
    values = []
    for case in task.cases:
        expect = case.get("expect")
        if isinstance(expect, dict) and field in expect and expect[field] not in values:
            values.append(expect[field])
    return values


def _replay_endings(task, endings):
    """Route every gold case through the proposed gates and insist they land right.

    This is the check that separates a plausible branch from a correct one. Two gates can
    both look right in prose — "escalate is true" and "priority is p0" — and only one of
    them reproduces the gold endings.
    """
    gates = [
        (ending["name"], ending["when_field"], _condition_value(task, ending))
        for ending in endings[:-1]
    ]
    for case in task.cases:
        wanted = case.get("end")
        if not isinstance(wanted, str) or not wanted:
            continue
        expect = case.get("expect")
        if not isinstance(expect, dict):
            continue
        landed = _route(gates, endings[-1]["name"], expect)
        if landed != wanted:
            raise _Reject(
                "gold case %r ends in %r, but the proposed endings route it to %r"
                % (_clip(str(case.get("name", "")), 60), wanted, landed)
            )


def _route(gates, fallthrough, expect):
    """Which ending `expect` reaches — the same first-match-wins rule graph.py uses."""
    for name, field, value in gates:
        if field in expect and expect[field] == value:
            return name
    return fallthrough


def _build_plan(task, answer):
    """Turn a validated answer into a GraphPlan, deriving the edges from the order."""
    nodes = [
        NodePlan(
            name=node["name"],
            writes=list(node["writes"]),
            purpose=node["purpose"].strip(),
            two_stage=bool(node["two_stage"]),
            reads=list(node["reads"]),
        )
        for node in answer["nodes"]
    ]
    endings = answer["endings"]

    edges = [
        {"from": nodes[index].name, "to": nodes[index + 1].name}
        for index in range(len(nodes) - 1)
    ]
    last = nodes[-1].name
    # Conditional edges first and the fallthrough last, because graph.py takes the first
    # matching edge and an unconditional edge matches everything.
    for ending in endings[:-1]:
        edges.append(
            {
                "from": last,
                "to": ending["name"],
                # The typed value, not the string the planner wrote: `when` is `==`
                # against committed state, where `escalate` is a boolean.
                "when": {ending["when_field"]: _condition_value(task, ending)},
            }
        )
    edges.append({"from": last, "to": endings[-1]["name"]})

    return GraphPlan(
        entry=nodes[0].name,
        nodes=nodes,
        endings=[ending["name"] for ending in endings],
        edges=edges,
    )


# ---------------------------------------------------------------------------
# write_prompts
# ---------------------------------------------------------------------------


def write_prompts(task, plan, model):
    """Write one prompt per node in `plan`, keyed by node name.

    One entry per node and nothing else. A two_stage node's optional `<node>.think.txt`
    is deliberately not emitted: jig/codegen.py already derives a think prompt from the
    emit prompt when the file is absent, and inventing a second key would put a filename
    convention into a contract that says one prompt per node.
    """
    prompts = {}
    for node in plan.nodes:
        answer = _ask(
            model,
            _scribe_prompt(task, node),
            _PROMPT_SCHEMA,
            512,
            lambda obj, node=node: _check_prompt(task, node, obj["prompt"]),
            "prompt for node %r" % node.name,
        )
        prompts[node.name] = answer["prompt"].strip()
    return prompts


def _available(task, node):
    """The state names this node's prompt may substitute, and where each came from."""
    available = [(name, "the run's input") for name in task.inputs]
    for name in node.reads:
        if name not in task.inputs:
            available.append((name, "written by an earlier step"))
    if node.two_stage:
        available.append((_SCRATCHPAD, "this step's own notes from its think pass"))
    return available


def _scribe_prompt(task, node):
    lines = [
        "You are writing the prompt for one step of a workflow. The step is a single "
        "call to a 7B model, and a JSON grammar already forces the shape of its answer.",
        "",
        "Workflow: %s" % task.name,
    ]
    if task.description.strip():
        lines.append(task.description.strip())
    lines.append("")
    lines.append("This step is %r: %s" % (node.name, node.purpose))
    lines.append("")
    lines.append("It must produce exactly these fields:")
    for name in node.writes:
        lines.append("  " + _field_line(task.field_named(name)))
    lines.append("")
    lines.append("It may name these state values, in braces:")
    for name, origin in _available(task, node):
        lines.append("  {%s} - %s" % (name, origin))
    lines.append("")
    lines.append("Rules for the prompt you write:")
    lines.append("  - State the one job, then name each field and say what it means.")
    lines.append(
        "  - Spell out every listed value literally: the grammar makes a wrong value "
        "impossible, but only the prompt can say what the right one means."
    )
    lines.append(
        "  - Name every state value listed above, each in braces, and name nothing else."
    )
    lines.append(
        "  - Under %d characters, and no reasoning instructions: the grammar handles "
        "the format%s."
        % (
            MAX_PROMPT_CHARS,
            " and this node's think pass handles the thinking" if node.two_stage else "",
        )
    )
    lines.append("")
    lines.append('Answer with a JSON object: {"prompt": "..."}')
    return "\n".join(lines)


def _check_prompt(task, node, text):
    if len(text.strip()) < MIN_PROMPT_CHARS:
        raise _Reject(
            "the prompt is %d characters; state the job, then the fields and what "
            "their values mean" % len(text.strip())
        )
    if len(text) > MAX_PROMPT_CHARS:
        raise _Reject(
            "the prompt is %d characters; keep it under %d — one job, no cleverness"
            % (len(text), MAX_PROMPT_CHARS)
        )

    lowered = text.lower()
    for phrase in _COT_PHRASES:
        if phrase in lowered:
            raise _Reject(
                "the prompt says %r; leave the reasoning out of an emit prompt — the "
                "grammar handles the format%s"
                % (
                    phrase,
                    " and two_stage handles the thinking" if node.two_stage else "",
                )
            )

    used = _variables_used(task, node, text)
    if not used:
        raise _Reject(
            "the prompt names no state value, so the model would answer without ever "
            "seeing the run's input"
        )
    for name in node.reads:
        if name not in used:
            raise _Reject(
                "the plan says this node reads %r, but the prompt never names {%s}"
                % (name, name)
            )

    for name in node.writes:
        if name not in text:
            raise _Reject(
                "the prompt never mentions %r, which this node has to produce" % name
            )
        _check_values_quoted(task.field_named(name), text)


def _variables_used(task, node, text):
    """The state names the prompt actually substitutes — found with jig's own renderer.

    Rendering a probe state is the check, rather than a second copy of render's regex:
    whatever the runtime would substitute is exactly what is counted here, and whatever
    the runtime would refuse raises MissingVariable here instead of at 3am.
    """
    allowed = [name for name, _ in _available(task, node)]
    if _SCRATCHPAD in text and not node.two_stage:
        raise _Reject(
            "the prompt names {%s}, which only exists on a two_stage node" % _SCRATCHPAD
        )
    # A NUL-delimited marker, because a prompt that happens to *write* "<<ticket>>" is
    # not a prompt that substitutes {ticket}, and only the substitution counts here.
    probe = {name: "\x00%s\x00" % name for name in allowed}
    try:
        rendered = render(text, probe)
    except MissingVariable as exc:
        raise _Reject(
            "%s; this node may only name %s"
            % (str(exc).split(" but state has ")[0], ", ".join(allowed) or "nothing")
        )
    return [name for name in allowed if ("\x00%s\x00" % name) in rendered]


def _check_values_quoted(spec, text):
    """An enum small enough to spell out must be spelled out, in full."""
    values = [value for value in (spec.enum or []) if isinstance(value, str)]
    if not values:
        return
    absent = [value for value in values if value not in text]
    if len(values) <= MAX_ENUM_TO_SPELL_OUT:
        if absent:
            raise _Reject(
                "the prompt never gives %s for %r; a %d-value enum belongs in the "
                "prompt in full"
                % (", ".join(repr(value) for value in absent), spec.name, len(values))
            )
    elif len(values) - len(absent) < MIN_ENUM_TO_QUOTE:
        raise _Reject(
            "the prompt quotes fewer than %d of the %d values %r can take; show a few"
            % (MIN_ENUM_TO_QUOTE, len(values), spec.name)
        )


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def _ask(model, prompt, schema, max_tokens, check, what):
    """Ask for one JSON answer and re-ask until it passes `check`.

    The same three rungs the runtime spends on a node, and the same rule about what a
    retry is allowed to say: the re-ask carries the reason the last answer was rejected,
    never the answer itself. Feeding a model its own bad output back is the
    self-conditioning spiral jig/verify.py exists to prevent, and a compiler that did it
    would be arguing against its own runtime.

    The one thing a rejection may repeat back is a *name* the model invented — the field
    it wrote that does not exist, the variable it reached for — because a name is what
    the model has to change and it is one clipped token, not the draft. The draft itself,
    the prompt text or the plan body, is never quoted.
    """
    grammar = schema_to_grammar(schema)
    reason = "no answer was asked for"
    for rung in range(ATTEMPTS):
        text = model.generate(
            _reask(prompt, reason if rung else None),
            grammar=grammar,
            max_tokens=max_tokens,
        )
        try:
            answer = extract_json(text)
        except Rejected as exc:
            # `feedback`, not `str(exc)`: the detail half quotes the text that came back.
            reason = exc.feedback
            continue
        try:
            validate_against(schema, answer)
            check(answer)
        except (ValidationError, _Reject) as exc:
            reason = exc.safe_text if isinstance(exc, ValidationError) else str(exc)
            continue
        return answer
    raise BuildError(
        "the model did not produce a usable %s in %d attempts; the last one was "
        "rejected because %s" % (what, ATTEMPTS, reason)
    )


def _reask(prompt, reason):
    if reason is None:
        return prompt
    return (
        "%s\n\nYour previous answer was rejected: %s\nAnswer again, fixing exactly that."
        % (prompt, reason)
    )


def _show(value):
    return value if isinstance(value, str) else json.dumps(value)


def _clip(text, limit=120):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."
