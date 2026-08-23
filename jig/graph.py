"""The graph walker — jig's runtime.

This is the whole "the small model never plans" idea in one loop (docs/PLAN.md §3): the
walker decides what happens next, the model only ever fills one node's slot. Nothing here
asks a model where to go; edges are data.

    state = inputs
    node  = entry
    loop:  execute node -> commit its output to state -> pick the next edge

Three node types. `generate` renders a prompt from state, calls the model under the
node's grammar, and commits the parsed object. `assert` evaluates a deterministic
expression and either continues or diverts to `on_fail`. `end` stops and returns.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import (
    AssertFailed,
    DanglingEdge,
    DeadEnd,
    MaxStepsExceeded,
    NodeFailed,
)
from .expr import is_true
from .verify import run_node

__all__ = ["Failure", "RunResult", "run"]


@dataclass
class Failure:
    """A node that exhausted its retry ladder and was diverted by `on_fail`."""

    node: str
    reason: str
    attempts: int


@dataclass
class RunResult:
    """What a finished run leaves behind."""

    run_id: str
    output: Dict[str, Any]
    state: Dict[str, Any]
    path: List[str] = field(default_factory=list)
    steps: int = 0
    provenance: Dict[str, str] = field(default_factory=dict)
    end_node: Optional[str] = None
    failures: List["Failure"] = field(default_factory=list)


def run(pack, model, inputs=None, run_id=None, max_steps=None):
    """Walk `pack` from its entry node until an `end` node, and return a `RunResult`."""
    state = dict(inputs or {})
    provenance = {}
    path = []
    failures = []
    budget = max_steps if max_steps is not None else pack.max_steps

    node = _node(pack, pack.entry, "entry node")
    steps = 0
    while True:
        steps += 1
        if steps > budget:
            raise MaxStepsExceeded(
                "run exceeded max_steps=%d at node %r — the graph is looping"
                % (budget, node.name)
            )
        path.append(node.name)

        if node.type == "end":
            return RunResult(
                run_id=run_id or uuid.uuid4().hex,
                output=_project(node, state),
                state=state,
                path=path,
                steps=steps,
                provenance=provenance,
                end_node=node.name,
                failures=failures,
            )

        if node.type == "generate":
            try:
                value = run_node(node, state, model)
            except NodeFailed as exc:
                # The ladder is spent. Divert if the node declares where to go; the
                # rejected output is dropped either way and never touches state.
                if not node.on_fail:
                    raise
                failures.append(
                    Failure(node=exc.node, reason=exc.reason, attempts=exc.attempts)
                )
                node = _node(pack, node.on_fail, "on_fail of %r" % node.name)
                continue
            commit(node, value, state, provenance)
            node = _node(pack, _next(pack, node, state), "edge from %r" % node.name)
            continue

        if node.type == "assert":
            if is_true(node.expr, state):
                node = _node(pack, _next(pack, node, state), "edge from %r" % node.name)
            elif node.on_fail:
                node = _node(pack, node.on_fail, "on_fail of %r" % node.name)
            else:
                raise AssertFailed(node.name, node.expr)
            continue

        raise DeadEnd("node %r has unknown type %r" % (node.name, node.type))


def commit(node, value, state, provenance):
    """Write a node's verified output into run state, recording who wrote what."""
    if node.output:
        state[node.output] = value
        provenance[node.output] = node.name
        return
    for key, item in value.items():
        state[key] = item
        provenance[key] = node.name


# ------------------------------------------------------------------------ traversal


def _node(pack, name, why):
    if name not in pack.nodes:
        raise DanglingEdge("%s points at undefined node %r" % (why, name))
    return pack.nodes[name]


def _next(pack, node, state):
    for edge in pack.edges_from(node.name):
        if _matches(edge.when, state):
            return edge.target
    raise DeadEnd(
        "no outgoing edge from %r matched the current state" % node.name
    )


def _matches(when, state):
    if not when:
        return True
    for path, expected in when.items():
        if _lookup(path, state) != expected:
            return False
    return True


_MISSING = object()


def _lookup(path, state):
    current = state
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _project(node, state):
    if not node.output:
        return dict(state)
    return {key: state[key] for key in node.output if key in state}
