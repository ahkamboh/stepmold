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
from dataclasses import asdict, dataclass, field
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

__all__ = ["Failure", "RunResult", "replay", "run"]


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


def run(pack, model, inputs=None, run_id=None, max_steps=None, store=None,
        resume_from=None):
    """Walk `pack` from its entry node until an `end` node, and return a `RunResult`.

    `store` is anything with a `save(...)` method (see `jig.state.Store`); when given, a
    checkpoint is written after every node that completes. `resume_from` is a checkpoint
    to continue from instead of starting at the entry node — that is how `state.resume`
    picks a dead run back up without re-executing what already succeeded.
    """
    budget = max_steps if max_steps is not None else pack.max_steps

    if resume_from is not None:
        run_id = run_id or resume_from.run_id
        state = dict(resume_from.state)
        provenance = dict(resume_from.provenance)
        path = list(resume_from.path)
        failures = [Failure(**record) for record in resume_from.failures]
        steps = resume_from.step
        node = _node(pack, resume_from.next_node, "resume point of run %r" % run_id)
    else:
        state = dict(inputs or {})
        provenance = {}
        path = []
        failures = []
        steps = 0
        node = _node(pack, pack.entry, "entry node")
    run_id = run_id or uuid.uuid4().hex
    while True:
        steps += 1
        if steps > budget:
            raise MaxStepsExceeded(
                "run exceeded max_steps=%d at node %r — the graph is looping"
                % (budget, node.name)
            )
        path.append(node.name)

        if node.type == "end":
            output = _project(node, state)
            _checkpoint(store, pack, run_id, steps, node.name, None, state, path,
                        provenance, failures, output)
            return RunResult(
                run_id=run_id,
                output=output,
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
                _checkpoint(store, pack, run_id, steps, node.name, node.on_fail, state,
                            path, provenance, failures)
                node = _node(pack, node.on_fail, "on_fail of %r" % node.name)
                continue
            commit(node, value, state, provenance)
            following = _next(pack, node, state)
            _checkpoint(store, pack, run_id, steps, node.name, following, state, path,
                        provenance, failures)
            node = _node(pack, following, "edge from %r" % node.name)
            continue

        if node.type == "assert":
            if is_true(node.expr, state):
                following = _next(pack, node, state)
            elif node.on_fail:
                following = node.on_fail
            else:
                raise AssertFailed(node.name, node.expr)
            _checkpoint(store, pack, run_id, steps, node.name, following, state, path,
                        provenance, failures)
            node = _node(pack, following, "edge from %r" % node.name)
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


def _checkpoint(store, pack, run_id, step, node, next_node, state, path, provenance,
                failures, output=None):
    """Persist one completed node, if this run is being checkpointed at all."""
    if store is None:
        return
    store.save(
        run_id=run_id,
        step=step,
        node=node,
        next_node=next_node,
        state=state,
        path=path,
        provenance=provenance,
        failures=[asdict(failure) for failure in failures],
        output=output,
        pack=pack.name,
    )


def replay(checkpoint):
    """Rebuild the `RunResult` of a run that already finished, from its checkpoint."""
    return RunResult(
        run_id=checkpoint.run_id,
        output=checkpoint.output if checkpoint.output is not None else dict(checkpoint.state),
        state=dict(checkpoint.state),
        path=list(checkpoint.path),
        steps=checkpoint.step,
        provenance=dict(checkpoint.provenance),
        end_node=checkpoint.node,
        failures=[Failure(**record) for record in checkpoint.failures],
    )
