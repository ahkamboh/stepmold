"""The graph walker — jig's runtime.

This is the whole "the small model never plans" idea in one loop (docs/PLAN.md §3): the
walker decides what happens next, the model only ever fills one node's slot. Nothing here
asks a model where to go; edges are data.

    state = inputs
    node  = entry
    loop:  execute node -> commit its output to state -> pick the next edge

Three node types. `generate` renders a prompt from state, generates under the node's
grammar, and commits the result — but only once `jig.verify` has accepted it, so a
rejected output never lands here. `assert` evaluates a deterministic expression and
either continues or diverts to `on_fail`. `end` stops and returns.

Failure routing is uniform, and that is deliberate: everything that stops a node
producing a verified output — a spent retry ladder, a prompt that cannot be rendered, an
expression that cannot be evaluated — takes the node's declared `on_fail` edge, and only
escapes as an exception when the node declares none. A node's failure edge is a promise
in the pack; the walker keeps it whatever the node failed on.
"""

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .errors import (
    AssertFailed,
    DanglingEdge,
    DeadEnd,
    ExprError,
    MaxStepsExceeded,
    MissingVariable,
    NodeFailed,
    RunError,
    RunIdInUse,
)
from .expr import is_true
from .verify import run_node

__all__ = ["Failure", "RunResult", "StateCollision", "replay", "run"]


class StateCollision(RunError):
    """A node's commit would have overwritten one of the run's inputs.

    Merge-mode commit (a node with no `output:` key) drops its fields straight into run
    state, and run state is what every later prompt and every edge condition reads. Two
    kinds of collision can happen there, and they are not the same kind of accident:

    * **Node over node** is *recorded, not refused.* The graph author wrote two nodes
      that emit the same field, `provenance` names whichever wrote it last, and the
      earlier value is still in that node's checkpoint — nothing is lost.
    * **Node over a run input** is refused, because nothing records it. The caller's
      value is simply gone, and from then on the prompts and edge conditions that meant
      to read the caller's input read model output instead. Refusing is the same call
      `jig.verify` makes about a bad generation: nothing lands until it is safe.

    This one lives with the walker rather than in `jig.errors` because it is a rule about
    committing, which is the walker's own job; it still subclasses `RunError`, so every
    caller that already handles run errors handles it.
    """


@dataclass
class Failure:
    """A node that could not produce a verified output and was diverted by `on_fail`.

    `attempts` is how many generations were spent getting there — zero when the node's
    prompt could not even be rendered, since that failure costs no model call.
    """

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

    if resume_from is None and store is not None and run_id is not None:
        # A fresh run must not inherit another run's checkpoint chain. Without this,
        # a reused id silently splices two runs and `resume` replays the older one's
        # output as if it belonged to this run.
        previous = getattr(store, "latest", None)
        if previous is not None and previous(run_id) is not None:
            raise RunIdInUse(
                "run id %r already has checkpoints in this store; "
                "resume it, delete it, or choose another id" % run_id
            )

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
            except (NodeFailed, MissingVariable) as exc:
                # The two ways a generate node can fail to produce a verified output, and
                # `on_fail` is the declared answer to both: a spent retry ladder, and a
                # prompt that never rendered because it names state nobody wrote. The
                # second one costs no generation — no re-sample can fix a template — so
                # the ladder is skipped, not silently bypassed. Anything else (a backend
                # that is down, say) is not the node's declared failure mode and still
                # stops the run. Either way the rejected output is dropped and never
                # touches state.
                if not node.on_fail:
                    raise
                failures.append(_failure(node, exc))
                _checkpoint(store, pack, run_id, steps, node.name, node.on_fail, state,
                            path, provenance, failures)
                node = _node(pack, node.on_fail, "on_fail of %r" % node.name)
                continue
            commit(node, value, state, provenance)
            try:
                following = _next(pack, node, state)
            except DeadEnd:
                # The output is verified and already in state; only the routing failed.
                # Checkpoint before the DeadEnd escapes, or a node that really did run
                # leaves no record of what it committed. `next_node` points back at this
                # node because `None` is how a checkpoint says the run finished
                # (`state.Checkpoint.finished`) — this one did not, and an operator who
                # repairs the graph resumes by running this node again.
                _checkpoint(store, pack, run_id, steps, node.name, node.name, state,
                            path, provenance, failures)
                raise
            _checkpoint(store, pack, run_id, steps, node.name, following, state, path,
                        provenance, failures)
            node = _node(pack, following, "edge from %r" % node.name)
            continue

        if node.type == "assert":
            try:
                passed = is_true(node.expr, state)
            except ExprError:
                # An expression that cannot be evaluated is not a true one, and it is
                # what `on_fail` is for. `jig.verify` already downgrades the identical
                # call this way for a node's `assert:` (see `verify._check_assert`), so
                # both call sites route an unevaluable expression to the same place.
                # With nowhere to divert, the ExprError itself escapes rather than an
                # AssertFailed: it names the identifier that is missing, and the
                # expression was never false — it was unanswerable.
                if not node.on_fail:
                    raise
                passed = False
            if passed:
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
    """Write a node's verified output into run state, recording who wrote what.

    A key already in state may be written again — a node inside a loop rewrites its own
    counter every time round, and a later node may deliberately restate an earlier one's
    field — and `provenance` records who wrote it last. The one collision that is refused
    is a key that came from the run's inputs, because that overwrite leaves no record at
    all; see `StateCollision`.

    Every key is checked before the first one is written, so a refused commit leaves
    state and provenance exactly as they were.
    """
    keys = [node.output] if node.output else list(value)
    for key in keys:
        # In state but never committed by a node — so the caller supplied it.
        if key in state and key not in provenance:
            raise StateCollision(
                "node %r would overwrite %r, which came from the run inputs; give the "
                "node its own `output:` key or rename the field in its grammar"
                % (node.name, key)
            )
    if node.output:
        state[node.output] = value
        provenance[node.output] = node.name
        return
    for key, item in value.items():
        state[key] = item
        provenance[key] = node.name


def _failure(node, exc):
    """Record whichever way this node failed, for `RunResult.failures`."""
    if isinstance(exc, NodeFailed):
        return Failure(node=exc.node, reason=exc.reason, attempts=exc.attempts)
    # An unrenderable prompt is caught before the first generation, so nothing was spent.
    return Failure(node=node.name, reason=str(exc), attempts=0)


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
