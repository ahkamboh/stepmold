"""The graph walker — jig's runtime.

This is the whole "the small model never plans" idea in one loop (docs/ARCHITECTURE.md §3): the
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

The walk is also where a run's story gets written down (`jig.log`): `run.start`,
`node.ok` per node with the generations it spent and the milliseconds it took,
`edge.on_fail` when a node is diverted, `run.end` with the whole run's totals, and
`run.error` when it stops short. Names and counts only — the caller's data is reported as
a size and a digest, and a node failure carries the model-safe half of its reason (see
`_safe_reason`), with the full detail at DEBUG.
"""

import inspect
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache
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
from .log import DEBUG, ERROR, INFO, WARNING, digest, event, get_logger, size_of
from .verify import run_node

__all__ = ["Failure", "RunResult", "StateCollision", "replay", "run"]

_log = get_logger("graph")


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
    """What a finished run leaves behind.

    `attempts` counts the generations each `generate` node spent, so a run that was
    rescued is distinguishable from one that never needed rescuing. `failures` records
    only the nodes that ran out of ladder; a node that was rejected twice and got there
    on the third rung leaves no failure and used to leave no trace at all — which is
    exactly the run an operator wants to see *before* the next one fails. A node absent
    from the map spent one generation or none; a node at `retries + 1` was one rung from
    diverting.
    """

    run_id: str
    output: Dict[str, Any]
    state: Dict[str, Any]
    path: List[str] = field(default_factory=list)
    steps: int = 0
    provenance: Dict[str, str] = field(default_factory=dict)
    end_node: Optional[str] = None
    failures: List["Failure"] = field(default_factory=list)
    attempts: Dict[str, int] = field(default_factory=dict)


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
        # A store older than per-node attempt counts has no column for them; the run
        # resumes with what it can and keeps counting from there.
        attempts = dict(getattr(resume_from, "attempts", None) or {})
        steps = resume_from.step
        node = _node(pack, resume_from.next_node, "resume point of run %r" % run_id)
    else:
        state = dict(inputs or {})
        provenance = {}
        path = []
        failures = []
        attempts = {}
        steps = 0
        node = _node(pack, pack.entry, "entry node")
    run_id = run_id or uuid.uuid4().hex

    # One clock read per run and one per node, taken unconditionally. Guarding them
    # behind `isEnabledFor` would save tens of nanoseconds next to a node that blocks for
    # a fifth of a second on a GPU, and would cost the thing this module is for: a
    # duration that is missing whenever the level changed halfway through a run.
    run_started = time.perf_counter()
    if _log.isEnabledFor(INFO):
        event(_log, INFO, "run.start", run_id=run_id, pack=pack.name,
              version=pack.version, entry=node.name, resumed=resume_from is not None,
              max_steps=budget, inputs=sorted(str(key) for key in state))
    if resume_from is not None:
        event(_log, INFO, "run.resumed", run_id=run_id, pack=pack.name,
              from_step=resume_from.step, next_node=node.name,
              done=len(path), failures=len(failures))

    try:
        while True:
            steps += 1
            if steps > budget:
                raise MaxStepsExceeded(
                    "run exceeded max_steps=%d at node %r — the graph is looping"
                    % (budget, node.name)
                )
            path.append(node.name)
            node_started = time.perf_counter()

            if node.type == "end":
                output = _project(node, state)
                _checkpoint(store, pack, run_id, steps, node.name, None, state, path,
                            provenance, failures, attempts, output)
                if _log.isEnabledFor(INFO):
                    # The line an operator greps for first: did it finish, where, how
                    # many generations did it burn, and how long did the whole thing
                    # take. Sizes, never contents — the output is the caller's data.
                    event(_log, INFO, "run.end", run_id=run_id, pack=pack.name,
                          end_node=node.name, steps=steps,
                          generations=sum(attempts.values()), failures=len(failures),
                          output_keys=len(output), output_bytes=size_of(output),
                          duration_ms=_ms(run_started))
                return RunResult(
                    run_id=run_id,
                    output=output,
                    state=state,
                    path=path,
                    steps=steps,
                    provenance=provenance,
                    end_node=node.name,
                    failures=failures,
                    attempts=attempts,
                )

            if node.type == "generate":
                # `attempts` is cumulative across visits (a node inside a loop is
                # entered more than once), so this visit's cost is the difference.
                before = attempts.get(node.name, 0)
                try:
                    value = run_node(node, state, model, attempts=attempts)
                except (NodeFailed, MissingVariable) as exc:
                    # The two ways a generate node can fail to produce a verified output,
                    # and `on_fail` is the declared answer to both: a spent retry ladder,
                    # and a prompt that never rendered because it names state nobody
                    # wrote. The second one costs no generation — no re-sample can fix a
                    # template — so the ladder is skipped, not silently bypassed. A
                    # backend that is *down* is not the node's declared failure mode and
                    # still stops the run, but a 200 that carried no text is a bad sample
                    # rather than a dead endpoint: `verify.run_node` spends a rung on it,
                    # so it arrives here as a NodeFailed and takes this edge like any
                    # other. Either way the rejected output is dropped and never touches
                    # state.
                    if _log.isEnabledFor(WARNING):
                        event(_log, WARNING, "node.failed", run_id=run_id,
                              node=node.name, type=node.type,
                              attempts=attempts.get(node.name, 0) - before,
                              error=type(exc).__name__, reason=_safe_reason(exc),
                              on_fail=node.on_fail, duration_ms=_ms(node_started))
                    if _log.isEnabledFor(DEBUG):
                        # The unsafe half, once, at the only level allowed to hold it.
                        event(_log, DEBUG, "node.failed.detail", run_id=run_id,
                              node=node.name, detail=str(exc))
                    if not node.on_fail:
                        raise
                    failures.append(_failure(node, exc))
                    _checkpoint(store, pack, run_id, steps, node.name, node.on_fail,
                                state, path, provenance, failures, attempts)
                    event(_log, INFO, "edge.on_fail", run_id=run_id, node=node.name,
                          to=node.on_fail)
                    node = _node(pack, node.on_fail, "on_fail of %r" % node.name)
                    continue
                commit(node, value, state, provenance)
                if _log.isEnabledFor(INFO):
                    event(_log, INFO, "node.ok", run_id=run_id, node=node.name,
                          type=node.type,
                          attempts=attempts.get(node.name, 0) - before,
                          # "merge" is the walker's own word for a node with no
                          # `output:` key, whose fields drop straight into state.
                          output=node.output or "merge",
                          duration_ms=_ms(node_started))
                    # Nested rather than a second top-level check, and that is sound
                    # rather than clever: levels are thresholds, so DEBUG being enabled
                    # implies INFO is. The common case — logging off — asks once per
                    # node instead of twice, and this is the innermost loop jig has.
                    if _log.isEnabledFor(DEBUG):
                        # A fingerprint, not the state: two runs that diverge did so at
                        # the first node whose digest differs, and nobody reading the log
                        # learns a byte of the caller's data.
                        event(_log, DEBUG, "state.committed", run_id=run_id,
                              node=node.name, keys=len(state),
                              state_bytes=size_of(state), state_digest=digest(state))
                try:
                    following = _next(pack, node, state)
                except DeadEnd:
                    # The output is verified and already in state; only the routing
                    # failed. Checkpoint before the DeadEnd escapes, or a node that
                    # really did run leaves no record of what it committed. `next_node`
                    # points back at this node because `None` is how a checkpoint says
                    # the run finished (`state.Checkpoint.finished`) — this one did not,
                    # and an operator who repairs the graph resumes by running this node
                    # again.
                    _checkpoint(store, pack, run_id, steps, node.name, node.name, state,
                                path, provenance, failures, attempts)
                    raise
                _checkpoint(store, pack, run_id, steps, node.name, following, state,
                            path, provenance, failures, attempts)
                if _log.isEnabledFor(DEBUG):
                    event(_log, DEBUG, "edge.taken", run_id=run_id, node=node.name,
                          to=following)
                node = _node(pack, following, "edge from %r" % node.name)
                continue

            if node.type == "assert":
                try:
                    passed = is_true(node.expr, state)
                except ExprError:
                    # An expression that cannot be evaluated is not a true one, and it is
                    # what `on_fail` is for. `jig.verify` already downgrades the identical
                    # call this way for a node's `assert:` (see `verify._check_assert`),
                    # so both call sites route an unevaluable expression to the same
                    # place. With nowhere to divert, the ExprError itself escapes rather
                    # than an AssertFailed: it names the identifier that is missing, and
                    # the expression was never false — it was unanswerable.
                    if not node.on_fail:
                        raise
                    passed = False
                if passed:
                    following = _next(pack, node, state)
                elif node.on_fail:
                    following = node.on_fail
                else:
                    event(_log, WARNING, "node.failed", run_id=run_id, node=node.name,
                          type=node.type, attempts=0, error="AssertFailed",
                          reason=node.expr, on_fail=None,
                          duration_ms=_ms(node_started))
                    raise AssertFailed(node.name, node.expr)
                # The expression is pack-authored text, so it is safe to print whole —
                # it is the same string the graph author wrote in graph.yaml.
                event(_log, INFO if not passed else DEBUG, "node.assert", run_id=run_id,
                      node=node.name, type=node.type, passed=passed, expr=node.expr,
                      to=following, duration_ms=_ms(node_started))
                _checkpoint(store, pack, run_id, steps, node.name, following, state,
                            path, provenance, failures, attempts)
                node = _node(pack, following, "edge from %r" % node.name)
                continue

            raise DeadEnd("node %r has unknown type %r" % (node.name, node.type))
    except RunError as exc:
        # Every way a run can stop short passes through here on its way out — a spent
        # ladder with no `on_fail`, a dangling edge, an unbroken loop, a commit that
        # would have overwritten the caller's input. The exception was always raised;
        # until now it was never *recorded*, so a supervisor that swallowed it left
        # nothing behind saying which node the run died on.
        event(_log, ERROR, "run.error", run_id=run_id, pack=pack.name, node=node.name,
              step=steps, error=type(exc).__name__, reason=_safe_reason(exc),
              duration_ms=_ms(run_started))
        raise


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


def _ms(started):
    """Milliseconds since `started`, rounded to something a log line can hold."""
    return round((time.perf_counter() - started) * 1000.0, 1)


def _safe_reason(exc):
    """The half of a failure that may go into a default-level log line.

    `NodeFailed.reason` is the last rejection's full detail and may quote the generation
    verbatim — `verify.Rejected` keeps the two halves apart precisely so that the model
    never sees one of them, and a log an operator ships to a collector is the same kind
    of downstream. `feedback` is the sanitised half. When a NodeFailed was built without
    one, this says so rather than falling back to the detail: an empty log field is a
    smaller failure than a leaked one, and the detail is still there at DEBUG.

    Everything else here — a dangling edge, a step budget, an unevaluable expression —
    is jig's own words about the pack's own text, and goes out whole.
    """
    if isinstance(exc, NodeFailed):
        return exc.feedback or "rejected (detail at DEBUG)"
    return str(exc)


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
                failures, attempts, output=None):
    """Persist one completed node, if this run is being checkpointed at all."""
    if store is None:
        return
    extra = {"attempts": dict(attempts)} if _store_records_attempts(store) else {}
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
        # Hand over the pack itself, not just its name: the store records name AND
        # version, and `state.resume` needs both to catch a graph that moved on under
        # a run. Passing pack.name here threw the version away before the store saw it.
        pack=pack,
        **extra
    )


@lru_cache(maxsize=None)
def _save_accepts_attempts(store_type):
    """Whether this store's `save` can record per-node attempt counts.

    `store` is documented as anything with a `save(...)` method, so the counts are
    offered rather than imposed: a store written against the older signature would raise
    `TypeError` on a keyword it never asked for, and losing the checkpoint entirely is a
    worse outcome than losing one diagnostic field from it.
    """
    save = getattr(store_type, "save", None)
    if save is None:
        return False
    try:
        parameters = inspect.signature(save).parameters.values()
    except (TypeError, ValueError):
        return False
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (parameter.name == "attempts" and parameter.kind in kinds)
        for parameter in parameters
    )


def _store_records_attempts(store):
    return _save_accepts_attempts(type(store))


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
        attempts=dict(getattr(checkpoint, "attempts", None) or {}),
    )
