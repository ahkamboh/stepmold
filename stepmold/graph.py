"""The graph walker — stepmold's runtime.

This is the whole "the small model never plans" idea in one loop (docs/ARCHITECTURE.md §3): the
walker decides what happens next, the model only ever fills one node's slot. Nothing here
asks a model where to go; edges are data.

    state = inputs
    node  = entry
    loop:  execute node -> commit its output to state -> pick the next edge

Four node types. `generate` renders a prompt from state, generates under the node's
grammar, and commits the result — but only once `stepmold.verify` has accepted it, so a
rejected output never lands here. `tool` calls one of the actions the host registered
(`stepmold.tools`) and commits what it returns by exactly the same rules — no prompt, no
grammar and no retry ladder, because a tool is deterministic: same state in, same call
out, and a re-sample of a function is just the same call again. `assert` evaluates a
deterministic expression and either continues or diverts to `on_fail`. `end` stops and
returns.

A tool call is the one thing in a run that retrying cannot undo, so the walker writes it
down before it commits it. The moment a call returns, the node, the arguments it was
handed and what it gave back go into the checkpoint with the walk still standing on the
node; only once the node has been walked out of is the record cleared. A resumed run that
lands back on that node finds the record and replays it instead of calling. That is what
makes "the email is sent exactly once" true across a crash rather than merely likely —
committed state cannot say it, because state records what a call *returned* and never
that it *happened*. A tool declared `idempotent=True` skips the bookkeeping and is simply
called again, which is what declaring it means.

Failure routing is uniform, and that is deliberate: everything that stops a node
producing a verified output — a spent retry ladder, a prompt that cannot be rendered, an
expression that cannot be evaluated, a tool that raised or broke its own contract — takes
the node's declared `on_fail` edge, and only escapes as an exception when the node
declares none. A database being down therefore takes the same path a model failing does.
A node's failure edge is a promise in the pack; the walker keeps it whatever the node
failed on.

One failure is routed apart from the rest, because it is not the same claim. `Unsure` —
`stepmold.verify`'s outcome for a node whose independent samples disagreed — means the model
answered without the engine being able to believe it, which is a different thing from
the model not answering at all. A node may declare `on_unsure:` to send it somewhere a
person is waiting; if it does not, it falls back to `on_fail`, and with neither the run
aborts. Confidence is routed before it is acted on, which is why this landed before tool
nodes did rather than after.

The walk is also where a run's story gets written down (`stepmold.log`): `run.start`,
`node.ok` per node with the generations it spent and the milliseconds it took,
`edge.on_fail` when a node is diverted, `run.end` with the whole run's totals, and
`run.error` when it stops short. Names and counts only — the caller's data is reported as
a size and a digest, and a node failure carries the model-safe half of its reason (see
`_safe_reason`), with the full detail at DEBUG.
"""

import inspect
import json
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
from .tools import ToolContract, ToolError, ToolFailed
from .verify import run_node

try:
    from .verify import Unsure
except ImportError:  # pragma: no cover - until stepmold.verify grows the signal
    class Unsure(NodeFailed):
        """Independent samples of one node disagreed, so no answer is trustworthy.

        `stepmold.verify` owns this outcome; the walker owns where it goes. Until the two land
        together this stands in for it, so `on_unsure` routing is real and tested rather
        than written against a name and hoped for. It subclasses `NodeFailed` because a
        node nobody can believe did not produce a verified output either — a walker that
        somehow met one without knowing about `on_unsure` would still take `on_fail`,
        which is the conservative half of the rule.
        """


__all__ = ["Failure", "RunResult", "StateCollision", "ToolsNotAvailable",
           "ToolReplayMismatch", "Unsure", "replay", "run"]

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
      `stepmold.verify` makes about a bad generation: nothing lands until it is safe.

    This one lives with the walker rather than in `stepmold.errors` because it is a rule about
    committing, which is the walker's own job; it still subclasses `RunError`, so every
    caller that already handles run errors handles it.
    """


class ToolsNotAvailable(ToolError):
    """A pack reached a `tool` node and this run was handed no registry.

    Lives here rather than in `stepmold.tools` because it is a fact about the *call* — the
    host started a run without the thing the pack needs — and not about any tool. It is
    raised the moment the node is entered rather than swallowed into `on_fail`: a pack
    that cannot act at all is a wiring mistake in the caller, not a runtime condition the
    graph author wrote a rescue path for, and diverting it would quietly finish a
    workflow with the acting half missing.
    """


class ToolReplayMismatch(ToolError):
    """A recorded call is being resumed into state that would ask a different question.

    A replayed result is only the answer to the arguments it was computed from, so the
    walker compares them. They cannot differ on a faithful resume — the checkpoint the
    call was written into is the same one the state was restored from — so a difference
    means the record and the run have come apart. Neither choice left is safe on its own:
    calling again risks the second side effect the record exists to prevent, and
    committing the old result puts an answer to the wrong question into state. So the run
    stops and says which arguments moved.
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
        resume_from=None, tools=None):
    """Walk `pack` from its entry node until an `end` node, and return a `RunResult`.

    `store` is anything with a `save(...)` method (see `stepmold.state.Store`); when given, a
    checkpoint is written after every node that completes. `resume_from` is a checkpoint
    to continue from instead of starting at the entry node — that is how `state.resume`
    picks a dead run back up without re-executing what already succeeded.

    `tools` is a `stepmold.tools.ToolRegistry`: the set of actions this host is willing to let
    this pack take, and the only thing a `tool` node can reach. It is per-run and has no
    default — a pack that can act does so because the caller said so on this call, not
    because a module-level registry happened to exist. A pack with no tool nodes never
    looks at it.

    Exactly-once for those calls needs `store` as well as `tools`. Without a store there
    is nothing to resume from, so nothing to repeat; with one, the walker records each
    call before committing it and a resumed run replays rather than re-calls.
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
        # The calls this run made but never finished leaving a node for. Normally empty;
        # one entry means a process died between a tool returning and its node being
        # walked out of, and that entry is what stops the call being made twice. A store
        # older than the column hands back nothing, exactly as one that recorded no call
        # would — see `_store_records_tool_calls` for why that is warned about loudly.
        pending_calls = [dict(record)
                         for record in (getattr(resume_from, "tool_calls", None) or [])]
        steps = resume_from.step
        node = _node(pack, resume_from.next_node, "resume point of run %r" % run_id)
    else:
        state = dict(inputs or {})
        provenance = {}
        path = []
        failures = []
        attempts = {}
        pending_calls = []
        steps = 0
        node = _node(pack, pack.entry, "entry node")
    run_id = run_id or uuid.uuid4().hex
    # Said once per run rather than once per tool node: a store that cannot hold the
    # record is one fact about this run, and repeating it per node would bury it.
    warned_unrecorded = False

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
                            provenance, failures, attempts, output,
                            tool_calls=pending_calls)
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
                except Unsure as exc:
                    # Ahead of the clause below on purpose: `Unsure` is a kind of
                    # `NodeFailed`, and the whole point of it is that it does not go
                    # where a plain failure goes. The model answered; independent samples
                    # of it disagreed, so what the run has is an answer nobody can
                    # believe rather than no answer at all — which is a person's problem,
                    # not a rescue path's. `on_unsure` is where a pack says so. Falling
                    # back to `on_fail` is the conservative reading of a pack that never
                    # thought about it: somewhere declared is better than nowhere, and a
                    # node with neither aborts rather than committing on a coin flip.
                    target = getattr(node, "on_unsure", None) or node.on_fail
                    _log_failed(run_id, node, exc,
                                attempts.get(node.name, 0) - before, node_started)
                    if not target:
                        raise
                    failures.append(_failure(node, exc))
                    _checkpoint(store, pack, run_id, steps, node.name, target, state,
                                path, provenance, failures, attempts,
                                tool_calls=pending_calls)
                    edge = ("edge.on_unsure" if getattr(node, "on_unsure", None)
                            else "edge.on_fail")
                    event(_log, INFO, edge, run_id=run_id, node=node.name, to=target)
                    node = _node(pack, target, "unsure route of %r" % node.name)
                    continue
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
                    _log_failed(run_id, node, exc,
                                attempts.get(node.name, 0) - before, node_started)
                    if not node.on_fail:
                        raise
                    failures.append(_failure(node, exc))
                    _checkpoint(store, pack, run_id, steps, node.name, node.on_fail,
                                state, path, provenance, failures, attempts,
                                tool_calls=pending_calls)
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
                    # node instead of twice, and this is the innermost loop stepmold has.
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
                                path, provenance, failures, attempts,
                                tool_calls=pending_calls)
                    raise
                _checkpoint(store, pack, run_id, steps, node.name, following, state,
                            path, provenance, failures, attempts,
                            tool_calls=pending_calls)
                if _log.isEnabledFor(DEBUG):
                    event(_log, DEBUG, "edge.taken", run_id=run_id, node=node.name,
                          to=following)
                node = _node(pack, following, "edge from %r" % node.name)
                continue

            if node.type == "tool":
                tool = _tool_for(node, tools)
                # Read exactly as `Tool.invoke` will read them, because the arguments are
                # half of what makes a recorded call replayable: a result is only ever
                # the answer to the question it was asked. Missing reads are not filtered
                # out to be forgiving — `invoke` raises `ToolContract` on them, and this
                # dict is never used to call anything.
                arguments = {key: state[key] for key in tool.reads if key in state}
                recorded = _recorded_call(pending_calls, node.name)
                if recorded is not None:
                    # This node already ran, in a process that died before it could
                    # finish leaving it. Replaying is not an optimisation — it is the
                    # entire promise: calling again is the second email.
                    value = _replay_call(recorded, node, tool, arguments)
                else:
                    try:
                        value = tool.invoke(state, node.name)
                    except (ToolFailed, ToolContract) as exc:
                        # A database that is down and a retry ladder that is spent are
                        # the same fact to the graph: this node produced no output. So a
                        # tool failure takes the node's `on_fail` edge, and aborts the
                        # run when the node declares none, exactly as a generation
                        # failure does. `ToolContract` rides along with it for the same
                        # reason `MissingVariable` rides with `NodeFailed` above: a tool
                        # wired to state nobody wrote, or one that returned a shape the
                        # graph was not built around, is a node that cannot produce a
                        # verified output — not a new category of run-ending event.
                        # `ToolNotRegistered` deliberately does not: a pack naming an
                        # action the host never allowed is a fact about the pack, and an
                        # `on_fail` edge must not quietly finish a workflow around it.
                        _log_failed(run_id, node, exc, 0, node_started)
                        if not node.on_fail:
                            raise
                        failures.append(_failure(node, exc))
                        _checkpoint(store, pack, run_id, steps, node.name, node.on_fail,
                                    state, path, provenance, failures, attempts,
                                    tool_calls=pending_calls)
                        event(_log, INFO, "edge.on_fail", run_id=run_id, node=node.name,
                              to=node.on_fail)
                        node = _node(pack, node.on_fail, "on_fail of %r" % node.name)
                        continue
                    if not tool.idempotent:
                        pending_calls.append({
                            "node": node.name,
                            "tool": tool.name,
                            "args": arguments,
                            "result": value,
                        })
                        if store is not None and not warned_unrecorded \
                                and not _store_records_tool_calls(store):
                            warned_unrecorded = True
                            event(_log, WARNING, "tool.unrecorded", run_id=run_id,
                                  node=node.name, tool=tool.name,
                                  store=type(store).__name__,
                                  reason="store.save takes no tool_calls, so a resumed "
                                         "run cannot know this call already happened")
                        # Written down before the commit and before the edge, with
                        # `next_node` still this node because the walk has not left it.
                        # Everything after this line — the commit, the routing, the
                        # process — may die, and the resume lands here and replays.
                        _checkpoint(store, pack, run_id, steps, node.name, node.name,
                                    state, path, provenance, failures, attempts,
                                    tool_calls=pending_calls)
                # Whether this node's call is written down and still unsettled. It is
                # read after the commit, when `state` has already moved.
                pending = _recorded_call(pending_calls, node.name) is not None
                commit(node, value, state, provenance)
                if _log.isEnabledFor(INFO):
                    # `node.ok` rather than an event of its own: an operator asking "what
                    # did this run do, in order" wants one line per node whatever the
                    # node was. `attempts` is zero because a tool spends no generations,
                    # and that zero is the honest answer rather than a missing field.
                    event(_log, INFO, "node.ok", run_id=run_id, node=node.name,
                          type=node.type, attempts=0, tool=tool.name,
                          replayed=recorded is not None,
                          output=node.output or "merge",
                          duration_ms=_ms(node_started))
                try:
                    following = _next(pack, node, state)
                except DeadEnd:
                    # Only the routing failed, so an operator who repairs the edge
                    # resumes into this node — which must replay the call, not make it
                    # again. When the call is written down, the row that says so is
                    # already on disk from before the commit, and rewriting this step
                    # with the committed state is exactly the wrong move: a tool that
                    # reads a field it also writes would then be resumed against
                    # arguments its own output had moved, and refused as a mismatch. The
                    # pending row is the correct restart point, so it is left alone.
                    if not pending:
                        # Nothing was recorded — an idempotent tool — so the committed
                        # state is the only trace this node ran, exactly as for a
                        # `generate` node whose edge did not match.
                        _checkpoint(store, pack, run_id, steps, node.name, node.name,
                                    state, path, provenance, failures, attempts,
                                    tool_calls=pending_calls)
                    raise
                # The node has been left by an edge that exists, so the call is settled:
                # from here on the committed state is the record that it happened, and
                # resume will never re-enter this node. Clearing it is also what keeps a
                # loop honest — the next time round finds nothing and calls again, which
                # is what a tool inside a loop is for.
                pending_calls[:] = [record for record in pending_calls
                                    if record.get("node") != node.name]
                _checkpoint(store, pack, run_id, steps, node.name, following, state,
                            path, provenance, failures, attempts,
                            tool_calls=pending_calls)
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
                    # what `on_fail` is for. `stepmold.verify` already downgrades the identical
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
                            path, provenance, failures, attempts,
                            tool_calls=pending_calls)
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


def _log_failed(run_id, node, exc, spent, node_started):
    """The two lines a diverted node leaves behind, and the split between them.

    One event whatever failed, so `node.failed` is still the single thing to grep for.
    The default-level line carries only the half that may be shipped to a collector; the
    half that may quote a generation, or a tool's own exception text, is DEBUG-only. See
    `_safe_reason`.
    """
    if _log.isEnabledFor(WARNING):
        event(_log, WARNING, "node.failed", run_id=run_id, node=node.name,
              type=node.type, attempts=spent, error=type(exc).__name__,
              reason=_safe_reason(exc), on_fail=node.on_fail,
              duration_ms=_ms(node_started))
    if _log.isEnabledFor(DEBUG):
        # The unsafe half, once, at the only level allowed to hold it.
        event(_log, DEBUG, "node.failed.detail", run_id=run_id, node=node.name,
              detail=str(exc))


def _safe_reason(exc):
    """The half of a failure that may go into a default-level log line.

    `NodeFailed.reason` is the last rejection's full detail and may quote the generation
    verbatim — `verify.Rejected` keeps the two halves apart precisely so that the model
    never sees one of them, and a log an operator ships to a collector is the same kind
    of downstream. `feedback` is the sanitised half. When a NodeFailed was built without
    one, this says so rather than falling back to the detail: an empty log field is a
    smaller failure than a leaked one, and the detail is still there at DEBUG.

    `ToolFailed` gets the same treatment for a different reason. Its text is whatever the
    host's own function raised, and a tool raises about the thing it was given — "no such
    customer alice@example.com", the row it could not write. That is the caller's data
    arriving by a second door, so the default-level line names the tool and the type of
    what went wrong and stops there. `ToolContract` goes out whole: every value in it is
    a key name out of the pack or the tool's own declaration, never a value.

    Everything else here — a dangling edge, a step budget, an unevaluable expression —
    is stepmold's own words about the pack's own text, and goes out whole.
    """
    if isinstance(exc, NodeFailed):
        return exc.feedback or "rejected (detail at DEBUG)"
    if isinstance(exc, ToolFailed):
        return "tool %r raised %s (detail at DEBUG)" % (
            exc.tool, type(exc.cause).__name__
        )
    return str(exc)


def _failure(node, exc):
    """Record whichever way this node failed, for `RunResult.failures`.

    `getattr` rather than more isinstance branches: a failure that carries its own node
    and cost says so — `NodeFailed` and anything built on it — and one that does not is
    attributed to the node the walker was standing on, having spent nothing. That covers
    an unrenderable prompt, a tool that raised, and any outcome `stepmold.verify` grows later
    without this function needing to hear about it.
    """
    if isinstance(exc, NodeFailed):
        return Failure(node=exc.node, reason=exc.reason, attempts=exc.attempts)
    # A tool that raised and a prompt that never rendered both cost no generation.
    return Failure(
        node=getattr(exc, "node", None) or node.name,
        reason=str(exc),
        attempts=getattr(exc, "attempts", 0) or 0,
    )


# ----------------------------------------------------------------------------- tools


def _tool_for(node, tools):
    """The registered action this node names, or a refusal that says what to change."""
    if tools is None:
        raise ToolsNotAvailable(
            "node %r is a tool node, and this run was given no tools: this pack needs "
            "tools; pass tools= to run()" % node.name
        )
    name = getattr(node, "tool", None)
    if not name:
        raise ToolsNotAvailable(
            "node %r is type 'tool' but names no tool to call — give it a `tool:` key "
            "naming one of the actions the host registered (%s)"
            % (node.name, ", ".join(getattr(tools, "names", ())) or "none")
        )
    # A name the host never registered raises `ToolNotRegistered` out of here, and is not
    # caught anywhere in the walk: see the tool branch for why that is not an `on_fail`.
    return tools.get(name, node.name)


def _recorded_call(pending_calls, node_name):
    """The unsettled call this node already made, if a previous process made it.

    Last first, because a record is only ever added by the node the walk is standing on
    and only ever removed once that node is left — so at most one is ever pending, and
    the newest is the one this node is resuming into.
    """
    for record in reversed(pending_calls):
        if record.get("node") == node_name:
            return record
    return None


def _replay_call(record, node, tool, arguments):
    """The result a previous process already got, instead of calling again."""
    moved = _moved_args(record.get("args"), arguments)
    if moved:
        raise ToolReplayMismatch(
            "run resumed into node %r holding a call to %r that already happened, but "
            "state no longer matches the arguments it was made with (%s differ). The "
            "recorded result answers a question this run is no longer asking, and "
            "calling again would repeat a side effect that already took place — so the "
            "run stops instead of choosing between them. Read the run's checkpoints "
            "before resuming it again."
            % (node.name, tool.name, ", ".join(moved))
        )
    result = record.get("result")
    return dict(result) if isinstance(result, dict) else {}


def _canonical(value):
    """A comparable form of an argument set.

    Canonical JSON rather than `==`, for the reason `state._dict_delta` gives: `True == 1`
    in Python, so two argument sets that a tool would treat as different compare equal.
    A value that is not JSON never came out of a checkpoint, so `repr` is the honest
    fallback rather than a reason to raise.
    """
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _moved_args(recorded, current):
    """The argument names whose value changed, appeared or vanished. Names only.

    Names only, because this ends up in an exception message and an argument's value is
    the caller's data. Which argument moved is what an operator needs to find the cause;
    what it moved to is in the checkpoint, where it is already allowed to be.
    """
    if not isinstance(recorded, dict):
        # A record with no argument set at all cannot be matched against anything, and
        # saying so beats silently accepting it.
        return ["the recorded arguments (none were written down)"]
    names = set(recorded) | set(current)
    return sorted(
        name for name in names
        if name not in recorded or name not in current
        or _canonical(recorded[name]) != _canonical(current[name])
    )


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
                failures, attempts, output=None, tool_calls=None):
    """Persist one completed node, if this run is being checkpointed at all."""
    if store is None:
        return
    extra = {"attempts": dict(attempts)} if _store_records_attempts(store) else {}
    if _store_records_tool_calls(store):
        extra["tool_calls"] = [dict(record) for record in (tool_calls or [])]
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
def _save_accepts(store_type, keyword):
    """Whether this store's `save` will take `keyword`.

    `store` is documented as anything with a `save(...)` method, so anything the walker
    learned to record after that contract was written is offered rather than imposed: a
    store built against the older signature would raise `TypeError` on a keyword it never
    asked for, and losing the checkpoint entirely is a worse outcome than losing one
    field from it.
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
        or (parameter.name == keyword and parameter.kind in kinds)
        for parameter in parameters
    )


def _store_records_attempts(store):
    return _save_accepts(type(store), "attempts")


def _store_records_tool_calls(store):
    """Whether this store can hold the record that stops a tool firing twice.

    Losing a diagnostic to an old store is a shrug; losing this is a promise broken, so
    the walker says so out loud (`tool.unrecorded`) instead of degrading in silence. It
    still does not refuse the run: a run with no store at all has the same exposure and
    is perfectly legal, and a host that has chosen its own store has chosen this too.
    """
    return _save_accepts(type(store), "tool_calls")


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
