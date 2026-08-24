"""Checkpoints: a run that dies at node 17 resumes at node 17.

Long workflows are the whole point of jig, and a workflow that restarts from zero on any
failure is a workflow you cannot run on real volume (docs/PLAN.md §3). So state is written
to SQLite after every node that completes, keyed by run id, and `resume` picks the walk up
at the node the crash interrupted.

SQLite because it is in the standard library, it is a real database with real durability,
and a client can open the file with tools they already have. PLAN.md §7 lists Postgres as
the option for later; nothing here is SQLite-specific beyond `Store`.

`Store` knows nothing about walking a graph, and the walker knows nothing about SQL — it
just calls `store.save(...)` with keywords. That is what keeps these two modules from
importing each other.

Two contracts hold this together, and both are enforced rather than hoped for:

* **A checkpoint round-trips.** What `latest` returns is what `save` was handed, type for
  type. JSON does not preserve every Python value — a tuple comes back a list, an int
  dict key comes back a string, a dict mixing key types cannot be sorted at all — so a
  value that would change shape is refused at the door instead of corrupting the run it
  is supposed to rescue. Only strict JSON is written: NaN and Infinity are Python's
  extensions to the format, and a file carrying them is unreadable by any conforming
  reader, including the other-language runtimes PLAN.md §7 leaves room for.
* **A run resumes under the pack that started it.** The checkpoint records which pack
  wrote it, and `resume` refuses a pack that disagrees. Resuming under a different graph
  skips nodes that were inserted since and trusts nodes that were rewritten, which turns
  a crash into wrong output rather than a late one.

Concurrency
-----------

A deployed jig is a pool of workers over one store file, and every operation here is
written for that rather than for the single-threaded in-process case it grew up in.

* **A `Store` is safe to share between threads and between processes.** The connection is
  opened with `check_same_thread=False` and every statement runs under one lock, so the
  obvious deployment — one `Store`, a thread pool — works instead of raising
  `sqlite3.ProgrammingError` out of the first `save`. Across processes the file locking
  does the same job; the journal is WAL so a reader is never shut out by a writer.
* **The first checkpoint of a run claims its run id, atomically.** Checking
  `latest(run_id)` and then walking is a read-then-write: two runs that start together
  both see an empty store and weld their chains into one. The claim is a row in `runs`
  with `run_id` as the primary key, taken inside the same transaction as the checkpoint,
  so the loser of a race is refused with `RunIdInUse` and writes nothing.
* **Resuming an unfinished run takes a lease.** Two supervisors retrying the same resume
  would otherwise both replay from the same checkpoint and both execute every remaining
  node — twice the charges, twice the emails. `Store.lease` is a claim with an expiry, so
  a resumer that dies holding one does not wedge the run forever.
* **Contention arrives as a `JigError`.** sqlite's busy handler retries for `timeout`
  seconds; past that the raw `sqlite3.OperationalError` is wrapped in `StoreBusy`, which
  names the run and the node, so `except JigError` around a run keeps working.

Size and retention
------------------

A checkpoint stores what changed at that node, not the whole of state, so an N-node run
costs bytes linear in N instead of quadratic. A full snapshot is written again whenever
the deltas since the last one would cost more to read than the snapshot would, which
bounds rebuilding a state of size S at roughly 2S bytes read — reconstruction is an
implementation detail, `latest` and `history` still hand back whole `Checkpoint` objects.

Nothing is deleted on jig's own initiative: a checkpoint chain is an audit trail, and a
store that quietly forgot last month's runs would be worse than one that grows. Retention
is the operator's call, and `prune` plus `vacuum` are how they make it — drop finished
runs older than a date or beyond a count, then give the pages back to the filesystem.
"""

import contextlib
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .errors import RunError, RunIdInUse, UnknownRun

__all__ = [
    "Checkpoint",
    "CheckpointMismatch",
    "ResumeInProgress",
    "Store",
    "StoreBusy",
    "resume",
]


class CheckpointMismatch(RunError):
    """The checkpoint being resumed was written by a different pack than the one given.

    Lives here rather than in `jig.errors` for the same reason the pack-loading errors
    live in `jig.pack`: it is raised before the walk restarts, by the store's own
    bookkeeping, and nothing in the walker can produce it.
    """


class StoreBusy(RunError):
    """The store stayed locked for longer than its busy timeout.

    Raised instead of the bare `sqlite3.OperationalError`, which is not a `JigError` and
    names neither the run nor the node — so a caller wrapping its runs in
    `except JigError` was shown a traceback about a database it never opened.
    """


class ResumeInProgress(RunError):
    """Another resumer holds this run's lease.

    Two supervisors resuming one unfinished run both replay from the same checkpoint and
    both execute every remaining node. The second one is refused rather than allowed to
    do the work twice.
    """


SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id      TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    node        TEXT    NOT NULL,
    next_node   TEXT,
    state       TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    provenance  TEXT    NOT NULL,
    failures    TEXT    NOT NULL,
    output      TEXT,
    pack        TEXT,
    pack_version TEXT,
    state_kind  TEXT,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (run_id, step)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    owner         TEXT NOT NULL,
    claimed_at    TEXT NOT NULL,
    lease_owner   TEXT,
    lease_expires REAL
);
"""

# Columns added after the first release. A store file written by an older jig has the
# table already, so CREATE TABLE IF NOT EXISTS silently leaves it short a column and
# every read of that column fails. Adding them on open keeps old files readable.
_ADDED_COLUMNS = (("pack_version", "TEXT"), ("state_kind", "TEXT"))

#: The walker's first checkpoint of a fresh run. `graph.run` starts its step counter at
#: zero and increments before the first node, so step 1 is the moment a run id is taken
#: and the only write that has to fight for it.
_FIRST_STEP = 1

#: Seconds sqlite may spend retrying a locked database before `save` gives up. Higher
#: than sqlite3's own 5s default because a jig store is written by a fleet and may sit on
#: a slow or networked filesystem; `Store(path, timeout=...)` is the knob.
DEFAULT_TIMEOUT = 30.0

#: How long a resume lease is honoured before another resumer may take it. A resumer that
#: exits normally releases it immediately; this only covers one that died holding it, so
#: it is long enough that a slow run is never overtaken by its own retry.
DEFAULT_LEASE = 300.0


@dataclass
class Checkpoint:
    """The run, frozen after one node completed."""

    run_id: str
    step: int
    node: str
    next_node: Optional[str]
    state: Dict[str, Any] = field(default_factory=dict)
    path: List[str] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)
    failures: List[dict] = field(default_factory=list)
    output: Optional[Dict[str, Any]] = None
    pack: Optional[str] = None
    pack_version: Optional[str] = None
    created_at: Optional[str] = None

    @property
    def finished(self):
        """True when this checkpoint is the end of the run, not a pause in it."""
        return self.next_node is None


class Store:
    """Run checkpoints in a SQLite file (or `:memory:` for tests).

    Safe to share between threads and between processes; see the module docstring for
    what that costs and what it buys.
    """

    def __init__(self, path=":memory:", timeout=DEFAULT_TIMEOUT):
        self.path = path
        self.timeout = float(timeout)
        #: Identifies this Store among everything else touching the file. A run id is
        #: claimed by an owner, so re-saving a step is allowed to the writer that already
        #: holds the run and refused to anyone else.
        self._owner = uuid.uuid4().hex
        self._lock = threading.RLock()
        #: run_id -> (step, frame) for the last checkpoint this Store wrote, so the usual
        #: case — a walk saving step after step — computes its delta without re-reading
        #: the chain. Keyed by step, so a stale entry is ignored rather than trusted.
        self._recent = {}
        if path != ":memory:":
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                # exist_ok, because a fleet cold-starting on a new store directory has
                # every worker but one arrive after it exists. `isdir` then `makedirs` is
                # a read-then-write and loses that race.
                os.makedirs(directory, exist_ok=True)
        self._connection = sqlite3.connect(
            path,
            timeout=self.timeout,
            # The store is shared; one lock around every statement is what makes that
            # safe, and it is cheaper than a connection per thread on a file this small.
            check_same_thread=False,
            # Autocommit, so a claim and its checkpoint can be one explicit
            # BEGIN IMMEDIATE ... COMMIT rather than whatever sqlite3 decides to open.
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = %d" % int(self.timeout * 1000))
        if path != ":memory:":
            # WAL so a reader — `history` over a large store, a backup — never blocks the
            # writer, and the writer never blocks it. Meaningless for `:memory:`.
            self._enable_wal()
        self._connection.executescript(SCHEMA)
        self._migrate()

    def _enable_wal(self):
        """Switch the file to WAL, tolerating another opener holding it right now.

        `PRAGMA journal_mode=WAL` wants the file to itself for a moment and answers
        SQLITE_BUSY without ever consulting the busy handler, so a fleet cold-starting on
        one store has openers that collide over it. The journal mode belongs to the file
        rather than to this connection: whoever wins sets it for everyone, and a loser
        left on the rollback journal is slower, not wrong. Refusing to open a perfectly
        good store over that would be the worse trade, so this retries briefly and then
        carries on.
        """
        deadline = time.monotonic() + min(self.timeout, 1.0)
        while True:
            try:
                self._connection.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.005)

    def _migrate(self):
        """Add columns a store file written by an older jig is missing."""
        for column, declared_type in _ADDED_COLUMNS:
            _add_column(self._connection, column, declared_type)

    def save(self, run_id, step, node, next_node, state, path=None, provenance=None,
             failures=None, output=None, pack=None, pack_version=None):
        """Record one completed node. Re-saving the same step replaces it.

        `pack` is the pack's identity: either its name, or the pack itself, in which case
        its version is recorded too and `resume` can catch a graph that moved on under a
        run. Every field is checked before anything is written, so a value JSON would
        change on the way back raises here and leaves no half-written checkpoint.

        The first checkpoint of a run claims its run id in the same transaction that
        writes the checkpoint; a second run under a live id raises `RunIdInUse` here
        rather than quietly welding itself onto the first one's chain.

        Steps are assumed to arrive in order within a run — that is what the walker does,
        and what lets a checkpoint record only the state its node changed. Rewriting a
        step that already has successors is the one thing that would leave those
        successors describing a state that no longer precedes them.
        """
        name, version = _identity(pack, pack_version)
        # Validate and serialise before opening a transaction: a value that cannot be
        # checkpointed must leave the store exactly as it was, lock and all.
        snapshot = (
            _dump(state, "state"),
            _dump(list(path or []), "path"),
            _dump(dict(provenance or {}), "provenance"),
        )
        tail = (
            _dump(list(failures or []), "failures"),
            None if output is None else _dump(output, "output"),
            name,
            version,
        )
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._lock, self._transaction(run_id, step, node):
            self._claim(run_id, step)
            kind, blobs = self._encode(run_id, step, snapshot)
            self._connection.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(run_id, step, node, next_node, state, path, provenance, failures, "
                " output, pack, pack_version, state_kind, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, step, node, next_node) + blobs + tail + (kind, created_at),
            )
        # Only once the transaction committed: a cache entry for a row that was rolled
        # back would describe a chain the file does not have.
        self._recent[run_id] = (step, _parse(snapshot))

    def latest(self, run_id):
        """The most recent checkpoint for `run_id`, or None."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return _to_checkpoint(row, self._read_frame(run_id, row["step"]))

    def history(self, run_id):
        """Every checkpoint for `run_id`, oldest first — the run's audit trail."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step", (run_id,)
            ).fetchall()
        checkpoints = []
        frame = _EMPTY_FRAME
        for row in rows:
            frame = _apply(frame, row)
            checkpoints.append(_to_checkpoint(row, frame))
        return checkpoints

    def runs(self):
        """Every run id the store knows about."""
        with self._lock:
            cursor = self._connection.execute(
                "SELECT DISTINCT run_id FROM checkpoints"
            )
            return [row["run_id"] for row in cursor.fetchall()]

    def delete(self, run_id):
        """Forget a run: its checkpoints and its claim on the id."""
        with self._lock, self._transaction(run_id, None, None):
            self._connection.execute(
                "DELETE FROM checkpoints WHERE run_id = ?", (run_id,)
            )
            self._connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._recent.pop(run_id, None)

    # ------------------------------------------------------------- retention

    def prune(self, keep_last=None, before=None):
        """Delete finished runs, and return the ids deleted.

        Retention is the operator's policy, not jig's, so this does nothing unless it is
        called and it never touches a run that has not finished — an unfinished chain is
        the only copy of work someone still intends to resume.

        `keep_last` keeps the N most recently created finished runs and drops the rest.
        `before` drops finished runs whose last checkpoint predates it, given as an ISO
        timestamp string or a datetime. Both may be given; a run is dropped if either
        rule says so.
        """
        if keep_last is None and before is None:
            return []
        if hasattr(before, "isoformat"):
            before = before.isoformat(timespec="seconds")
        with self._lock:
            # The last checkpoint of each run: finished when its next_node is NULL.
            rows = self._connection.execute(
                "SELECT c.run_id AS run_id, c.next_node AS next_node,"
                "       c.created_at AS created_at FROM checkpoints c"
                " JOIN (SELECT run_id, MAX(step) AS step FROM checkpoints"
                "       GROUP BY run_id) last"
                " ON c.run_id = last.run_id AND c.step = last.step"
                " ORDER BY c.created_at, c.run_id"
            ).fetchall()
        finished = [row for row in rows if row["next_node"] is None]
        doomed = set()
        if before is not None:
            doomed.update(
                row["run_id"] for row in finished if row["created_at"] < before
            )
        if keep_last is not None and len(finished) > keep_last:
            keep = len(finished) - max(int(keep_last), 0)
            doomed.update(row["run_id"] for row in finished[:keep])
        for run_id in sorted(doomed):
            self.delete(run_id)
        return sorted(doomed)

    def vacuum(self):
        """Give freed pages back to the filesystem.

        Deleting runs does not shrink the file — sqlite keeps the pages for reuse — so an
        operator who has just pruned a year of runs and wants the disk back needs this.
        It rewrites the whole database, so it is theirs to schedule, not jig's to run.
        """
        with self._lock:
            self._connection.execute("VACUUM")
            # In WAL mode the rebuilt database lands in the write-ahead log, so the file
            # an operator is watching does not shrink until the log is folded back in.
            if self.path != ":memory:":
                self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ------------------------------------------------------------ the resume lease

    @contextlib.contextmanager
    def lease(self, run_id, seconds=DEFAULT_LEASE):
        """Hold an exclusive claim on `run_id` for the body of the `with`.

        Raises `ResumeInProgress` if someone else holds it. The expiry is the answer to a
        holder that died: nothing else can release the lease on its behalf, so the lease
        releases itself.

        The expiry is wall-clock rather than monotonic because the workers contending for
        it are separate processes, and a monotonic clock means nothing outside the one
        that read it.
        """
        token = uuid.uuid4().hex
        deadline = time.time() + float(seconds)
        with self._lock, self._transaction(run_id, None, None):
            self._connection.execute(
                "INSERT OR IGNORE INTO runs (run_id, owner, claimed_at)"
                " VALUES (?,?,?)",
                (run_id, self._owner, _now()),
            )
            cursor = self._connection.execute(
                "UPDATE runs SET lease_owner = ?, lease_expires = ?"
                " WHERE run_id = ?"
                "   AND (lease_owner IS NULL OR lease_expires IS NULL"
                "        OR lease_expires < ?)",
                (token, deadline, run_id, time.time()),
            )
            if cursor.rowcount == 0:
                raise ResumeInProgress(
                    "run %r is already being resumed by another worker — a second "
                    "resume would execute every remaining node a second time" % run_id
                )
        try:
            yield token
        finally:
            with self._lock, self._transaction(run_id, None, None):
                self._connection.execute(
                    "UPDATE runs SET lease_owner = NULL, lease_expires = NULL"
                    " WHERE run_id = ? AND lease_owner = ?",
                    (run_id, token),
                )

    # ------------------------------------------------------------ internals

    @contextlib.contextmanager
    def _transaction(self, run_id, step, node):
        """One BEGIN IMMEDIATE, and contention reported as a jig error.

        IMMEDIATE rather than the default deferred begin: the write lock is taken before
        the claim is read, so two workers cannot both read "unclaimed" and both write.
        """
        try:
            self._connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            _raise_busy(exc, run_id, step, node)
        try:
            yield
        except sqlite3.OperationalError as exc:
            self._rollback()
            _raise_busy(exc, run_id, step, node)
        except BaseException:
            self._rollback()
            raise
        try:
            self._connection.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            self._rollback()
            _raise_busy(exc, run_id, step, node)

    def _rollback(self):
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - nothing to roll back
            pass

    def _claim(self, run_id, step):
        """Take the run id, or confirm this Store already holds it.

        The claim is an INSERT against a primary key inside the checkpoint's own
        transaction, so it is decided by sqlite rather than by a read the caller performs
        and then acts on. Only the first step fights for the id: a resume continues a run
        that is already claimed, and a caller saving rows by hand may start anywhere.
        """
        if step != _FIRST_STEP:
            self._connection.execute(
                "INSERT OR IGNORE INTO runs (run_id, owner, claimed_at) VALUES (?,?,?)",
                (run_id, self._owner, _now()),
            )
            return
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO runs (run_id, owner, claimed_at) VALUES (?,?,?)",
            (run_id, self._owner, _now()),
        )
        if cursor.rowcount == 1:
            return
        holder = self._connection.execute(
            "SELECT owner FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if holder is not None and holder["owner"] != self._owner:
            raise RunIdInUse(
                "run id %r is already claimed in this store by another run; "
                "resume it, delete it, or choose another id" % run_id
            )

    def _encode(self, run_id, step, snapshot):
        """('full'|'delta', (state, path, provenance)) as this row should be written.

        A checkpoint that repeated the whole of state cost an N-node run O(N^2) bytes,
        which is the one shape jig's own pitch — long workflows — guarantees. `path` and
        `provenance` grow the same way and are encoded the same way, because moving the
        quadratic term from one column to the next is not a fix.

        Snapshots are what keep a read cheap, and the rule for taking one is measured in
        bytes rather than in steps: write a full row as soon as rebuilding it would read
        more than twice what it holds. That size is the floor — it is how much there is
        to return — so this bounds a read at twice it, while leaving a run whose state
        only grows with the single snapshot it started with. Counting steps instead would
        take snapshots a run does not need and put the quadratic term straight back.
        """
        base_step = self._previous_step(run_id, step)
        if base_step is None:
            return "full", snapshot
        last_full = self._last_full_step(run_id, base_step)
        if last_full is None:
            return "full", snapshot
        cached = self._recent.get(run_id)
        base = (
            cached[1] if cached is not None and cached[0] == base_step
            else self._read_frame(run_id, base_step)
        )
        current = _parse(snapshot)
        delta = (
            _dumps(_dict_delta(base[0], current[0])),
            _dumps(_path_delta(base[1], current[1])),
            _dumps(_dict_delta(base[2], current[2])),
        )
        full_bytes = sum(len(blob) for blob in snapshot)
        delta_bytes = sum(len(blob) for blob in delta)
        # A delta at least as big as the snapshot buys nothing and costs a hop on every
        # read — which is what a node that rewrites the whole of state produces.
        if delta_bytes >= full_bytes:
            return "full", snapshot
        # Everything a read would have to walk to rebuild this step: the snapshot it
        # starts from, plus every delta between that and here.
        carried = self._connection.execute(
            "SELECT COALESCE(SUM(LENGTH(state) + LENGTH(path) + LENGTH(provenance)), 0)"
            " AS bytes FROM checkpoints WHERE run_id = ? AND step >= ? AND step <= ?",
            (run_id, last_full, base_step),
        ).fetchone()["bytes"]
        if carried + delta_bytes >= 2 * full_bytes:
            return "full", snapshot
        return "delta", delta

    def _previous_step(self, run_id, step):
        row = self._connection.execute(
            "SELECT MAX(step) AS step FROM checkpoints WHERE run_id = ? AND step < ?",
            (run_id, step),
        ).fetchone()
        return row["step"]

    def _last_full_step(self, run_id, step):
        """The newest snapshot at or before `step`. NULL state_kind is a pre-delta row."""
        return self._connection.execute(
            "SELECT MAX(step) AS step FROM checkpoints WHERE run_id = ? AND step <= ?"
            "  AND (state_kind IS NULL OR state_kind = 'full')",
            (run_id, step),
        ).fetchone()["step"]

    def _read_frame(self, run_id, step):
        """Rebuild (state, path, provenance) for a step, from its snapshot forward."""
        last_full = self._last_full_step(run_id, step)
        start = last_full if last_full is not None else 0
        rows = self._connection.execute(
            "SELECT state, path, provenance, state_kind FROM checkpoints"
            " WHERE run_id = ? AND step >= ? AND step <= ? ORDER BY step",
            (run_id, start, step),
        ).fetchall()
        frame = _EMPTY_FRAME
        for row in rows:
            frame = _apply(frame, row)
        return frame

    def close(self):
        with self._lock:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _raise_busy(exc, run_id, step, node):
    """Re-raise a lock timeout as `StoreBusy`; leave every other sqlite error alone.

    A schema that is wrong, a disk that is full and a database that is merely contended
    are three different operator problems, and only the last one is jig's to rename.
    """
    text = str(exc).lower()
    if "locked" not in text and "busy" not in text:
        raise exc
    where = "run %r" % run_id
    if step is not None:
        where += " at step %d" % step
    if node is not None:
        where += " (node %r)" % node
    raise StoreBusy(
        "the store stayed locked longer than the busy timeout while writing %s: %s "
        "— raise Store(path, timeout=...) or reduce the writers on this file"
        % (where, exc)
    ) from exc


def _add_column(connection, column, declared_type):
    """ALTER TABLE ... ADD COLUMN, tolerating the column already being there.

    Reading `PRAGMA table_info` and then deciding is a read-then-write: two workers
    opening one legacy store file both see the column missing and both run the ALTER, and
    the second is answered with `duplicate column name` out of a constructor. Asking
    sqlite and accepting its refusal is the same work done atomically — a duplicate here
    means the other worker did it, which is the outcome either way.
    """
    try:
        connection.execute(
            "ALTER TABLE checkpoints ADD COLUMN %s %s" % (column, declared_type)
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


#: What a checkpoint carries that grows along the walk, in the order the columns hold it.
#: A run starts from nothing, which is what a delta chain with no snapshot in front of it
#: would be read against.
_EMPTY_FRAME = ({}, [], {})


def _dumps(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _parse(snapshot):
    """(state, path, provenance) from the three JSON blobs of a full row."""
    return tuple(json.loads(blob) for blob in snapshot)


def _dict_delta(base, new):
    """What changed between two dicts: keys set to a new value, and keys removed."""
    changed = {
        key: value
        for key, value in new.items()
        # Canonical JSON rather than `!=`, because `True == 1` in Python and a value that
        # went from true to 1 would otherwise be recorded as unchanged and come back as
        # the wrong one of the two — which is the round-trip contract, broken quietly.
        if key not in base or _dumps(base[key]) != _dumps(value)
    }
    return {"set": changed, "del": sorted(key for key in base if key not in new)}


def _path_delta(base, new):
    """How a path grew: how much of the old one still stands, and what was appended.

    A walk only ever appends, so `keep` is normally the whole of the previous path — but
    a path that diverges is recorded correctly rather than assumed away.
    """
    keep = 0
    for before, after in zip(base, new):
        if _dumps(before) != _dumps(after):
            break
        keep += 1
    return {"keep": keep, "add": list(new[keep:])}


def _apply(frame, row):
    """The frame after one row: a snapshot replaces it, a delta edits it."""
    if row["state_kind"] != "delta":
        return (
            json.loads(row["state"]),
            json.loads(row["path"]),
            json.loads(row["provenance"]),
        )
    return (
        _apply_dict(frame[0], json.loads(row["state"])),
        _apply_path(frame[1], json.loads(row["path"])),
        _apply_dict(frame[2], json.loads(row["provenance"])),
    )


def _apply_dict(base, delta):
    result = dict(base)
    for key in delta.get("del", ()):
        result.pop(key, None)
    result.update(delta.get("set", {}))
    return result


def _apply_path(base, delta):
    return list(base[:delta.get("keep", 0)]) + list(delta.get("add", ()))


def resume(pack, model, run_id, store, **kwargs):
    """Continue `run_id` from its last checkpoint, under the pack that started it.

    A run that already finished is returned as-is, without calling the model — resuming
    is idempotent, so a supervisor can retry a resume without paying for it twice.

    `pack` is checked against the checkpoint first, finished run or not: replaying one
    graph's output as another's is the same lie whether the walk continues or not.

    An unfinished run is resumed under a lease, so two supervisors retrying the same
    resume do not both execute every remaining node. The second one is refused with
    `ResumeInProgress` rather than allowed to charge the card twice.
    """
    from .graph import run as run_pack, replay  # local: graph imports nothing from here

    checkpoint = store.latest(run_id)
    if checkpoint is None:
        raise UnknownRun("no checkpoint found for run %r" % run_id)
    _check_same_pack(pack, checkpoint)
    if checkpoint.finished:
        return replay(checkpoint)

    lease = getattr(store, "lease", None)
    if lease is None:
        # A store that predates the lease — anything with save/latest is a store, per
        # graph.run's contract. Nothing to serialise against, so resume as before.
        return run_pack(pack, model, run_id=run_id, store=store,
                        resume_from=checkpoint, **kwargs)
    with lease(run_id):
        # Re-read under the lease: the run may have been finished by whoever held it
        # while this caller was waiting for it.
        checkpoint = store.latest(run_id)
        if checkpoint is None:  # pragma: no cover - deleted mid-resume
            raise UnknownRun("no checkpoint found for run %r" % run_id)
        _check_same_pack(pack, checkpoint)
        if checkpoint.finished:
            return replay(checkpoint)
        return run_pack(pack, model, run_id=run_id, store=store,
                        resume_from=checkpoint, **kwargs)


def _check_same_pack(pack, checkpoint):
    """Refuse to resume `checkpoint` under a pack that is not the one that wrote it.

    Identity first, then shape. The recorded identity is whatever the writer supplied —
    a checkpoint written before these columns existed carries no name and no version, so
    those comparisons are skipped rather than failed, and the shape check below is all
    that stands between the run and a graph it never walked.
    """
    described = _describe(checkpoint)
    if checkpoint.pack is not None and checkpoint.pack != pack.name:
        raise CheckpointMismatch(
            "run %r was checkpointed under %s, but resume was handed pack %r version %s "
            "— resume it under the pack that started it, or start a new run"
            % (checkpoint.run_id, described, pack.name, pack.version)
        )
    recorded_version = checkpoint.pack_version
    if recorded_version is not None and recorded_version != str(pack.version):
        raise CheckpointMismatch(
            "run %r was checkpointed under %s, but resume was handed version %s of %r "
            "— a version bump means the graph moved on, so the checkpoint no longer "
            "describes a walk through it"
            % (checkpoint.run_id, described, pack.version, pack.name)
        )

    # Shape. A node the run already walked, or the one it stopped in front of, must still
    # exist. This catches a renamed or deleted node even when the version was never
    # recorded — it cannot catch a node whose prompt changed, which is what the version
    # is for.
    walked = list(checkpoint.path)
    if checkpoint.next_node is not None:
        walked.append(checkpoint.next_node)
    for name in walked:
        if name not in pack.nodes:
            raise CheckpointMismatch(
                "run %r was checkpointed under %s and needs node %r, which pack %r "
                "version %s does not define — the graph changed under the run"
                % (checkpoint.run_id, described, name, pack.name, pack.version)
            )


def _describe(checkpoint):
    """How a checkpoint names the pack that wrote it, including when it does not."""
    if checkpoint.pack is None:
        return "a pack that did not record its name"
    if checkpoint.pack_version is None:
        return "pack %r (no version recorded)" % checkpoint.pack
    return "pack %r version %s" % (checkpoint.pack, checkpoint.pack_version)


def _identity(pack, pack_version=None):
    """(name, version) for the store, from either a pack name or the pack itself."""
    name = getattr(pack, "name", pack)
    version = pack_version if pack_version is not None else getattr(pack, "version", None)
    return (
        None if name is None else str(name),
        None if version is None else str(version),
    )


# ------------------------------------------------------- the round-trip contract

# What a checkpoint may hold: exactly the values JSON hands back unchanged. Anything
# else is refused by name, because a checkpoint that quietly alters the run it is
# restoring is worse than no checkpoint at all.
_SCALARS = (str, int, float, bool)


def _dump(value, where):
    _check(value, where, set())
    # allow_nan=False is the belt to _check's braces: nothing reaches the file that a
    # conforming JSON reader would choke on.
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _check(value, where, open_containers):
    """Raise if `value` would not survive a JSON round trip unchanged.

    TypeError for a value JSON has no representation for, ValueError for one it can
    represent only with a non-standard extension — the same split json.dumps uses, so a
    caller already catching one keeps catching it.
    """
    if value is None or isinstance(value, _SCALARS):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                "cannot checkpoint %s: %r is not a JSON number — strict JSON has no "
                "NaN or Infinity, and a store file carrying one cannot be read back"
                % (where, value)
            )
        return

    if id(value) in open_containers:
        raise ValueError(
            "cannot checkpoint %s: it contains itself, so there is nothing finite to "
            "write" % where
        )

    if isinstance(value, list):
        open_containers.add(id(value))
        for index, item in enumerate(value):
            _check(item, "%s[%d]" % (where, index), open_containers)
        open_containers.discard(id(value))
        return

    if isinstance(value, dict):
        open_containers.add(id(value))
        for key, item in value.items():
            # Not `isinstance(key, str)`: a bool key is not a str, and json would write
            # it as "true" and hand back a string.
            if type(key) is not str:
                raise TypeError(
                    "cannot checkpoint %s: the key %r is a %s, and JSON object keys are "
                    "strings — it would come back as %r"
                    % (where, key, type(key).__name__, str(key))
                )
            _check(item, "%s.%s" % (where, key), open_containers)
        open_containers.discard(id(value))
        return

    if isinstance(value, tuple):
        raise TypeError(
            "cannot checkpoint %s: a tuple comes back from JSON as a list — commit a "
            "list, so what resumes is what was committed" % where
        )

    raise TypeError(
        "cannot checkpoint %s: a %s is not JSON data (commit only str, int, float, "
        "bool, None, list, or dict with string keys)" % (where, type(value).__name__)
    )


def _to_checkpoint(row, frame):
    state, path, provenance = frame
    return Checkpoint(
        run_id=row["run_id"],
        step=row["step"],
        node=row["node"],
        next_node=row["next_node"],
        state=state,
        path=path,
        provenance=provenance,
        failures=json.loads(row["failures"]),
        output=None if row["output"] is None else json.loads(row["output"]),
        pack=row["pack"],
        pack_version=row["pack_version"],
        created_at=row["created_at"],
    )
