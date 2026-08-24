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
"""

import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .errors import RunError, UnknownRun

__all__ = ["Checkpoint", "CheckpointMismatch", "Store", "resume"]


class CheckpointMismatch(RunError):
    """The checkpoint being resumed was written by a different pack than the one given.

    Lives here rather than in `jig.errors` for the same reason the pack-loading errors
    live in `jig.pack`: it is raised before the walk restarts, by the store's own
    bookkeeping, and nothing in the walker can produce it.
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
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (run_id, step)
);
"""

# Columns added after the first release. A store file written by an older jig has the
# table already, so CREATE TABLE IF NOT EXISTS silently leaves it short a column and
# every read of that column fails. Adding them on open keeps old files readable.
_ADDED_COLUMNS = (("pack_version", "TEXT"),)


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
    """Run checkpoints in a SQLite file (or `:memory:` for tests)."""

    def __init__(self, path=":memory:"):
        self.path = path
        if path != ":memory:":
            directory = os.path.dirname(os.path.abspath(path))
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._migrate()
        self._connection.commit()

    def _migrate(self):
        """Add columns a store file written by an older jig is missing."""
        present = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(checkpoints)")
        }
        for column, declared_type in _ADDED_COLUMNS:
            if column not in present:
                self._connection.execute(
                    "ALTER TABLE checkpoints ADD COLUMN %s %s" % (column, declared_type)
                )

    def save(self, run_id, step, node, next_node, state, path=None, provenance=None,
             failures=None, output=None, pack=None, pack_version=None):
        """Record one completed node. Re-saving the same step replaces it.

        `pack` is the pack's identity: either its name, or the pack itself, in which case
        its version is recorded too and `resume` can catch a graph that moved on under a
        run. Every field is checked before anything is written, so a value JSON would
        change on the way back raises here and leaves no half-written checkpoint.
        """
        name, version = _identity(pack, pack_version)
        row = (
            run_id,
            step,
            node,
            next_node,
            _dump(state, "state"),
            _dump(list(path or []), "path"),
            _dump(dict(provenance or {}), "provenance"),
            _dump(list(failures or []), "failures"),
            None if output is None else _dump(output, "output"),
            name,
            version,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(run_id, step, node, next_node, state, path, provenance, failures, "
            " output, pack, pack_version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        self._connection.commit()

    def latest(self, run_id):
        """The most recent checkpoint for `run_id`, or None."""
        cursor = self._connection.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step DESC LIMIT 1",
            (run_id,),
        )
        row = cursor.fetchone()
        return _to_checkpoint(row) if row else None

    def history(self, run_id):
        """Every checkpoint for `run_id`, oldest first — the run's audit trail."""
        cursor = self._connection.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step", (run_id,)
        )
        return [_to_checkpoint(row) for row in cursor.fetchall()]

    def runs(self):
        """Every run id the store knows about."""
        cursor = self._connection.execute("SELECT DISTINCT run_id FROM checkpoints")
        return [row["run_id"] for row in cursor.fetchall()]

    def delete(self, run_id):
        self._connection.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
        self._connection.commit()

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def resume(pack, model, run_id, store, **kwargs):
    """Continue `run_id` from its last checkpoint, under the pack that started it.

    A run that already finished is returned as-is, without calling the model — resuming
    is idempotent, so a supervisor can retry a resume without paying for it twice.

    `pack` is checked against the checkpoint first, finished run or not: replaying one
    graph's output as another's is the same lie whether the walk continues or not.
    """
    from .graph import run as run_pack, replay  # local: graph imports nothing from here

    checkpoint = store.latest(run_id)
    if checkpoint is None:
        raise UnknownRun("no checkpoint found for run %r" % run_id)
    _check_same_pack(pack, checkpoint)
    if checkpoint.finished:
        return replay(checkpoint)
    return run_pack(pack, model, run_id=run_id, store=store, resume_from=checkpoint,
                    **kwargs)


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


def _to_checkpoint(row):
    return Checkpoint(
        run_id=row["run_id"],
        step=row["step"],
        node=row["node"],
        next_node=row["next_node"],
        state=json.loads(row["state"]),
        path=json.loads(row["path"]),
        provenance=json.loads(row["provenance"]),
        failures=json.loads(row["failures"]),
        output=None if row["output"] is None else json.loads(row["output"]),
        pack=row["pack"],
        pack_version=row["pack_version"],
        created_at=row["created_at"],
    )
