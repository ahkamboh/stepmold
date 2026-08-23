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
"""

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .errors import UnknownRun

__all__ = ["Checkpoint", "Store", "resume"]

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
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (run_id, step)
);
"""


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
        self._connection.commit()

    def save(self, run_id, step, node, next_node, state, path=None, provenance=None,
             failures=None, output=None, pack=None):
        """Record one completed node. Re-saving the same step replaces it."""
        row = (
            run_id,
            step,
            node,
            next_node,
            _dump(state),
            _dump(list(path or [])),
            _dump(dict(provenance or {})),
            _dump(list(failures or [])),
            None if output is None else _dump(output),
            pack,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(run_id, step, node, next_node, state, path, provenance, failures, "
            " output, pack, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
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
    """Continue `run_id` from its last checkpoint.

    A run that already finished is returned as-is, without calling the model — resuming
    is idempotent, so a supervisor can retry a resume without paying for it twice.
    """
    from .graph import run as run_pack, replay  # local: graph imports nothing from here

    checkpoint = store.latest(run_id)
    if checkpoint is None:
        raise UnknownRun("no checkpoint found for run %r" % run_id)
    if checkpoint.finished:
        return replay(checkpoint)
    return run_pack(pack, model, run_id=run_id, store=store, resume_from=checkpoint,
                    **kwargs)


def _dump(value):
    return json.dumps(value, sort_keys=True)


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
        created_at=row["created_at"],
    )
