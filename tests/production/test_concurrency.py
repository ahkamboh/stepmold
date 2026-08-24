"""Many runs, one store file — what jig's SQLite state layer does under real concurrency.

`jig/state.py` has only ever been driven one run at a time, in-process, with a model that
answers instantly. Production is the opposite: a pool of workers, one store file, and a
network in front of every generation, so the window between any two of a run's operations
is milliseconds wide instead of microseconds. This file walks that window.

What holds up (and is guarded here so it keeps holding):

* N concurrent runs with distinct ids share one file cleanly. Every chain is complete,
  every checkpoint carries its own run's state, and nothing bleeds across runs.
* Concurrent writers, threads or processes, do not lose rows. sqlite3's default 5s busy
  timeout absorbs contention at this scale.
* Resuming a run that already finished is idempotent even when two supervisors do it at
  once: `state.resume` replays the checkpoint and never calls the model.

And what used to give way, each now asserted the other way round:

* Two runs that start together under one id no longer weld into one chain. The first
  checkpoint of a run claims the id inside its own transaction, so the loser is refused
  with `RunIdInUse` and writes nothing.
* Concurrent `resume` of an *unfinished* run no longer double-executes the tail:
  `state.resume` holds a lease for the length of the walk.
* A `Store` is safe to share between threads, and says so.
* `Store.__init__` no longer races itself on first open: `makedirs` is `exist_ok`, and
  the migration asks sqlite for the column rather than reading and then deciding.
* Contention past the busy timeout arrives as `StoreBusy`, a `JigError`, not as a raw
  `sqlite3.OperationalError`.
* A run's checkpoints cost bytes linear in its length, and `prune`/`vacuum` give an
  operator a retention policy.

Every test here is deterministic: threads meet at barriers, never at sleeps, and the two
races whose interleaving cannot be forced through the public API are reproduced by
replaying that interleaving statement for statement.
"""

import json
import multiprocessing
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import types
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import jig.state
from jig.errors import JigError, RunIdInUse
from jig.graph import run
from jig.pack import Edge, Node, Pack
from jig.state import ResumeInProgress, Store, StoreBusy, resume


# How long any thread will wait for its partner before the test fails instead of hanging.
# Nothing here is timing-sensitive: a barrier either meets or the test reports a real
# deadlock, and a machine slow enough to miss this is broken, not busy.
RENDEZVOUS = 5.0

# The latency a real endpoint adds to every generation. Small enough to keep the suite
# fast, large enough that a thread reliably yields inside the model call — which is where
# a production interleaving actually happens.
NETWORK_DELAY = 0.002


# ------------------------------------------------------------------ fixtures


def _generate(name, prompt, output):
    return Node(
        name=name, type="generate", prompt=prompt, grammar={"type": "object"},
        output=output,
    )


def two_step_pack(name="triage", version="2.1"):
    """one -> two -> done. Two generations, three checkpoints, one output."""
    return Pack(
        path="<memory>",
        name=name,
        version=version,
        entry="one",
        model=None,
        nodes={
            "one": _generate("one", "step one for {ticket}", "a"),
            "two": _generate("two", "step two given {a}", "b"),
            "done": Node(name="done", type="end"),
        },
        edges=[Edge("one", "two", None), Edge("two", "done", None)],
    )


def chain_pack(length):
    """A straight line of `length` generate nodes, each committing its own key.

    State only grows along the walk, which is what a real multi-step workflow does and
    what makes the checkpoint-size question below worth asking.
    """
    nodes = {}
    edges = []
    for index in range(length):
        name = "n%d" % index
        target = "n%d" % (index + 1) if index + 1 < length else "done"
        nodes[name] = _generate(name, "step {ticket}", "out%d" % index)
        edges.append(Edge(name, target, None))
    nodes["done"] = Node(name="done", type="end")
    return Pack(
        path="<memory>", name="chain", version=1, entry="n0", model=None,
        nodes=nodes, edges=edges, max_steps=1000,
    )


@dataclass
class MarkerModel:
    """A model that answers with its own marker, after a small network-shaped delay.

    Every generation is recorded, so a test can prove how many times a node ran — which
    is the only way to catch a resume that executed twice.
    """

    marker: str
    delay: float = NETWORK_DELAY
    payload: str = ""
    calls: List[str] = field(default_factory=list, init=False)
    #: waited on once, at the start of the first generation
    gate: Optional[threading.Barrier] = None
    #: waited on after the gate, so one thread finishes its whole run before the other
    after: Optional[threading.Event] = None

    def generate(self, prompt, grammar=None, max_tokens=512):
        self.calls.append(prompt)
        if len(self.calls) == 1:
            if self.gate is not None:
                self.gate.wait(timeout=RENDEZVOUS)
            if self.after is not None and not self.after.wait(timeout=RENDEZVOUS):
                raise AssertionError("partner thread never finished its run")
        time.sleep(self.delay)
        return json.dumps({"v": self.marker, "pad": self.payload})


class DyingModel:
    """Answers `alive` times, then behaves like a backend that fell over.

    Used to leave a real half-finished run in the store: the walker checkpoints node one,
    then the RuntimeError escapes (it is not `NodeFailed`, so no `on_fail` catches it).
    """

    def __init__(self, marker, alive=1):
        self.marker = marker
        self.alive = alive
        self.calls = []

    def generate(self, prompt, grammar=None, max_tokens=512):
        self.calls.append(prompt)
        if len(self.calls) > self.alive:
            raise RuntimeError("backend down")
        return json.dumps({"v": self.marker})


class StoreTempDir(unittest.TestCase):
    """A real store file on disk — `:memory:` cannot be contended."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "runs.db")
        Store(self.path).close()  # create the schema once, outside the race

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def open_store(self):
        store = Store(self.path)
        self.addCleanup(store.close)
        return store

    def rows(self):
        store = self.open_store()
        return store._connection.execute(
            "SELECT COUNT(*) FROM checkpoints"
        ).fetchone()[0]


def _run_all(targets):
    """Start every callable in its own thread, join them, return nothing.

    Joins with a timeout so a deadlock is reported as a failed test rather than a hung
    suite — the whole point of a production test is that it terminates.
    """
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=RENDEZVOUS * 4)
    alive = [thread for thread in threads if thread.is_alive()]
    if alive:
        raise AssertionError("%d worker thread(s) never finished" % len(alive))


# ------------------------------------------------- N runs, distinct ids, one file


class ConcurrentRunsShareOneStoreFile(StoreTempDir):
    """The shape jig is actually deployed in: a worker pool, one store file.

    Each worker opens its own `Store` (see `SharingOneStoreBetweenThreads` below for why
    it has no choice) and walks its own run id.
    """

    WORKERS = 24

    def test_every_concurrent_run_completes_with_its_chain_intact(self):
        results = {}
        errors = {}

        def worker(index):
            def go():
                store = Store(self.path)
                try:
                    results[index] = run(
                        two_step_pack(),
                        MarkerModel("m%d" % index),
                        inputs={"ticket": "t%d" % index},
                        run_id="run-%02d" % index,
                        store=store,
                    )
                except BaseException as exc:  # noqa: BLE001 - the test is the report
                    errors[index] = repr(exc)
                finally:
                    store.close()
            return go

        _run_all([worker(index) for index in range(self.WORKERS)])

        self.assertEqual(errors, {}, "concurrent runs raised")
        self.assertEqual(len(results), self.WORKERS)

        store = self.open_store()
        self.assertEqual(len(store.runs()), self.WORKERS)
        for index in range(self.WORKERS):
            run_id = "run-%02d" % index
            history = store.history(run_id)
            # Three nodes, three checkpoints, numbered without a gap: a chain that lost a
            # write or picked up a neighbour's would show up here first.
            self.assertEqual([c.step for c in history], [1, 2, 3], run_id)
            self.assertEqual([c.node for c in history], ["one", "two", "done"], run_id)
            for checkpoint in history:
                # Attribution: this run's inputs, this run's generations, this run's pack.
                self.assertEqual(checkpoint.run_id, run_id)
                self.assertEqual(checkpoint.state["ticket"], "t%d" % index)
                self.assertEqual(checkpoint.state["a"]["v"], "m%d" % index)
                self.assertEqual(checkpoint.pack, "triage")
                self.assertEqual(checkpoint.pack_version, "2.1")
            self.assertTrue(history[-1].finished)
            self.assertEqual(history[-1].output["b"]["v"], "m%d" % index)
            self.assertEqual(results[index].output, history[-1].output)

    def test_auto_generated_run_ids_do_not_collide_under_concurrency(self):
        """With no id supplied the walker mints one; two workers must not mint the same."""
        results = []
        lock = threading.Lock()

        def worker(index):
            def go():
                store = Store(self.path)
                try:
                    result = run(
                        two_step_pack(),
                        MarkerModel("m%d" % index),
                        inputs={"ticket": "t%d" % index},
                        store=store,
                    )
                    with lock:
                        results.append(result.run_id)
                finally:
                    store.close()
            return go

        _run_all([worker(index) for index in range(12)])

        self.assertEqual(len(results), 12)
        self.assertEqual(len(set(results)), 12, "two runs were minted the same id")
        self.assertEqual(sorted(self.open_store().runs()), sorted(results))


# --------------------------------------------------- two runs, one run id: the race


class TwoRunsRacingOnOneRunId(StoreTempDir):
    """One run id, two runs starting together — and only one of them survives.

    `graph.run` asks `store.latest(run_id)` and, seeing nothing, walks on. Nothing is
    written until the first node completes, one whole model call later, so two callers
    that enter that window together both see an empty store and both clear that check.
    The read-then-write guard cannot close its own window.

    The store closes it instead. The first checkpoint of a run — step 1, the walker's
    first — claims the run id by inserting a row keyed on it, inside the same transaction
    that writes the checkpoint. sqlite decides the winner; the loser is refused with
    `RunIdInUse` and leaves nothing behind, so the surviving chain is one run's and the
    caller of the other is told, rather than handed a zero exit code over deleted data.
    """

    def _race(self):
        """Run two workers under one id, with the interleaving pinned.

        `gate` holds both inside their first generation, which proves both cleared the
        `RunIdInUse` check in `graph.run` before either wrote anything — the window the
        store now has to cover. `first_done` then lets the "first" worker finish alone,
        so "second" is deterministically the one that arrives at a claimed id.
        """
        gate = threading.Barrier(2)
        first_done = threading.Event()
        results = {}
        errors = {}

        def worker(marker, after):
            def go():
                store = Store(self.path)
                model = MarkerModel(marker, gate=gate, after=after)
                try:
                    results[marker] = run(
                        two_step_pack(), model, inputs={"ticket": marker},
                        run_id="R", store=store,
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors[marker] = exc
                finally:
                    if marker == "first":
                        first_done.set()
                    store.close()
            return go

        _run_all([worker("first", None), worker("second", first_done)])
        return results, errors

    def test_the_second_run_under_a_live_id_is_refused_by_the_store(self):
        results, errors = self._race()

        self.assertEqual(sorted(results), ["first"])
        self.assertEqual(sorted(errors), ["second"])
        self.assertIsInstance(errors["second"], RunIdInUse)
        self.assertIn("R", str(errors["second"]))
        self.assertEqual(results["first"].run_id, "R")
        self.assertEqual(results["first"].output["ticket"], "first")

    def test_the_surviving_chain_belongs_to_exactly_one_of_them(self):
        results, errors = self._race()
        self.assertEqual(sorted(errors), ["second"])

        store = self.open_store()
        history = store.history("R")

        # Three nodes, three checkpoints, all of them the winner's. The loser's refusal
        # happened inside the transaction that would have written its first row, so there
        # is no half-run to clean up and nothing of "second" in the chain.
        self.assertEqual([c.step for c in history], [1, 2, 3])
        self.assertEqual(store.runs(), ["R"])
        self.assertEqual(
            {checkpoint.state["ticket"] for checkpoint in history}, {"first"}
        )
        self.assertEqual(results["first"].output, history[-1].output)

    def test_resuming_the_contested_run_id_hands_back_the_winners_data(self):
        """The leak, end to end — and now it is not one."""
        results, _ = self._race()
        store = self.open_store()

        replayed = resume(two_step_pack(), MarkerModel("unused"), "R", store)

        self.assertEqual(replayed.output["ticket"], "first")
        self.assertEqual(replayed.output, results["first"].output)

    def test_the_check_still_works_when_the_runs_do_not_overlap(self):
        """The sequential case the walker\'s own guard already caught still raises."""
        store = self.open_store()
        run(two_step_pack(), MarkerModel("a"), inputs={"ticket": "a"},
            run_id="S", store=store)
        with self.assertRaises(RunIdInUse):
            run(two_step_pack(), MarkerModel("b"), inputs={"ticket": "b"},
                run_id="S", store=store)

    def test_a_deleted_run_frees_its_id_again(self):
        """The claim is the run, not the name: delete the run and the id is reusable."""
        store = self.open_store()
        run(two_step_pack(), MarkerModel("a"), inputs={"ticket": "a"},
            run_id="S", store=store)
        store.delete("S")
        second = run(two_step_pack(), MarkerModel("b"), inputs={"ticket": "b"},
                     run_id="S", store=store)
        self.assertEqual(second.output["ticket"], "b")
        self.assertEqual([c.step for c in store.history("S")], [1, 2, 3])

    def test_the_claim_survives_the_store_that_made_it(self):
        """A worker that exits does not release the id — only `delete` does.

        Otherwise a fleet that restarts mid-run would let a second run start on top of a
        chain it is about to resume.
        """
        first = Store(self.path)
        first.save(run_id="T", step=1, node="one", next_node="two", state={"a": 1})
        first.close()

        second = self.open_store()
        with self.assertRaises(RunIdInUse):
            second.save(run_id="T", step=1, node="one", next_node="two", state={"a": 2})
        self.assertEqual(second.latest("T").state, {"a": 1})


# ------------------------------------------------------------ concurrent resume


class ConcurrentResumeOfOneRun(StoreTempDir):
    """Two supervisors retrying the same resume at once.

    This is not exotic: a retry loop plus a stuck health check produces it on the first
    bad night. `state.resume` promises idempotence — "a supervisor can retry a resume
    without paying for it twice" — and used to deliver it for a *finished* run only. An
    unfinished one now runs under a lease: whoever takes it walks the tail, and the other
    is refused with `ResumeInProgress` or, if it arrives after the walk finished, handed
    the same replay everyone else gets.
    """

    def _half_finished_run(self):
        """Leave run "R" checkpointed at node one, pointing at node two."""
        store = self.open_store()
        with self.assertRaises(RuntimeError):
            run(two_step_pack(), DyingModel("seed"), inputs={"ticket": "t"},
                run_id="R", store=store)
        history = store.history("R")
        self.assertEqual([c.step for c in history], [1])
        self.assertEqual(history[0].next_node, "two")
        self.assertFalse(history[0].finished)
        return store

    def test_concurrent_resume_of_an_unfinished_run_runs_the_tail_once(self):
        self._half_finished_run()

        start = threading.Barrier(2)
        models = {}
        results = {}
        errors = {}

        def worker(marker):
            def go():
                store = Store(self.path)
                model = MarkerModel(marker)
                models[marker] = model
                start.wait(timeout=RENDEZVOUS)
                try:
                    results[marker] = resume(two_step_pack(), model, "R", store)
                except BaseException as exc:  # noqa: BLE001
                    errors[marker] = exc
                finally:
                    store.close()
            return go

        _run_all([worker("first"), worker("second")])

        # Node "two" is the only node left to run, and whatever it does downstream —
        # charge, email, ticket — it does once for this run. Either the loser was refused
        # while the winner walked, or it arrived afterwards and replayed the winner\'s
        # finished chain; neither path generates.
        spent = sum(len(model.calls) for model in models.values())
        self.assertEqual(spent, 1, "the tail of the run was executed twice")
        for exc in errors.values():
            self.assertIsInstance(exc, ResumeInProgress)
            self.assertIsInstance(exc, JigError)
        self.assertEqual(len(results) + len(errors), 2)
        self.assertGreaterEqual(len(results), 1)

        store = self.open_store()
        history = store.history("R")
        self.assertEqual([c.step for c in history], [1, 2, 3])
        # Every caller that was told its run succeeded holds the chain that is in the
        # store — nobody walks away with an answer the store does not have.
        for result in results.values():
            self.assertEqual(result.output, history[-1].output)

    def test_a_second_resume_while_the_first_holds_the_lease_is_refused(self):
        """The refusal on its own, with the interleaving held open rather than raced for."""
        holder = self._half_finished_run()
        other = Store(self.path)
        self.addCleanup(other.close)
        model = MarkerModel("late")

        with holder.lease("R"):
            with self.assertRaises(ResumeInProgress) as caught:
                resume(two_step_pack(), model, "R", other)

        self.assertIn("R", str(caught.exception))
        self.assertEqual(model.calls, [], "a refused resume still paid for a generation")
        self.assertEqual([c.step for c in other.history("R")], [1])

    def test_the_lease_is_released_when_the_resume_returns(self):
        """A held lease is a bug if it outlives its holder, so releasing it is the test."""
        self._half_finished_run()
        store = self.open_store()
        result = resume(two_step_pack(), MarkerModel("done"), "R", store)
        self.assertEqual(result.output["b"]["v"], "done")
        # Nothing holds it any more: the next caller gets it without waiting.
        with store.lease("R"):
            pass

    def test_a_lease_left_behind_by_a_dead_resumer_expires(self):
        """The holder cannot release what it no longer has a process for.

        So the lease carries an expiry, and a resumer that arrives after it can take the
        run. Without this a worker killed mid-resume would wedge its run forever.
        """
        self._half_finished_run()
        dead = self.open_store()
        alive = self.open_store()
        with dead.lease("R", seconds=-1.0):  # already stale by the time anyone asks
            result = resume(two_step_pack(), MarkerModel("taken-over"), "R", alive)
        self.assertEqual(result.output["b"]["v"], "taken-over")

    def test_a_lone_resume_of_a_half_finished_run_still_works(self):
        """The forward walk that died left no lease behind to block its own recovery."""
        self._half_finished_run()
        store = self.open_store()
        result = resume(two_step_pack(), MarkerModel("recovered"), "R", store)
        self.assertEqual(result.output["b"]["v"], "recovered")
        self.assertEqual([c.step for c in store.history("R")], [1, 2, 3])

    def test_concurrent_resume_of_a_finished_run_is_idempotent(self):
        """The half that always held: a finished run replays, and never calls the model."""
        store = self.open_store()
        run(two_step_pack(), MarkerModel("done"), inputs={"ticket": "t"},
            run_id="F", store=store)
        rows_before = self.rows()

        models = {}
        results = {}

        def worker(marker):
            def go():
                worker_store = Store(self.path)
                model = MarkerModel(marker)
                models[marker] = model
                try:
                    results[marker] = resume(two_step_pack(), model, "F", worker_store)
                finally:
                    worker_store.close()
            return go

        _run_all([worker("p"), worker("q")])

        for marker in ("p", "q"):
            self.assertEqual(models[marker].calls, [], "replay called the model")
            self.assertEqual(results[marker].output["a"]["v"], "done")
        self.assertEqual(results["p"].output, results["q"].output)
        self.assertEqual(self.rows(), rows_before, "replay wrote to the store")


# ------------------------------------------------- sharing one Store between threads


class SharingOneStoreBetweenThreads(StoreTempDir):
    """One `Store`, a thread pool — the obvious deployment, and now a supported one.

    `sqlite3.connect` defaults to `check_same_thread=True`, so a `Store` touched from any
    thread but the one that built it used to die on a bare `sqlite3.ProgrammingError`:
    not a `JigError`, so `except JigError` around a run did not catch it, and a message
    about thread identities rather than about jig. `Store` now opens its connection with
    `check_same_thread=False` and serialises every statement on its own lock, which is
    cheaper than a connection per thread on a file this small.
    """

    def test_a_store_created_in_one_thread_can_be_used_from_another(self):
        store = self.open_store()
        failures = []

        def go():
            try:
                store.save(run_id="r", step=1, node="n", next_node=None, state={"a": 1})
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        _run_all([go])

        self.assertEqual([repr(f) for f in failures], [])
        self.assertEqual(store.latest("r").state, {"a": 1})

    def test_reads_cross_threads_too(self):
        store = self.open_store()
        store.save(run_id="r", step=1, node="n", next_node=None, state={"a": 1})
        seen = []
        failures = []

        def go():
            try:
                seen.append(store.latest("r").state)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        _run_all([go])
        self.assertEqual([repr(f) for f in failures], [])
        self.assertEqual(seen, [{"a": 1}])

    def test_many_threads_writing_through_one_shared_store_lose_nothing(self):
        """The lock has to serialise, not just permit: eight threads, one connection."""
        store = self.open_store()
        writers, per_writer = 8, 30
        errors = []

        def worker(index):
            def go():
                try:
                    for step in range(1, per_writer + 1):
                        store.save(run_id="s%d" % index, step=step, node="n",
                                   next_node="m", state={"w": index, "k": step})
                except BaseException as exc:  # noqa: BLE001
                    errors.append(repr(exc))
            return go

        _run_all([worker(index) for index in range(writers)])

        self.assertEqual(errors, [])
        self.assertEqual(self.rows(), writers * per_writer)
        for index in range(writers):
            history = store.history("s%d" % index)
            self.assertEqual([c.state["k"] for c in history],
                             list(range(1, per_writer + 1)))
            self.assertEqual([c.state["w"] for c in history], [index] * per_writer)

    def test_a_whole_run_on_a_store_owned_by_another_thread_completes(self):
        """The expensive shape of the old defect: the failure used to land on the first
        `save`, after the first generation had been issued, billed and thrown away."""
        store = self.open_store()
        model = MarkerModel("m")
        failures = []
        results = []

        def go():
            try:
                results.append(
                    run(two_step_pack(), model, inputs={"ticket": "t"}, store=store)
                )
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        _run_all([go])

        self.assertEqual([repr(f) for f in failures], [])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.rows(), 3, "the run was not checkpointed")
        self.assertEqual(results[0].output["b"]["v"], "m")

    def test_the_store_documents_that_it_is_shareable(self):
        """A supported property nobody can find is still a trap.

        The old version of this test asserted the opposite — that nothing anywhere said
        `Store` was single-threaded — and named its own deletion as the fix.
        """
        text = (jig.state.__doc__ or "") + (Store.__doc__ or "") + (
            Store.__init__.__doc__ or ""
        )
        lowered = text.lower()
        self.assertIn("thread", lowered)
        self.assertIn("check_same_thread", lowered)


# ------------------------------------------------------------- sqlite under load


class SqliteLockingUnderContention(StoreTempDir):
    """What happens when many writers hit one file, and what it looks like when it hurts.

    At the scale a jig deployment writes — a handful of small rows per run — contention
    is absorbed and nothing is lost. The question this class exists for is the other one:
    what a caller sees when the wait is not enough, and whether they have any way to
    change the wait.
    """

    def test_the_store_picks_its_own_timeout_and_journal_and_offers_the_knob(self):
        store = self.open_store()
        timeout_ms = store._connection.execute("PRAGMA busy_timeout").fetchone()[0]
        journal = store._connection.execute("PRAGMA journal_mode").fetchone()[0]

        # A decision rather than sqlite3.connect\'s 5s default: a store is written by a
        # fleet and may sit on a slow or networked filesystem. WAL because the rollback
        # journal makes a writer exclude every reader for the length of its transaction,
        # and `history()` over a large store is exactly such a reader.
        self.assertEqual(timeout_ms, int(jig.state.DEFAULT_TIMEOUT * 1000))
        self.assertEqual(journal, "wal")

        import inspect

        parameters = inspect.signature(Store.__init__).parameters
        self.assertEqual(list(parameters), ["self", "path", "timeout"])
        slow = Store(self.path, timeout=0.25)
        self.addCleanup(slow.close)
        self.assertEqual(
            slow._connection.execute("PRAGMA busy_timeout").fetchone()[0], 250
        )

    def test_an_in_memory_store_does_not_try_to_journal_to_a_file(self):
        """WAL is meaningless without a file, and asking for it there is not harmless."""
        store = Store(":memory:")
        self.addCleanup(store.close)
        journal = store._connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(journal, "memory")

    def test_many_threaded_writers_on_one_file_lose_nothing(self):
        """Eight writers, thirty checkpoints each. Every row must be there afterwards."""
        writers, per_writer = 8, 30
        errors = []

        def worker(index):
            def go():
                store = Store(self.path)
                try:
                    for step in range(per_writer):
                        store.save(
                            run_id="w%d" % index, step=step, node="n",
                            next_node="m", state={"w": index, "k": step},
                        )
                except BaseException as exc:  # noqa: BLE001
                    errors.append(repr(exc))
                finally:
                    store.close()
            return go

        _run_all([worker(index) for index in range(writers)])

        self.assertEqual(errors, [], "concurrent writers hit a lock error")
        self.assertEqual(self.rows(), writers * per_writer)
        store = self.open_store()
        for index in range(writers):
            history = store.history("w%d" % index)
            self.assertEqual(len(history), per_writer)
            # A row that landed under the wrong run id, or a state that came back as some
            # other writer\'s, would show here.
            self.assertEqual(
                [c.state["w"] for c in history], [index] * per_writer
            )

    def test_a_write_that_outlasts_the_busy_timeout_raises_a_jig_error(self):
        """The failure the timeout only postpones, reported as jig\'s rather than sqlite\'s.

        The timeout is shortened on this one connection so the test costs 100ms instead
        of the store\'s real wait; the wait is the only thing scaled, the outcome is not.
        A long write elsewhere — a backup, another host on an NFS mount — produces the
        same thing at the real timeout.
        """
        store = self.open_store()
        store._connection.execute("PRAGMA busy_timeout=100")

        blocker = sqlite3.connect(self.path)
        self.addCleanup(blocker.close)
        blocker.execute("BEGIN EXCLUSIVE")
        blocker.execute(
            "INSERT INTO checkpoints (run_id, step, node, next_node, state, path,"
            " provenance, failures, created_at) VALUES"
            " ('blocker', 1, 'n', NULL, '{}', '[]', '{}', '[]', 'now')"
        )

        started = time.monotonic()
        with self.assertRaises(StoreBusy) as caught:
            store.save(run_id="victim", step=1, node="one", next_node="two",
                       state={"work": "already done"})
        waited = time.monotonic() - started
        blocker.rollback()

        self.assertLess(waited, 2.0)
        # A JigError, so `except JigError` around a run catches it, and it names the run,
        # the step and the node so an operator knows what to re-drive.
        self.assertIsInstance(caught.exception, JigError)
        self.assertIn("victim", str(caught.exception))
        self.assertIn("one", str(caught.exception))
        self.assertIn("timeout", str(caught.exception))
        self.assertIsNone(store.latest("victim"))

        # And the failure is clean: the transaction was rolled back rather than left
        # open, so the same Store writes again the moment the blocker lets go.
        self.assertFalse(store._connection.in_transaction)
        store.save(run_id="victim", step=1, node="one", next_node="two", state={"a": 1})
        self.assertEqual(store.latest("victim").state, {"a": 1})

    def test_an_error_that_is_not_contention_is_not_disguised_as_contention(self):
        """`StoreBusy` must mean the lock and only the lock."""
        store = self.open_store()
        store._connection.execute("DROP TABLE checkpoints")
        with self.assertRaises(sqlite3.OperationalError) as caught:
            store.save(run_id="r", step=1, node="n", next_node=None, state={})
        self.assertNotIsInstance(caught.exception, StoreBusy)


def _process_writer(path, worker, count, queue):
    """One child process: open its own Store on the shared file and write `count` rows.

    Module level because `spawn` pickles the target by name.
    """
    try:
        from jig.state import Store as ChildStore

        store = ChildStore(path)
        for step in range(count):
            store.save(run_id="p%d" % worker, step=step, node="n", next_node="m",
                       state={"p": worker, "k": step})
        store.close()
        queue.put(("ok", worker))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("err", "%s: %s" % (type(exc).__name__, exc)))


class SeparateProcessesWritingOneStore(StoreTempDir):
    """Threads share a GIL; processes share only the file, and the OS locks it for real.

    This is the deployment shape jig will actually meet — several worker processes on one
    box, one store file — so it is worth proving the file locking holds where the
    in-process serialisation cannot help.
    """

    WORKERS = 4
    PER_WORKER = 30

    def test_processes_writing_one_file_lose_nothing(self):
        try:
            context = multiprocessing.get_context("spawn")
        except ValueError:  # pragma: no cover - every supported platform has spawn
            self.skipTest("no spawn start method on this platform")

        queue = context.Queue()
        processes = [
            context.Process(
                target=_process_writer,
                args=(self.path, index, self.PER_WORKER, queue),
            )
            for index in range(self.WORKERS)
        ]
        for process in processes:
            process.start()
        self.addCleanup(lambda: [p.terminate() for p in processes if p.is_alive()])

        outcomes = [queue.get(timeout=RENDEZVOUS * 6) for _ in range(self.WORKERS)]
        for process in processes:
            process.join(timeout=RENDEZVOUS * 6)

        self.assertEqual(
            [kind for kind, _ in outcomes], ["ok"] * self.WORKERS, outcomes
        )
        self.assertEqual(self.rows(), self.WORKERS * self.PER_WORKER)
        store = self.open_store()
        for index in range(self.WORKERS):
            history = store.history("p%d" % index)
            self.assertEqual(len(history), self.PER_WORKER)
            self.assertEqual([c.state["p"] for c in history], [index] * self.PER_WORKER)


# ------------------------------------------- opening a Store is no longer a race


class OpeningAStoreRacesItself(unittest.TestCase):
    """`Store.__init__` used to do two read-then-write things before it was usable.

    Both only matter on a cold start — the first time a fleet of workers opens a store
    that does not exist yet, or the first time it opens a store file an older jig wrote.
    That is exactly the moment a deployment restarts every worker at once, so both were
    guaranteed to fire and neither had anything to do with a real fault.
    """

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_makedirs_survives_another_worker_creating_the_directory_first(self):
        """The losing worker\'s exact code path, forced rather than raced for.

        The loser of a two-worker cold start is the process whose look at the directory
        ran before the winner\'s `makedirs` and whose own `makedirs` ran after it.
        Substituting a `makedirs` that creates the directory before delegating reproduces
        that interleaving with no scheduling luck involved: whatever jig passes has to
        survive the directory already being there.
        """
        real_makedirs = os.makedirs

        def racing_makedirs(name, *args, **kwargs):
            real_makedirs(name, exist_ok=True)  # the winner, in the window
            return real_makedirs(name, *args, **kwargs)  # the loser\'s own call

        stale = types.SimpleNamespace(path=os.path, makedirs=racing_makedirs)
        real_os = jig.state.os
        jig.state.os = stale
        try:
            store = Store(os.path.join(self.directory, "state", "runs.db"))
        finally:
            jig.state.os = real_os
        store.close()
        self.assertTrue(os.path.isdir(os.path.join(self.directory, "state")))

    def test_racing_workers_creating_the_store_directory_all_succeed(self):
        """The unforced version: eight threads meet at a barrier on a fresh path."""
        path = os.path.join(self.directory, "fresh", "runs.db")
        barrier = threading.Barrier(8)
        failures = []
        opened = []

        def go():
            barrier.wait(timeout=RENDEZVOUS)
            try:
                store = Store(path)
                store.close()
                opened.append(True)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        _run_all([go] * 8)

        self.assertEqual([repr(f) for f in failures], [])
        self.assertEqual(len(opened), 8)

    def test_a_store_opens_even_when_the_journal_switch_is_blocked(self):
        """`PRAGMA journal_mode=WAL` answers SQLITE_BUSY without asking the busy handler.

        So a fleet cold-starting on one store file collides over it, and the loser used
        to die in the constructor over a preference. The journal belongs to the file, not
        to this connection: a loser left on the rollback journal is slower, not wrong.
        """
        path = os.path.join(self.directory, "delete_mode.db")
        seed = sqlite3.connect(path)
        self.addCleanup(seed.close)
        seed.executescript(jig.state.SCHEMA)
        seed.commit()

        blocker = sqlite3.connect(path)
        self.addCleanup(blocker.close)
        blocker.execute("BEGIN")
        blocker.execute("SELECT COUNT(*) FROM checkpoints").fetchone()

        store = Store(path, timeout=0.2)
        self.addCleanup(store.close)
        self.assertEqual(
            store._connection.execute("PRAGMA journal_mode").fetchone()[0], "delete"
        )
        blocker.rollback()
        store.save(run_id="r", step=1, node="n", next_node=None, state={"a": 1})
        self.assertEqual(store.latest("r").state, {"a": 1})

    def test_adding_a_migration_column_twice_is_not_an_error(self):
        """The other read-then-write in the constructor, replayed statement for statement.

        `_migrate` used to read `PRAGMA table_info` and, for every column missing, run an
        `ALTER TABLE ... ADD COLUMN`. Two workers opening a pre-`pack_version` store file
        together both read "missing" and both run the ALTER; the second was answered with
        `duplicate column name` out of a constructor, on a store file that was fine. The
        fix is to ask sqlite instead of reading and deciding, so running the statement
        twice is the test.
        """
        self.assertIn(
            "pack_version", [name for name, _ in jig.state._ADDED_COLUMNS],
            "the migration this test replays is gone — retarget or delete it",
        )
        path = os.path.join(self.directory, "legacy.db")
        _write_legacy_store(path)

        connection = sqlite3.connect(path)
        self.addCleanup(connection.close)
        jig.state._add_column(connection, "pack_version", "TEXT")
        jig.state._add_column(connection, "pack_version", "TEXT")  # the loser
        columns = [row[1] for row in connection.execute(
            "PRAGMA table_info(checkpoints)"
        )]
        self.assertEqual(columns.count("pack_version"), 1)

    def test_a_real_sql_error_in_the_migration_still_escapes(self):
        """Tolerating a duplicate column must not tolerate a missing table."""
        path = os.path.join(self.directory, "empty.db")
        connection = sqlite3.connect(path)
        self.addCleanup(connection.close)
        with self.assertRaises(sqlite3.OperationalError) as caught:
            jig.state._add_column(connection, "pack_version", "TEXT")
        self.assertIn("no such table", str(caught.exception))

    def test_two_openers_of_a_legacy_store_both_migrate_it_cleanly(self):
        path = os.path.join(self.directory, "legacy_pair.db")
        _write_legacy_store(path)

        first = Store(path)
        self.addCleanup(first.close)
        second = Store(path)
        self.addCleanup(second.close)
        for store in (first, second):
            columns = [
                row["name"] for row in
                store._connection.execute("PRAGMA table_info(checkpoints)")
            ]
            self.assertEqual(columns.count("pack_version"), 1)

    def test_racing_workers_migrating_a_legacy_store_all_succeed(self):
        """As above: the unforced race, which must now produce no failures at all."""
        path = os.path.join(self.directory, "legacy_race.db")
        _write_legacy_store(path)

        barrier = threading.Barrier(8)
        failures = []
        opened = []

        def go():
            barrier.wait(timeout=RENDEZVOUS)
            try:
                store = Store(path)
                store.close()
                opened.append(True)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        _run_all([go] * 8)

        self.assertEqual([repr(f) for f in failures], [])
        self.assertEqual(len(opened), 8)
        # Whatever else happened, the migration itself succeeded exactly once.
        store = Store(path)
        self.addCleanup(store.close)
        columns = [
            row["name"] for row in
            store._connection.execute("PRAGMA table_info(checkpoints)")
        ]
        self.assertEqual(columns.count("pack_version"), 1)


def _write_legacy_store(path):
    """The checkpoints table as jig shipped it before a055476 added the version column."""
    legacy = sqlite3.connect(path)
    legacy.executescript(
        "CREATE TABLE checkpoints ("
        " run_id TEXT NOT NULL, step INTEGER NOT NULL, node TEXT NOT NULL,"
        " next_node TEXT, state TEXT NOT NULL, path TEXT NOT NULL,"
        " provenance TEXT NOT NULL, failures TEXT NOT NULL, output TEXT,"
        " pack TEXT, created_at TEXT NOT NULL, PRIMARY KEY (run_id, step));"
    )
    legacy.commit()
    legacy.close()


# ----------------------------------------------------------------------- the soak


class TheStoreGrowsAndNothingPrunesIt(StoreTempDir):
    """Sustained runs against one store file, which is what a deployed jig does forever.

    Two separate growth problems live here: how much one run costs, and what an operator
    can do about a month of them.
    """

    def _state_bytes(self, store):
        """Every column that grows along the walk, not just `state`.

        `path` and `provenance` lengthen with every node too, so measuring `state` alone
        would let the quadratic term move one column to the right and call it fixed.
        """
        return store._connection.execute(
            "SELECT SUM(LENGTH(state) + LENGTH(path) + LENGTH(provenance))"
            " FROM checkpoints"
        ).fetchone()[0]

    def test_one_run_costs_bytes_linear_in_its_own_length(self):
        """A checkpoint records what its node changed, so step N no longer re-writes 1..N.

        A twenty-node run used to cost roughly four times a ten-node one; it now costs
        twice. The numbers are exact — the model\'s payload is fixed — so this is a
        measurement, not a heuristic.
        """
        sizes = {}
        naive = {}
        for length in (10, 20):
            directory = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            store = Store(os.path.join(directory, "runs.db"))
            self.addCleanup(store.close)
            run(chain_pack(length), MarkerModel("m", delay=0, payload="x" * 200),
                inputs={"ticket": "t"}, run_id="R", store=store)
            sizes[length] = self._state_bytes(store)
            # What the old store wrote: all three, in full, once per checkpoint.
            naive[length] = sum(
                len(json.dumps(c.state, sort_keys=True))
                + len(json.dumps(c.path))
                + len(json.dumps(c.provenance, sort_keys=True))
                for c in store.history("R")
            )

        self.assertLess(sizes[20], 2.2 * sizes[10], "growth is still superlinear")
        # And the saving against repeating everything grows with the run, which is the
        # point: the longer the workflow, the more the old shape cost.
        self.assertLess(sizes[10] * 4, naive[10])
        self.assertLess(sizes[20] * 8, naive[20])

    def test_a_delta_encoded_chain_still_hands_back_every_state_intact(self):
        """Cheaper on disk is worthless if what comes back is not what was committed."""
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        store = Store(os.path.join(directory, "runs.db"))
        self.addCleanup(store.close)
        result = run(chain_pack(30), MarkerModel("m", delay=0, payload="x" * 200),
                     inputs={"ticket": "t"}, run_id="R", store=store)

        history = store.history("R")
        self.assertEqual(len(history), 31)
        for index, checkpoint in enumerate(history):
            expected = {"ticket": "t"}
            for earlier in range(min(index + 1, 30)):
                expected["out%d" % earlier] = {"v": "m", "pad": "x" * 200}
            self.assertEqual(checkpoint.state, expected, "step %d" % checkpoint.step)
            self.assertEqual(
                checkpoint.path,
                ["n%d" % walked for walked in range(min(index + 1, 30))]
                + (["done"] if index == 30 else []),
            )
            self.assertEqual(
                checkpoint.provenance,
                {"out%d" % w: "n%d" % w for w in range(min(index + 1, 30))},
            )
        # `latest` rebuilds from the snapshot forward; `history` walks the chain. Both
        # have to agree, and with the run itself.
        self.assertEqual(history[-1].state, result.state)
        self.assertEqual(store.latest("R").state, result.state)
        self.assertEqual(store.latest("R").path, result.path)
        self.assertEqual(store.latest("R").provenance, result.provenance)

    def test_a_state_that_is_rewritten_rather_than_grown_is_still_exact(self):
        """The awkward shape for a delta: keys removed, and a value that changes type.

        `True` and `1` are equal in Python, so a naive comparison would call this
        unchanged and hand back the wrong one.
        """
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        store = Store(os.path.join(directory, "runs.db"))
        self.addCleanup(store.close)
        states = [
            {"a": 1, "b": [1, 2], "c": {"k": "v"}},
            {"a": True, "b": [1, 2]},
            {"a": True, "b": [1, 2], "d": None},
            {},
            {"a": 1.0},
        ]
        # A path that diverges rather than only growing — a loop re-walked down another
        # edge — is the same question asked of the column that is encoded as a suffix.
        paths = [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "other"],
            ["one", "other", "three"],
        ]
        for step, (state, path) in enumerate(zip(states, paths), start=1):
            store.save(run_id="R", step=step, node="n", next_node="m", state=state,
                       path=path, provenance={"a": "n%d" % step})
        for checkpoint, state, path in zip(store.history("R"), states, paths):
            self.assertEqual(checkpoint.state, state)
            self.assertEqual(checkpoint.path, path)
            self.assertEqual(checkpoint.provenance, {"a": "n%d" % checkpoint.step})
            for key, value in state.items():
                self.assertIs(type(checkpoint.state[key]), type(value), key)
        self.assertEqual(store.latest("R").state, states[-1])
        self.assertEqual(store.latest("R").path, paths[-1])

    def test_a_soak_of_sequential_runs_grows_until_an_operator_prunes_it(self):
        store = self.open_store()
        model = MarkerModel("m", delay=0, payload="x" * 200)
        pack = chain_pack(3)

        for index in range(120):
            run(pack, model, inputs={"ticket": "t%d" % index},
                run_id="soak-%04d" % index, store=store)
        self.assertEqual(len(store.runs()), 120)

        # Retention is a policy, so nothing happens without one — a store that quietly
        # forgot last month\'s runs would be worse than one that grows.
        self.assertEqual(store.prune(), [])
        self.assertEqual(len(store.runs()), 120)

        dropped = store.prune(keep_last=10)
        self.assertEqual(len(dropped), 110)
        self.assertEqual(sorted(store.runs()), sorted("soak-%04d" % i
                                                      for i in range(110, 120)))
        # The window is by run, not by row: what survives is whole chains.
        for run_id in store.runs():
            self.assertEqual(len(store.history(run_id)), 4)

    def test_prune_never_touches_a_run_that_has_not_finished(self):
        """An unfinished chain is the only copy of work someone still means to resume."""
        store = self.open_store()
        run(two_step_pack(), MarkerModel("done"), inputs={"ticket": "t"},
            run_id="finished", store=store)
        with self.assertRaises(RuntimeError):
            run(two_step_pack(), DyingModel("seed"), inputs={"ticket": "t"},
                run_id="halfway", store=store)

        dropped = store.prune(keep_last=0)
        self.assertEqual(dropped, ["finished"])
        self.assertEqual(store.runs(), ["halfway"])
        # And it is still resumable afterwards.
        result = resume(two_step_pack(), MarkerModel("later"), "halfway", store)
        self.assertEqual(result.output["b"]["v"], "later")

    def test_prune_by_date_drops_what_finished_before_the_cutoff(self):
        store = self.open_store()
        for index in range(3):
            run(two_step_pack(), MarkerModel("m"), inputs={"ticket": "t"},
                run_id="r%d" % index, store=store)

        past = datetime(2000, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(store.prune(before=past), [])
        self.assertEqual(len(store.runs()), 3)

        future = datetime.now(timezone.utc) + timedelta(days=1)
        self.assertEqual(store.prune(before=future), ["r0", "r1", "r2"])
        self.assertEqual(store.runs(), [])

    def test_vacuum_gives_the_disk_back_after_a_prune(self):
        """Deleting rows does not shrink the file — sqlite keeps the pages for reuse."""
        store = self.open_store()
        model = MarkerModel("m", delay=0, payload="x" * 2000)
        for index in range(40):
            run(chain_pack(3), model, inputs={"ticket": "t%d" % index},
                run_id="big-%03d" % index, store=store)
        store._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        grown = os.path.getsize(self.path)

        store.prune(keep_last=0)
        self.assertEqual(self.rows(), 0)
        store._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.assertGreaterEqual(os.path.getsize(self.path), grown)

        store.vacuum()
        self.assertLess(os.path.getsize(self.path), grown)
        # And the store still works afterwards: VACUUM rewrites the file, it does not
        # retire it.
        run(chain_pack(3), model, inputs={"ticket": "after"}, run_id="after",
            store=store)
        self.assertEqual(store.latest("after").state["ticket"], "after")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
