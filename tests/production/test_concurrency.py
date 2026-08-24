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

What does not, each marked `FINDING` where it is documented:

* `RunIdInUse` is read-then-write, so two runs that start together both pass it and weld
  themselves into one chain — the exact data leak the check was added to stop.
* Concurrent `resume` of an *unfinished* run double-executes every remaining node.
* A `Store` cannot be shared between threads at all, and nothing says so.
* `Store.__init__` races itself on first open: unguarded `makedirs`, and an unguarded
  schema migration.
* Nothing prunes. Per-run bytes grow quadratically in the length of the run.

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
from typing import List, Optional

import jig.state
from jig.errors import JigError, RunIdInUse
from jig.graph import run
from jig.pack import Edge, Node, Pack
from jig.state import Store, resume


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
    """`RunIdInUse` is read-then-write, so it does not survive two runs starting together.

    `graph.run` asks `store.latest(run_id)` and, seeing nothing, walks on. Nothing is
    written until the first node completes — one whole model call later. Two callers that
    enter that window together both see an empty store, both pass the check, and both
    write into the same `(run_id, step)` primary key, where `INSERT OR REPLACE` makes the
    later write win.

    The result is precisely what commit 6f5da57 added `RunIdInUse` to prevent, restated
    as a race: one caller's checkpoint chain is gone, and anyone who resumes that run id
    is handed the *other* caller's data with a zero exit code.
    """

    def _race(self):
        """Run two workers under one id, with the interleaving pinned.

        `gate` holds both inside their first generation, which proves both cleared the
        `RunIdInUse` check before either wrote anything. `first_done` then lets the
        "first" worker finish alone, so the "second" worker's writes land last and the
        outcome is the same on every machine.
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

    def test_both_runs_pass_the_run_id_check_that_exists_to_stop_them(self):
        results, errors = self._race()

        # FINDING (critical): one of these two SHOULD have raised RunIdInUse. The check
        # reads the store and writes a checkpoint one model call later, and nothing
        # closes that window — not a transaction, not a UNIQUE constraint, not a claim
        # row. Both callers are told their run started cleanly.
        self.assertEqual(errors, {}, "the race no longer happens — tighten this test")
        self.assertEqual(sorted(results), ["first", "second"])
        for marker, result in results.items():
            self.assertEqual(result.run_id, "R")
            self.assertEqual(result.output["ticket"], marker)

    def test_the_two_runs_weld_into_one_chain_and_one_of_them_disappears(self):
        results, errors = self._race()
        self.assertEqual(errors, {})

        store = self.open_store()
        history = store.history("R")

        # FINDING (critical): two runs of three steps each left three rows, not six.
        # `INSERT OR REPLACE` on (run_id, step) let the later writer overwrite the
        # earlier one step for step, so the store's audit trail describes one run that
        # never happened as the record of two that did.
        self.assertEqual(len(history), 3)
        self.assertEqual(store.runs(), ["R"])
        surviving = {checkpoint.state["ticket"] for checkpoint in history}
        self.assertEqual(surviving, {"second"})

        # The "first" caller holds a successful RunResult for run "R" whose every trace
        # in the store has been overwritten. Nothing told it, and nothing can tell it now.
        self.assertEqual(results["first"].output["ticket"], "first")
        self.assertNotEqual(results["first"].output, history[-1].output)

    def test_resuming_the_shared_run_id_hands_back_the_other_callers_data(self):
        """The leak, end to end: caller A resumes A's run id and gets B's output."""
        results, _ = self._race()
        store = self.open_store()

        replayed = resume(two_step_pack(), MarkerModel("unused"), "R", store)

        # FINDING (critical): "first" asked for its own run and received "second"'s
        # ticket, "second"'s generations and "second"'s output — one customer's data
        # returned as another's. `resume` cannot detect this: the chain it reads is
        # internally consistent, it is just not the chain the caller committed.
        self.assertEqual(replayed.output["ticket"], "second")
        self.assertEqual(replayed.output, results["second"].output)
        self.assertNotEqual(replayed.output, results["first"].output)

    def test_the_check_still_works_when_the_runs_do_not_overlap(self):
        """The guard is not useless — it is only unsound. Sequentially it holds."""
        store = self.open_store()
        run(two_step_pack(), MarkerModel("a"), inputs={"ticket": "a"},
            run_id="S", store=store)
        with self.assertRaises(RunIdInUse):
            run(two_step_pack(), MarkerModel("b"), inputs={"ticket": "b"},
                run_id="S", store=store)


# ------------------------------------------------------------ concurrent resume


class ConcurrentResumeOfOneRun(StoreTempDir):
    """Two supervisors retrying the same resume at once.

    This is not exotic: a retry loop plus a stuck health check produces it on the first
    bad night. `state.resume` promises idempotence — "a supervisor can retry a resume
    without paying for it twice" — and delivers it for a *finished* run only.
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

    def test_concurrent_resume_of_an_unfinished_run_executes_the_rest_twice(self):
        self._half_finished_run()

        gate = threading.Barrier(2)
        first_done = threading.Event()
        models = {}
        results = {}
        errors = {}

        def worker(marker, after):
            def go():
                store = Store(self.path)
                model = MarkerModel(marker, gate=gate, after=after)
                models[marker] = model
                try:
                    results[marker] = resume(
                        two_step_pack(), model, "R", store
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors[marker] = exc
                finally:
                    if marker == "first":
                        first_done.set()
                    store.close()
            return go

        _run_all([worker("first", None), worker("second", first_done)])

        self.assertEqual(errors, {})
        # FINDING (high): node "two" — the only node left to run — was generated twice,
        # once per resumer. `resume` re-enters `graph.run` with `resume_from` set, which
        # is the one path that skips the RunIdInUse check entirely, so there is no claim,
        # no lease and no lock between the two walks. Whatever that node does downstream
        # (charge, email, ticket) happens twice, for one run.
        self.assertEqual(len(models["first"].calls), 1)
        self.assertEqual(len(models["second"].calls), 1)
        self.assertNotEqual(results["first"].output, results["second"].output)

        store = self.open_store()
        history = store.history("R")
        self.assertEqual([c.step for c in history], [1, 2, 3])
        # Only one of the two answers survives, and the caller holding the other one was
        # told its run succeeded.
        self.assertEqual(history[-1].output["b"]["v"], "second")
        self.assertEqual(results["first"].output["b"]["v"], "first")

    def test_concurrent_resume_of_a_finished_run_is_idempotent(self):
        """The half that does hold: a finished run replays, and never calls the model."""
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


# --------------------------------------------------- sharing one Store: it cannot be


class SharingOneStoreBetweenThreads(StoreTempDir):
    """`Store` wraps one `sqlite3.connect(path)`, and that call defaults to
    `check_same_thread=True`.

    Nothing in `jig/state.py` says so — the module docstring talks about durability and
    round-tripping, `Store.__init__` takes a path and nothing else, and `graph.run`
    accepts any object with `save`. The obvious deployment (one store, a thread pool)
    therefore fails, and fails in a way that costs money.
    """

    def test_a_store_created_in_one_thread_cannot_be_used_from_another(self):
        store = self.open_store()
        failures = []

        def go():
            try:
                store.save(run_id="r", step=1, node="n", next_node=None, state={})
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        _run_all([go])

        self.assertEqual(len(failures), 1)
        # FINDING (high): a raw sqlite3.ProgrammingError, not a JigError. A caller that
        # wraps its runs in `except JigError` — the hierarchy jig documents — does not
        # catch this, and the message talks about thread ids rather than about jig.
        self.assertIsInstance(failures[0], sqlite3.ProgrammingError)
        self.assertNotIsInstance(failures[0], JigError)

    def test_reads_are_refused_across_threads_too(self):
        """So even the RunIdInUse guard, which only reads, dies on a shared store."""
        store = self.open_store()
        failures = []

        def go():
            try:
                store.latest("anything")
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        _run_all([go])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], sqlite3.ProgrammingError)

    def test_a_shared_store_kills_the_run_only_after_a_generation_is_paid_for(self):
        """The expensive shape of the same defect.

        With `run_id` supplied the guard reads the store first and dies before any model
        call. With `run_id` left to the walker there is no read, so the failure lands on
        the first `save` — after the first generation has already been issued, billed and
        thrown away, with no checkpoint to show for it.
        """
        store = self.open_store()
        model = MarkerModel("m")
        failures = []

        def go():
            try:
                run(two_step_pack(), model, inputs={"ticket": "t"}, store=store)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        _run_all([go])

        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], sqlite3.ProgrammingError)
        self.assertEqual(len(model.calls), 1, "a generation was spent")
        self.assertEqual(self.rows(), 0, "and nothing was checkpointed")

    def test_nothing_in_the_store_documents_the_thread_restriction(self):
        """A documented limitation is a design; an undocumented one is a trap.

        When jig grows either a `check_same_thread=False` connection with its own lock or
        a line of prose telling operators to open one `Store` per thread, this test is
        what should be deleted.
        """
        text = (jig.state.__doc__ or "") + (Store.__doc__ or "") + (
            Store.__init__.__doc__ or ""
        )
        lowered = text.lower()
        self.assertNotIn("thread", lowered)
        self.assertNotIn("check_same_thread", lowered)


# ------------------------------------------------------------- sqlite under load


class SqliteLockingUnderContention(StoreTempDir):
    """What actually happens when many writers hit one file.

    The good news first: sqlite3's connect() default is a 5 second busy timeout, so it
    does have one, and at the scale a jig deployment writes — a handful of small rows per
    run — contention is absorbed and nothing is lost. The bad news is everything about
    what happens when that 5 seconds is not enough.
    """

    def test_the_store_runs_on_stdlib_defaults_and_offers_no_knob(self):
        store = self.open_store()
        timeout_ms = store._connection.execute("PRAGMA busy_timeout").fetchone()[0]
        journal = store._connection.execute("PRAGMA journal_mode").fetchone()[0]

        # FINDING (medium): 5000ms is `sqlite3.connect`'s default, not a decision — and
        # the journal is the rollback journal, so a writer excludes every reader for the
        # length of its transaction. WAL would let readers through. Neither is reachable:
        # `Store(path)` takes a path and nothing else, so an operator whose store is on a
        # slow or networked filesystem has nowhere to raise the timeout.
        self.assertEqual(timeout_ms, 5000)
        self.assertEqual(journal, "delete")
        import inspect

        parameters = inspect.signature(Store.__init__).parameters
        self.assertEqual(list(parameters), ["self", "path"])

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
            # other writer's, would show here.
            self.assertEqual(
                [c.state["w"] for c in history], [index] * per_writer
            )

    def test_a_write_that_outlasts_the_busy_timeout_loses_the_checkpoint(self):
        """The failure mode the 5s default only postpones.

        The timeout is shortened on this one connection so the test costs 100ms instead
        of five seconds; the wait is the only thing scaled, the outcome is not. A long
        write elsewhere — a backup, a `history()` over a huge store, another host on an
        NFS mount — produces the same thing at the real timeout.
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
        with self.assertRaises(sqlite3.OperationalError) as caught:
            store.save(run_id="victim", step=1, node="one", next_node="two",
                       state={"work": "already done"})
        waited = time.monotonic() - started
        blocker.rollback()

        self.assertIn("locked", str(caught.exception))
        self.assertLess(waited, 2.0)
        # FINDING (medium): the run dies on a raw sqlite3.OperationalError, which is not
        # a JigError and carries no run id, no node and no retry. The node had already
        # completed and been verified; its checkpoint is simply gone, which is the one
        # outcome the whole checkpoint layer exists to prevent. `save` should retry a
        # busy database, and a give-up should arrive as a jig error naming the run.
        self.assertNotIsInstance(caught.exception, JigError)
        self.assertIsNone(store.latest("victim"))

        # Not sticky, at least: once the blocker lets go the same Store writes again, so
        # a caller that catches OperationalError itself can retry. The connection is left
        # mid-transaction by the failed INSERT, and that turns out to be harmless because
        # a deferred BEGIN that never got its write lock holds nothing.
        self.assertTrue(store._connection.in_transaction)
        store.save(run_id="victim", step=1, node="one", next_node="two", state={"a": 1})
        self.assertEqual(store.latest("victim").state, {"a": 1})


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


# ------------------------------------------------- opening a Store is itself a race


class OpeningAStoreRacesItself(unittest.TestCase):
    """`Store.__init__` does two read-then-write things before it is usable.

    Both only matter on a cold start — the first time a fleet of workers opens a store
    that does not exist yet, or the first time it opens a store file an older jig wrote.
    That is exactly the moment a deployment restarts every worker at once.
    """

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_makedirs_is_unguarded_when_two_workers_create_the_store_directory(self):
        """The losing worker's exact code path, forced rather than raced for.

        `Store.__init__` reads `os.path.isdir(directory)` and, if it is False, calls
        `os.makedirs(directory)` with no `exist_ok`. The loser of a two-worker cold start
        is the process whose `isdir` ran before the winner's `makedirs` and whose
        `makedirs` ran after it. Substituting an `os` whose `isdir` always says "not yet"
        reproduces that interleaving with no scheduling luck involved.
        """
        existing = os.path.join(self.directory, "state")
        os.makedirs(existing)  # the winner already made it
        stale = types.SimpleNamespace(
            path=types.SimpleNamespace(
                dirname=os.path.dirname,
                abspath=os.path.abspath,
                isdir=lambda path: False,  # the loser's stale read
            ),
            makedirs=os.makedirs,
        )
        real_os = jig.state.os
        jig.state.os = stale
        try:
            # FINDING (medium): FileExistsError, out of a constructor, on a perfectly
            # healthy store. `os.makedirs(directory, exist_ok=True)` is the whole fix.
            with self.assertRaises(FileExistsError):
                Store(os.path.join(existing, "runs.db"))
        finally:
            jig.state.os = real_os

    def test_racing_workers_creating_the_store_directory_fail_this_way_or_not_at_all(self):
        """The unforced version, as a check that the injection above is not fiction.

        Eight threads meet at a barrier and open the same not-yet-existing store. This
        asserts nothing about *whether* the race fires — that would be a coin flip in the
        suite — only that when it does, it is the unguarded `makedirs` and nothing else.
        """
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

        self.assertGreaterEqual(len(opened), 1, "not one worker opened the store")
        for failure in failures:
            self.assertIsInstance(failure, FileExistsError)

    def test_the_schema_migration_is_unguarded_between_two_openers(self):
        """The other read-then-write in the constructor, replayed the same way.

        `Store._migrate` reads `PRAGMA table_info` and, for every column missing, runs an
        `ALTER TABLE ... ADD COLUMN`. Two workers opening a pre-`pack_version` store file
        together both read "missing" and both run the ALTER; the second one is answered
        with `duplicate column name`. The connections below execute exactly the two
        statements `_migrate` executes, in the order two cold-starting workers produce.
        """
        self.assertIn(
            "pack_version", [name for name, _ in jig.state._ADDED_COLUMNS],
            "the migration this test replays is gone — retarget or delete it",
        )
        path = os.path.join(self.directory, "legacy.db")
        legacy = sqlite3.connect(path)
        # The table as jig shipped it before commit a055476 added the version column.
        legacy.executescript(
            "CREATE TABLE checkpoints ("
            " run_id TEXT NOT NULL, step INTEGER NOT NULL, node TEXT NOT NULL,"
            " next_node TEXT, state TEXT NOT NULL, path TEXT NOT NULL,"
            " provenance TEXT NOT NULL, failures TEXT NOT NULL, output TEXT,"
            " pack TEXT, created_at TEXT NOT NULL, PRIMARY KEY (run_id, step));"
        )
        legacy.commit()
        legacy.close()

        loser = sqlite3.connect(path)
        self.addCleanup(loser.close)
        columns = {row[1] for row in loser.execute("PRAGMA table_info(checkpoints)")}
        self.assertNotIn("pack_version", columns)  # the loser's read

        winner = Store(path)  # the winner migrates and commits
        winner.close()

        # FINDING (medium): the loser now runs the ALTER its read told it to run, and
        # dies in the constructor on a store file that is perfectly fine. The migration
        # needs to tolerate the column already being there.
        with self.assertRaises(sqlite3.OperationalError) as caught:
            loser.execute("ALTER TABLE checkpoints ADD COLUMN pack_version TEXT")
        self.assertIn("duplicate column", str(caught.exception))

    def test_racing_workers_migrating_a_legacy_store_fail_this_way_or_not_at_all(self):
        """As above: the unforced race, asserting only the shape of any failure."""
        path = os.path.join(self.directory, "legacy_race.db")
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

        self.assertGreaterEqual(len(opened), 1)
        for failure in failures:
            self.assertIsInstance(failure, sqlite3.OperationalError)
            self.assertIn("duplicate column", str(failure))
        # Whatever else happened, the migration itself succeeded exactly once.
        store = Store(path)
        self.addCleanup(store.close)
        columns = [
            row["name"] for row in
            store._connection.execute("PRAGMA table_info(checkpoints)")
        ]
        self.assertEqual(columns.count("pack_version"), 1)


# ----------------------------------------------------------------------- the soak


class TheStoreGrowsAndNothingPrunesIt(StoreTempDir):
    """Sustained runs against one store file, which is what a deployed jig does forever.

    Two separate growth problems live here: how much one run costs, and what happens to
    the file over a month of them.
    """

    def _state_bytes(self, store):
        return store._connection.execute(
            "SELECT SUM(LENGTH(state)) FROM checkpoints"
        ).fetchone()[0]

    def test_one_run_costs_bytes_quadratic_in_its_own_length(self):
        """Every checkpoint holds the *whole* state, so step N re-writes steps 1..N-1.

        A twenty-node run therefore costs roughly four times a ten-node run, not twice.
        The numbers below are exact — the model's payload is fixed — so this is a
        measurement, not a heuristic.
        """
        sizes = {}
        for length in (10, 20):
            directory = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            store = Store(os.path.join(directory, "runs.db"))
            self.addCleanup(store.close)
            run(chain_pack(length), MarkerModel("m", delay=0, payload="x" * 200),
                inputs={"ticket": "t"}, run_id="R", store=store)
            sizes[length] = self._state_bytes(store)

        # FINDING (medium): doubling the length of a run more than triples what it
        # stores. jig's own pitch is long workflows; a 200-node run stores its state 200
        # times over. A checkpoint that recorded only the node's own commit, with the
        # walker rebuilding state from the chain, would be linear.
        self.assertGreater(sizes[20], 3 * sizes[10])

    def test_a_soak_of_sequential_runs_grows_the_file_with_nothing_to_stop_it(self):
        store = self.open_store()
        model = MarkerModel("m", delay=0, payload="x" * 200)
        pack = chain_pack(3)

        for index in range(60):
            run(pack, model, inputs={"ticket": "t%d" % index},
                run_id="soak-%04d" % index, store=store)
        after_60 = self._state_bytes(store)
        for index in range(60, 120):
            run(pack, model, inputs={"ticket": "t%d" % index},
                run_id="soak-%04d" % index, store=store)
        after_120 = self._state_bytes(store)

        self.assertEqual(len(store.runs()), 120)
        self.assertGreater(after_120, after_60)

        # FINDING (medium): the only way to remove anything is `delete(run_id)`, one run
        # at a time, and the caller has to already know which ids to remove — there is no
        # retention window, no "keep the last N", no "drop everything that finished
        # before X", and `Checkpoint.created_at` is written but never used for any of it.
        for name in ("prune", "vacuum", "purge", "trim", "retain"):
            self.assertFalse(hasattr(store, name), "a prune API appeared: %s" % name)

        # And deleting every run does not give the disk back: sqlite keeps the freed
        # pages for reuse, and nothing ever runs VACUUM.
        size_before_delete = os.path.getsize(self.path)
        for run_id in store.runs():
            store.delete(run_id)
        self.assertEqual(self.rows(), 0)
        self.assertGreaterEqual(os.path.getsize(self.path), size_before_delete)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
