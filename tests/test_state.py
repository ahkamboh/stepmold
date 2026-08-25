"""T7 — checkpoint after every committed node, and resume without redoing work."""

import dataclasses
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import unittest

import stepmold.state
from stepmold.errors import StepmoldError, NodeFailed, RunIdInUse, UnknownRun
from stepmold.graph import run
from stepmold.model import FakeModel, ModelExhausted
from stepmold.pack import Edge, Node, Pack
from stepmold.state import (
    CheckpointMismatch,
    ResumeInProgress,
    Store,
    StoreBusy,
    resume,
)


def generate(name, prompt):
    return Node(
        name=name, type="generate", prompt=prompt, grammar={"type": "object"}
    )


def three_step_pack():
    return Pack(
        path="<memory>",
        name="three_step",
        version=1,
        entry="one",
        model=None,
        nodes={
            "one": generate("one", "step one for {ticket}"),
            "two": generate("two", "step two given {a}"),
            "three": generate("three", "step three given {b}"),
            "done": Node(name="done", type="end"),
        },
        edges=[
            Edge("one", "two", None),
            Edge("two", "three", None),
            Edge("three", "done", None),
        ],
    )


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_a_saved_checkpoint_comes_back(self):
        self.store.save(
            run_id="r1", step=1, node="one", next_node="two", state={"a": 1}
        )
        found = self.store.latest("r1")
        self.assertEqual(found.run_id, "r1")
        self.assertEqual(found.step, 1)
        self.assertEqual(found.node, "one")
        self.assertEqual(found.next_node, "two")
        self.assertEqual(found.state, {"a": 1})

    def test_latest_is_the_highest_step(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        self.store.save(run_id="r1", step=2, node="two", next_node="three", state={"a": 1, "b": 2})
        self.assertEqual(self.store.latest("r1").node, "two")
        self.assertEqual(self.store.latest("r1").state, {"a": 1, "b": 2})

    def test_latest_of_an_unknown_run_is_none(self):
        self.assertIsNone(self.store.latest("nope"))

    def test_history_is_ordered(self):
        for step, name in ((2, "two"), (1, "one"), (3, "three")):
            self.store.save(run_id="r1", step=step, node=name, next_node=None, state={})
        self.assertEqual([c.node for c in self.store.history("r1")], ["one", "two", "three"])

    def test_runs_are_listed_once_each(self):
        self.store.save(run_id="r1", step=1, node="one", next_node=None, state={})
        self.store.save(run_id="r1", step=2, node="two", next_node=None, state={})
        self.store.save(run_id="r2", step=1, node="one", next_node=None, state={})
        self.assertEqual(sorted(self.store.runs()), ["r1", "r2"])

    def test_saving_the_same_step_twice_replaces_it(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 9})
        self.assertEqual(len(self.store.history("r1")), 1)
        self.assertEqual(self.store.latest("r1").state, {"a": 9})

    def test_delete_removes_a_run(self):
        self.store.save(run_id="r1", step=1, node="one", next_node=None, state={})
        self.store.delete("r1")
        self.assertIsNone(self.store.latest("r1"))

    def test_state_that_cannot_be_serialised_fails_loudly(self):
        with self.assertRaises(TypeError):
            self.store.save(run_id="r1", step=1, node="one", next_node=None,
                            state={"bad": object()})


class TestPersistenceOnDisk(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "runs.sqlite3")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_a_checkpoint_survives_reopening_the_database(self):
        store = Store(self.path)
        store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        store.close()

        reopened = Store(self.path)
        self.assertEqual(reopened.latest("r1").state, {"a": 1})
        reopened.close()

    def test_the_database_file_is_created(self):
        store = Store(self.path)
        store.save(run_id="r1", step=1, node="one", next_node=None, state={})
        store.close()
        self.assertTrue(os.path.isfile(self.path))


class TestCheckpointingDuringARun(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def _model(self):
        return FakeModel(['{"a": 1}', '{"b": 2}', '{"c": 3}'])

    def test_every_node_leaves_a_checkpoint(self):
        run(three_step_pack(), self._model(), {"ticket": "t"},
            run_id="r1", store=self.store)
        self.assertEqual(
            [c.node for c in self.store.history("r1")],
            ["one", "two", "three", "done"],
        )

    def test_a_checkpoint_records_where_to_go_next(self):
        run(three_step_pack(), self._model(), {"ticket": "t"},
            run_id="r1", store=self.store)
        history = self.store.history("r1")
        self.assertEqual(history[0].next_node, "two")
        self.assertIsNone(history[-1].next_node)

    def test_the_final_checkpoint_holds_the_output(self):
        run(three_step_pack(), self._model(), {"ticket": "t"},
            run_id="r1", store=self.store)
        self.assertEqual(
            self.store.latest("r1").output,
            {"ticket": "t", "a": 1, "b": 2, "c": 3},
        )

    def test_a_run_without_a_store_still_works(self):
        result = run(three_step_pack(), self._model(), {"ticket": "t"})
        self.assertEqual(result.path, ["one", "two", "three", "done"])


class TestResume(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.pack = three_step_pack()

    def tearDown(self):
        self.store.close()

    def _crash_at_node_three(self):
        """Two scripted responses; the third node's call runs the model dry."""
        dying = FakeModel(['{"a": 1}', '{"b": 2}'])
        with self.assertRaises(ModelExhausted):
            run(self.pack, dying, {"ticket": "t"}, run_id="r1", store=self.store)
        return dying

    def test_the_run_is_checkpointed_up_to_the_crash(self):
        self._crash_at_node_three()
        self.assertEqual([c.node for c in self.store.history("r1")], ["one", "two"])
        self.assertEqual(self.store.latest("r1").next_node, "three")

    def test_resume_completes_the_run(self):
        self._crash_at_node_three()
        result = resume(self.pack, FakeModel(['{"c": 3}']), "r1", self.store)
        self.assertEqual(result.state, {"ticket": "t", "a": 1, "b": 2, "c": 3})
        self.assertEqual(result.end_node, "done")

    def test_resume_does_not_re_execute_earlier_nodes(self):
        self._crash_at_node_three()
        second = FakeModel(['{"c": 3}'])
        resume(self.pack, second, "r1", self.store)
        self.assertEqual(second.call_count, 1)
        self.assertIn("step three", second.calls[0].prompt)

    def test_resume_keeps_the_run_id_and_the_path_walked_so_far(self):
        self._crash_at_node_three()
        result = resume(self.pack, FakeModel(['{"c": 3}']), "r1", self.store)
        self.assertEqual(result.run_id, "r1")
        self.assertEqual(result.path, ["one", "two", "three", "done"])

    def test_resume_keeps_provenance_from_before_the_crash(self):
        self._crash_at_node_three()
        result = resume(self.pack, FakeModel(['{"c": 3}']), "r1", self.store)
        self.assertEqual(result.provenance, {"a": "one", "b": "two", "c": "three"})

    def test_resume_continues_checkpointing(self):
        self._crash_at_node_three()
        resume(self.pack, FakeModel(['{"c": 3}']), "r1", self.store)
        self.assertEqual(
            [c.node for c in self.store.history("r1")],
            ["one", "two", "three", "done"],
        )

    def test_resuming_a_finished_run_returns_its_result_without_calling_the_model(self):
        run(self.pack, FakeModel(['{"a": 1}', '{"b": 2}', '{"c": 3}']),
            {"ticket": "t"}, run_id="r1", store=self.store)
        untouched = FakeModel(["should never be used"])
        result = resume(self.pack, untouched, "r1", self.store)
        self.assertEqual(untouched.call_count, 0)
        self.assertEqual(result.output["c"], 3)

    def test_resuming_an_unknown_run_says_so(self):
        with self.assertRaises(UnknownRun) as caught:
            resume(self.pack, FakeModel(["x"]), "never-existed", self.store)
        self.assertIn("never-existed", str(caught.exception))

    def test_a_failure_recorded_before_the_crash_survives_the_resume(self):
        pack = Pack(
            path="<memory>",
            name="p",
            version=1,
            entry="one",
            model=None,
            nodes={
                "one": Node(
                    name="one",
                    type="generate",
                    prompt="one",
                    grammar={"type": "object", "required": ["a"]},
                    on_fail="two",
                ),
                "two": generate("two", "two"),
                "done": Node(name="done", type="end"),
            },
            edges=[Edge("one", "done", None), Edge("two", "done", None)],
        )
        dying = FakeModel(['{"z": 0}', '{"z": 0}', '{"z": 0}'])
        with self.assertRaises(ModelExhausted):
            run(pack, dying, {}, run_id="r2", store=self.store)
        result = resume(pack, FakeModel(['{"b": 2}']), "r2", self.store)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].node, "one")


class TestTheRoundTripContract(unittest.TestCase):
    """What comes out of a checkpoint must be what went in — or the write is refused."""

    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def _save(self, **kwargs):
        fields = dict(run_id="r1", step=1, node="one", next_node="two", state={})
        fields.update(kwargs)
        self.store.save(**fields)

    def test_json_shaped_state_comes_back_identical(self):
        state = {
            "text": "x",
            "count": 3,
            "ratio": 1.5,
            "flag": True,
            "nothing": None,
            "nested": {"list": [1, ["a", {"k": "v"}]], "empty": {}},
        }
        self._save(state=state)
        self.assertEqual(self.store.latest("r1").state, state)

    def test_a_tuple_is_refused_rather_than_returned_as_a_list(self):
        with self.assertRaises(TypeError) as caught:
            self._save(state={"pair": (1, 2)})
        self.assertIn("state.pair", str(caught.exception))
        self.assertIn("tuple", str(caught.exception))

    def test_an_integer_dict_key_is_refused_rather_than_stringified(self):
        with self.assertRaises(TypeError) as caught:
            self._save(state={"counts": {1: "one"}})
        self.assertIn("state.counts", str(caught.exception))

    def test_mixed_key_types_name_the_offending_field(self):
        # json.dumps(sort_keys=True) raises a bare "'<' not supported" TypeError here,
        # which tells an operator nothing about which node committed the value.
        with self.assertRaises(TypeError) as caught:
            self._save(state={"mixed": {"a": 1, 2: "b"}})
        self.assertIn("state.mixed", str(caught.exception))

    def test_a_set_is_refused(self):
        with self.assertRaises(TypeError) as caught:
            self._save(state={"tags": {"a", "b"}})
        self.assertIn("state.tags", str(caught.exception))

    def test_nan_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._save(state={"score": float("nan")})
        self.assertIn("state.score", str(caught.exception))

    def test_infinity_nested_in_a_list_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._save(state={"scores": [1.0, float("inf")]})
        self.assertIn("state.scores[1]", str(caught.exception))

    def test_a_refused_value_leaves_no_checkpoint_behind(self):
        with self.assertRaises(ValueError):
            self._save(state={"score": float("nan")})
        self.assertIsNone(self.store.latest("r1"))

    def test_the_output_field_is_checked_too(self):
        with self.assertRaises(ValueError):
            self._save(next_node=None, output={"score": float("-inf")})

    def test_a_circular_reference_is_refused(self):
        loop = {}
        loop["self"] = loop
        with self.assertRaises(ValueError) as caught:
            self._save(state=loop)
        self.assertIn("state.self", str(caught.exception))

    def test_a_model_supplied_nan_never_reaches_the_store(self):
        """json.loads accepts JSON's NaN extension, so a model can try to commit one.

        It no longer gets as far as the store: `grammar.validate_against` refuses a
        non-finite number where the value enters, so the retry ladder sees an ordinary
        rejection and a run that never answers otherwise ends as `NodeFailed` — a
        `StepmoldError` the walker can route — instead of a bare `ValueError` raised from
        inside the checkpoint after the node had already committed. The store's own
        refusal, tested above, is the belt to that pair of braces.
        """
        with self.assertRaises(NodeFailed) as caught:
            run(three_step_pack(), FakeModel(['{"a": NaN}'] * 3),
                {"ticket": "t"}, run_id="nan-run", store=self.store)
        self.assertIn("not a JSON number", str(caught.exception))
        self.assertIsNone(self.store.latest("nan-run"))

    def test_what_is_written_to_the_file_is_strict_json(self):
        import sqlite3 as _sqlite3

        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = os.path.join(directory, "runs.sqlite3")
        store = Store(path)
        store.save(run_id="r1", step=1, node="one", next_node=None,
                   state={"ratio": 1.5}, output={"ratio": 1.5})
        store.close()

        connection = _sqlite3.connect(path)
        try:
            columns = connection.execute(
                "SELECT state, path, provenance, failures, output FROM checkpoints"
            ).fetchall()
        finally:
            connection.close()

        def refuse(token):
            raise AssertionError("non-standard JSON constant %r in the store" % token)

        for row in columns:
            for column in row:
                if column is not None:
                    json.loads(column, parse_constant=refuse)


class TestResumeChecksTheCheckpointAgainstThePack(unittest.TestCase):
    """A run must not resume under a graph other than the one that checkpointed it."""

    def setUp(self):
        self.store = Store(":memory:")
        self.pack = three_step_pack()

    def tearDown(self):
        self.store.close()

    def _crash_at_node_three(self):
        dying = FakeModel(['{"a": 1}', '{"b": 2}'])
        with self.assertRaises(ModelExhausted):
            run(self.pack, dying, {"ticket": "t"}, run_id="r1", store=self.store)

    def test_a_pack_with_another_name_is_refused(self):
        self._crash_at_node_three()
        other = dataclasses.replace(three_step_pack(), name="something_else")
        with self.assertRaises(CheckpointMismatch) as caught:
            resume(other, FakeModel(['{"c": 3}']), "r1", self.store)
        self.assertIn("three_step", str(caught.exception))
        self.assertIn("something_else", str(caught.exception))

    def test_a_finished_run_is_not_replayed_under_another_pack(self):
        run(self.pack, FakeModel(['{"a": 1}', '{"b": 2}', '{"c": 3}']),
            {"ticket": "t"}, run_id="r1", store=self.store)
        other = dataclasses.replace(three_step_pack(), name="something_else")
        with self.assertRaises(CheckpointMismatch):
            resume(other, FakeModel(["unused"]), "r1", self.store)

    def test_the_pack_version_is_recorded_when_the_writer_supplies_it(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two",
                        state={"ticket": "t", "a": 1}, path=["one"],
                        provenance={"a": "one"}, pack=self.pack)
        found = self.store.latest("r1")
        self.assertEqual(found.pack, "three_step")
        self.assertEqual(found.pack_version, "1")

    def test_a_pack_whose_version_moved_on_is_refused(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two",
                        state={"ticket": "t", "a": 1}, path=["one"],
                        provenance={"a": "one"}, pack=self.pack)
        newer = dataclasses.replace(three_step_pack(), version=2)
        with self.assertRaises(CheckpointMismatch) as caught:
            resume(newer, FakeModel(['{"b": 2}', '{"c": 3}']), "r1", self.store)
        self.assertIn("2", str(caught.exception))

    def test_the_same_version_resumes(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two",
                        state={"ticket": "t", "a": 1}, path=["one"],
                        provenance={"a": "one"}, pack=self.pack)
        result = resume(self.pack, FakeModel(['{"b": 2}', '{"c": 3}']), "r1", self.store)
        self.assertEqual(result.end_node, "done")

    def test_a_missing_resume_point_is_refused_by_name(self):
        self._crash_at_node_three()
        shrunk = three_step_pack()
        del shrunk.nodes["three"]
        with self.assertRaises(CheckpointMismatch) as caught:
            resume(shrunk, FakeModel(['{"c": 3}']), "r1", self.store)
        self.assertIn("three", str(caught.exception))

    def test_a_node_already_walked_that_the_pack_no_longer_defines_is_refused(self):
        self._crash_at_node_three()
        renamed = three_step_pack()
        renamed.nodes["uno"] = renamed.nodes.pop("one")
        with self.assertRaises(CheckpointMismatch) as caught:
            resume(renamed, FakeModel(['{"c": 3}']), "r1", self.store)
        self.assertIn("one", str(caught.exception))

    def test_a_checkpoint_without_a_recorded_pack_still_resumes(self):
        """Checkpoints written before the identity columns existed must not crash."""
        self.store.save(run_id="r1", step=1, node="one", next_node="two",
                        state={"ticket": "t", "a": 1}, path=["one"],
                        provenance={"a": "one"})
        result = resume(self.pack, FakeModel(['{"b": 2}', '{"c": 3}']), "r1", self.store)
        self.assertEqual(result.end_node, "done")


class TestAnOlderDatabaseFile(unittest.TestCase):
    """A store file written before the identity columns existed must still open."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "runs.sqlite3")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write_old_schema(self):
        import sqlite3 as _sqlite3

        connection = _sqlite3.connect(self.path)
        connection.executescript(
            "CREATE TABLE checkpoints ("
            " run_id TEXT NOT NULL, step INTEGER NOT NULL, node TEXT NOT NULL,"
            " next_node TEXT, state TEXT NOT NULL, path TEXT NOT NULL,"
            " provenance TEXT NOT NULL, failures TEXT NOT NULL, output TEXT,"
            " pack TEXT, created_at TEXT NOT NULL, PRIMARY KEY (run_id, step));"
        )
        connection.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("r1", 1, "one", "two", json.dumps({"ticket": "t", "a": 1}),
             json.dumps(["one"]), json.dumps({"a": "one"}), json.dumps([]),
             None, "three_step", "2020-01-01T00:00:00+00:00"),
        )
        connection.commit()
        connection.close()

    def test_an_old_file_reads_without_crashing(self):
        self._write_old_schema()
        store = Store(self.path)
        self.addCleanup(store.close)
        found = store.latest("r1")
        self.assertEqual(found.pack, "three_step")
        self.assertIsNone(found.pack_version)

    def test_an_old_file_still_resumes(self):
        self._write_old_schema()
        store = Store(self.path)
        self.addCleanup(store.close)
        result = resume(three_step_pack(), FakeModel(['{"b": 2}', '{"c": 3}']),
                        "r1", store)
        self.assertEqual(result.end_node, "done")


class TestTheRunIdClaim(unittest.TestCase):
    """The first checkpoint of a run takes its id, so a second run cannot join the chain.

    `graph.run` asks the store whether the id is free and then walks — a read followed a
    model call later by a write, which two runs starting together both pass. The store
    closes that window because it is the only place that can: the claim is an insert
    against a primary key inside the transaction that writes the checkpoint.
    """

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.path = os.path.join(self.directory, "runs.sqlite3")
        self.store = Store(self.path)
        self.addCleanup(self.store.close)

    def test_a_second_writer_cannot_start_a_run_under_a_claimed_id(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        other = Store(self.path)
        self.addCleanup(other.close)
        with self.assertRaises(RunIdInUse) as caught:
            other.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 2})
        self.assertIn("r1", str(caught.exception))

    def test_the_refused_write_leaves_the_first_run_untouched(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        other = Store(self.path)
        self.addCleanup(other.close)
        with self.assertRaises(RunIdInUse):
            other.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 2})
        self.assertEqual(self.store.latest("r1").state, {"a": 1})
        self.assertEqual(len(self.store.history("r1")), 1)

    def test_the_writer_that_holds_the_run_may_rewrite_its_own_first_step(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 9})
        self.assertEqual(self.store.latest("r1").state, {"a": 9})

    def test_another_writer_may_continue_a_run_it_did_not_start(self):
        """Resume is a different Store picking up someone else's chain — the whole point.

        Only the first step fights for the id; a later one is a continuation, and
        refusing it would break the feature checkpoints exist for.
        """
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        other = Store(self.path)
        self.addCleanup(other.close)
        other.save(run_id="r1", step=2, node="two", next_node=None, state={"a": 1, "b": 2})
        self.assertEqual([c.step for c in self.store.history("r1")], [1, 2])

    def test_deleting_a_run_releases_its_id(self):
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        self.store.delete("r1")
        other = Store(self.path)
        self.addCleanup(other.close)
        other.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 2})
        self.assertEqual(other.latest("r1").state, {"a": 2})

    def test_a_fresh_run_under_a_used_id_is_refused_even_across_a_restart(self):
        """The claim is in the file, not in the process that made it."""
        self.store.save(run_id="r1", step=1, node="one", next_node="two", state={"a": 1})
        self.store.close()
        reopened = Store(self.path)
        self.addCleanup(reopened.close)
        with self.assertRaises(RunIdInUse):
            reopened.save(run_id="r1", step=1, node="one", next_node="two",
                          state={"a": 2})


class TestTheResumeLease(unittest.TestCase):
    """Only one worker at a time may walk the tail of an unfinished run."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.path = os.path.join(self.directory, "runs.sqlite3")
        self.store = Store(self.path)
        self.addCleanup(self.store.close)
        self.pack = three_step_pack()

    def _crash_at_node_three(self):
        with self.assertRaises(ModelExhausted):
            run(self.pack, FakeModel(['{"a": 1}', '{"b": 2}']), {"ticket": "t"},
                run_id="r1", store=self.store)

    def test_a_second_holder_is_refused_while_the_first_holds_it(self):
        with self.store.lease("r1"):
            with self.assertRaises(ResumeInProgress):
                with self.store.lease("r1"):
                    pass

    def test_the_lease_is_free_again_once_the_first_holder_returns(self):
        with self.store.lease("r1"):
            pass
        with self.store.lease("r1"):
            pass

    def test_the_lease_is_released_even_when_the_body_raises(self):
        with self.assertRaises(ZeroDivisionError):
            with self.store.lease("r1"):
                1 / 0
        with self.store.lease("r1"):
            pass

    def test_leases_on_different_runs_do_not_contend(self):
        with self.store.lease("r1"), self.store.lease("r2"):
            pass

    def test_an_expired_lease_may_be_taken_by_someone_else(self):
        """Nothing can release a lease for a worker that died holding it."""
        with self.store.lease("r1", seconds=-1.0):
            with self.store.lease("r1"):
                pass

    def test_resume_of_an_unfinished_run_is_refused_while_the_lease_is_held(self):
        self._crash_at_node_three()
        model = FakeModel(['{"c": 3}'])
        with self.store.lease("r1"):
            with self.assertRaises(ResumeInProgress):
                resume(self.pack, model, "r1", self.store)
        self.assertEqual(model.call_count, 0, "a refused resume still called the model")

    def test_resume_takes_and_releases_the_lease(self):
        self._crash_at_node_three()
        result = resume(self.pack, FakeModel(['{"c": 3}']), "r1", self.store)
        self.assertEqual(result.end_node, "done")
        with self.store.lease("r1"):
            pass

    def test_replaying_a_finished_run_does_not_need_the_lease(self):
        """Replay executes nothing, so serialising it would only cost availability."""
        run(self.pack, FakeModel(['{"a": 1}', '{"b": 2}', '{"c": 3}']),
            {"ticket": "t"}, run_id="r1", store=self.store)
        with self.store.lease("r1"):
            result = resume(self.pack, FakeModel(["unused"]), "r1", self.store)
        self.assertEqual(result.output["c"], 3)

    def test_a_store_without_a_lease_still_resumes(self):
        """`graph.run` takes anything with `save`; `resume` must be as forgiving."""
        self._crash_at_node_three()
        plain = _StoreWithoutALease(self.store)
        result = resume(self.pack, FakeModel(['{"c": 3}']), "r1", plain)
        self.assertEqual(result.end_node, "done")


class _StoreWithoutALease:
    """A store from before leases existed: `save` and `latest`, and nothing else."""

    def __init__(self, store):
        self._store = store

    def save(self, **kwargs):
        return self._store.save(**kwargs)

    def latest(self, run_id):
        return self._store.latest(run_id)


class TestCheckpointsRecordOnlyWhatChanged(unittest.TestCase):
    """State, path and provenance are stored as deltas — and must come back whole.

    Repeating all three in every checkpoint made an N-node run cost O(N^2) bytes, which
    is precisely the shape stepmold exists to run. What a caller sees is unchanged: `latest`
    and `history` hand back the same `Checkpoint` objects they always did.
    """

    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def _save(self, step, state, path=None, provenance=None):
        self.store.save(run_id="r1", step=step, node="n%d" % step, next_node="next",
                        state=state, path=path, provenance=provenance)

    def test_a_growing_state_round_trips_at_every_step(self):
        expected = []
        state = {}
        for step in range(1, 21):
            state = dict(state, **{"k%d" % step: "v" * 50})
            expected.append(state)
            self._save(step, state)
        self.assertEqual([c.state for c in self.store.history("r1")], expected)
        self.assertEqual(self.store.latest("r1").state, expected[-1])

    def test_a_removed_key_is_removed_and_not_carried_forward(self):
        self._save(1, {"a": 1, "b": 2})
        self._save(2, {"a": 1})
        self.assertEqual(self.store.latest("r1").state, {"a": 1})
        self.assertEqual([c.state for c in self.store.history("r1")],
                         [{"a": 1, "b": 2}, {"a": 1}])

    def test_a_value_that_changes_type_between_steps_keeps_its_type(self):
        """`True == 1` in Python, so equality alone would call this unchanged."""
        self._save(1, {"flag": True})
        self._save(2, {"flag": 1})
        self._save(3, {"flag": 1.0})
        types = [type(c.state["flag"]) for c in self.store.history("r1")]
        self.assertEqual(types, [bool, int, float])

    def test_path_and_provenance_round_trip_too(self):
        self._save(1, {"a": 1}, path=["one"], provenance={"a": "one"})
        self._save(2, {"a": 1, "b": 2}, path=["one", "two"],
                   provenance={"a": "one", "b": "two"})
        history = self.store.history("r1")
        self.assertEqual([c.path for c in history], [["one"], ["one", "two"]])
        self.assertEqual(history[-1].provenance, {"a": "one", "b": "two"})

    def test_a_path_that_diverges_is_not_treated_as_a_suffix(self):
        self._save(1, {}, path=["one", "two", "three"])
        self._save(2, {}, path=["one", "other"])
        self.assertEqual([c.path for c in self.store.history("r1")],
                         [["one", "two", "three"], ["one", "other"]])

    def test_a_long_run_costs_bytes_linear_in_its_length(self):
        sizes = {}
        for length in (10, 20):
            store = Store(":memory:")
            self.addCleanup(store.close)
            state = {}
            for step in range(1, length + 1):
                state = dict(state, **{"k%d" % step: "v" * 200})
                store.save(run_id="r", step=step, node="n", next_node="m", state=state,
                           path=["n%d" % s for s in range(step)],
                           provenance={"k%d" % s: "n%d" % s for s in range(step)})
            sizes[length] = store._connection.execute(
                "SELECT SUM(LENGTH(state) + LENGTH(path) + LENGTH(provenance))"
                " FROM checkpoints"
            ).fetchone()[0]
        self.assertLess(sizes[20], 2.2 * sizes[10])

    def test_rebuilding_a_state_never_walks_the_whole_chain_of_a_rewritten_state(self):
        """A node that replaces state wholesale gets snapshots, not a chain of deltas.

        The snapshot rule is measured in bytes: a delta that saves nothing is not worth
        the hop it costs every later read.
        """
        for step in range(1, 21):
            self._save(step, {"only": "v%d" % step + "x" * 200})
        kinds = [
            row[0] for row in self.store._connection.execute(
                "SELECT state_kind FROM checkpoints WHERE run_id = 'r1' ORDER BY step"
            )
        ]
        self.assertEqual(kinds, ["full"] * 20)
        self.assertEqual(self.store.latest("r1").state["only"], "v20" + "x" * 200)

    def test_what_is_written_is_still_strict_json_in_every_column(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = os.path.join(directory, "runs.sqlite3")
        store = Store(path)
        for step in range(1, 6):
            store.save(run_id="r1", step=step, node="n", next_node="m",
                       state={"k%d" % step: step}, path=["n%d" % step],
                       provenance={"k%d" % step: "n"})
        store.close()

        connection = sqlite3.connect(path)
        try:
            rows = connection.execute(
                "SELECT state, path, provenance, failures FROM checkpoints"
            ).fetchall()
        finally:
            connection.close()

        def refuse(token):
            raise AssertionError("non-standard JSON constant %r in the store" % token)

        for row in rows:
            for column in row:
                json.loads(column, parse_constant=refuse)


class TestTheStoreUnderContention(unittest.TestCase):
    """The store is opened for a fleet, not for one in-process caller."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.path = os.path.join(self.directory, "runs.sqlite3")

    def test_a_store_can_be_written_from_a_thread_that_did_not_open_it(self):
        store = Store(self.path)
        self.addCleanup(store.close)
        failures = []

        def go():
            try:
                store.save(run_id="r1", step=1, node="one", next_node=None,
                           state={"a": 1})
            except BaseException as exc:  # noqa: BLE001 - the test is the report
                failures.append(exc)

        thread = threading.Thread(target=go)
        thread.start()
        thread.join(timeout=5)
        self.assertEqual([repr(f) for f in failures], [])
        self.assertEqual(store.latest("r1").state, {"a": 1})

    def test_the_busy_timeout_is_stepmolds_own_and_can_be_raised(self):
        store = Store(self.path)
        self.addCleanup(store.close)
        self.assertEqual(
            store._connection.execute("PRAGMA busy_timeout").fetchone()[0],
            int(stepmold.state.DEFAULT_TIMEOUT * 1000),
        )
        slow = Store(self.path, timeout=0.5)
        self.addCleanup(slow.close)
        self.assertEqual(
            slow._connection.execute("PRAGMA busy_timeout").fetchone()[0], 500
        )

    def test_a_lock_that_outlasts_the_timeout_is_a_stepmold_error(self):
        store = Store(self.path, timeout=0.1)
        self.addCleanup(store.close)
        blocker = sqlite3.connect(self.path)
        self.addCleanup(blocker.close)
        blocker.execute("BEGIN EXCLUSIVE")
        blocker.execute(
            "INSERT INTO checkpoints (run_id, step, node, next_node, state, path,"
            " provenance, failures, created_at) VALUES"
            " ('b', 1, 'n', NULL, '{}', '[]', '{}', '[]', 'now')"
        )
        with self.assertRaises(StoreBusy) as caught:
            store.save(run_id="r1", step=1, node="one", next_node=None, state={})
        blocker.rollback()
        self.assertIsInstance(caught.exception, StepmoldError)
        self.assertIn("r1", str(caught.exception))

    def test_the_store_directory_is_created_even_if_it_already_exists(self):
        nested = os.path.join(self.directory, "a", "b", "runs.sqlite3")
        first = Store(nested)
        first.close()
        second = Store(nested)
        second.close()
        self.assertTrue(os.path.isfile(nested))

    def test_reopening_a_store_migrates_it_only_once(self):
        Store(self.path).close()
        store = Store(self.path)
        self.addCleanup(store.close)
        store._migrate()  # the loser of a cold-start race, replayed
        columns = [
            row["name"] for row in
            store._connection.execute("PRAGMA table_info(checkpoints)")
        ]
        self.assertEqual(columns.count("pack_version"), 1)
        self.assertEqual(columns.count("state_kind"), 1)


class TestRetention(unittest.TestCase):
    """Nothing is dropped unless an operator asks — and then whole runs, never rows."""

    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def _finished(self, run_id):
        self.store.save(run_id=run_id, step=1, node="one", next_node="two",
                        state={"a": 1})
        self.store.save(run_id=run_id, step=2, node="two", next_node=None,
                        state={"a": 1, "b": 2}, output={"b": 2})

    def test_prune_with_no_policy_deletes_nothing(self):
        self._finished("r1")
        self.assertEqual(self.store.prune(), [])
        self.assertEqual(self.store.runs(), ["r1"])

    def test_prune_keeps_the_last_n_finished_runs(self):
        for index in range(5):
            self._finished("r%d" % index)
        dropped = self.store.prune(keep_last=2)
        self.assertEqual(dropped, ["r0", "r1", "r2"])
        self.assertEqual(sorted(self.store.runs()), ["r3", "r4"])

    def test_prune_never_drops_an_unfinished_run(self):
        self._finished("done")
        self.store.save(run_id="live", step=1, node="one", next_node="two", state={})
        self.assertEqual(self.store.prune(keep_last=0), ["done"])
        self.assertEqual(self.store.runs(), ["live"])

    def test_prune_by_date_uses_the_last_checkpoint(self):
        self._finished("r1")
        self.assertEqual(self.store.prune(before="2000-01-01T00:00:00+00:00"), [])
        self.assertEqual(self.store.prune(before="2999-01-01T00:00:00+00:00"), ["r1"])
        self.assertEqual(self.store.runs(), [])

    def test_a_pruned_run_id_can_be_used_again(self):
        self._finished("r1")
        self.store.prune(keep_last=0)
        self.store.save(run_id="r1", step=1, node="one", next_node=None, state={"a": 9})
        self.assertEqual(self.store.latest("r1").state, {"a": 9})
