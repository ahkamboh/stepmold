"""T7 — checkpoint after every committed node, and resume without redoing work."""

import dataclasses
import json
import os
import shutil
import tempfile
import unittest

from jig.errors import UnknownRun
from jig.graph import run
from jig.model import FakeModel, ModelExhausted
from jig.pack import Edge, Node, Pack
from jig.state import CheckpointMismatch, Store, resume


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
        """json.loads accepts JSON's NaN extension, so a model can commit one."""
        with self.assertRaises(ValueError) as caught:
            run(three_step_pack(), FakeModel(['{"a": NaN}', '{"b": 2}', '{"c": 3}']),
                {"ticket": "t"}, run_id="nan-run", store=self.store)
        self.assertIn("state.a", str(caught.exception))
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
