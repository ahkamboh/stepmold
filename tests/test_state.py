"""T7 — checkpoint after every committed node, and resume without redoing work."""

import os
import shutil
import tempfile
import unittest

from jig.errors import UnknownRun
from jig.graph import run
from jig.model import FakeModel, ModelExhausted
from jig.pack import Edge, Node, Pack
from jig.state import Store, resume


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
