"""The meeting_actions example pack, scored end to end with no network.

Mirrors tests/test_example.py for the second shipped pack. The interesting
difference is the branch: `assign` runs only when the notes left work unowned,
so this pack's scripted model is not twelve-of-everything.
"""

import json
import os
import subprocess
import sys
import unittest

from jig.cli import resolve_model
from jig.eval import evaluate
from jig.graph import run
from jig.model import FakeModel
from jig.pack import load_pack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "examples", "meeting_actions")

# How many of the twelve evalset cases reach each node. Everything reaches every
# node except `assign`, which four unowned-work cases reach and eight skip.
VISITS = {
    "attendees step": 12,
    "actions reasoning step": 12,
    "actions step": 12,
    "assign step": 4,
    "decisions step": 12,
    "followup step": 12,
}

NOTES = ("Standup, Tuesday. Ravi: shipped the export fix, picking up the CSV importer "
         "today. Sara: still blocked on the staging cert, will ping infra this morning. "
         "Tom is out sick. Decision: we hold the 2.1 release until the cert is sorted.")


def load():
    return load_pack(PACK)


def scripted(pack):
    return resolve_model(None, pack)


def keyed(**overrides):
    """A model keyed by node, so a test can bend one node and keep the rest sane."""
    script = {
        "attendees step":
            '{"attendees": ["Maya"], "lead": "Maya", "meeting_kind": "planning"}',
        "actions reasoning step": "Maya said she would book time with legal.",
        "actions step":
            '{"action_items": [{"task": "Book time with legal", "owner": "Maya",'
            ' "owner_source": "explicit", "due_bucket": "next_week", "priority": "p2"}],'
            ' "action_count": 1, "unowned": "none"}',
        "assign step": '{"fallback_owner": "Maya", "fallback_basis": "meeting_lead"}',
        "decisions step":
            '{"decisions": [{"decision": "Rewrite onboarding this quarter",'
            ' "firmness": "final"}], "decision_count": 1}',
        "followup step":
            '{"follow_up_needed": false, "follow_up_when": "none",'
            ' "follow_up_reason": "Not needed."}',
    }
    script.update(overrides)
    return FakeModel(script)


class TestTheExampleScores(unittest.TestCase):
    def test_it_scores_twelve_out_of_twelve(self):
        pack = load()
        report = evaluate(pack, scripted(pack))
        self.assertEqual((report.passed, report.total), (12, 12))
        self.assertTrue(report.passed_all)
        self.assertEqual(report.by_node, {})

    def test_the_cli_validates_and_scores_it(self):
        for argv, expected in (
            (["validate", PACK], "meeting_actions v1"),
            (["eval", PACK], "12/12"),
        ):
            completed = subprocess.run(
                [sys.executable, "-m", "jig"] + argv,
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(expected, completed.stdout)

    def test_the_evalset_and_the_fake_script_have_not_drifted(self):
        pack = load()
        with open(os.path.join(PACK, "fakes", "script.json")) as handle:
            script = json.load(handle)
        self.assertEqual(len(pack.evalset), 12)
        self.assertEqual(sorted(script), sorted(VISITS))
        for key, responses in script.items():
            self.assertEqual(len(responses), VISITS[key],
                             "%s has %d responses" % (key, len(responses)))

    def test_every_case_expects_the_fields_the_workflow_exists_to_produce(self):
        for case in load().evalset:
            for field in ("meeting_kind", "action_count", "unowned",
                          "decision_count", "follow_up_needed", "follow_up_when"):
                self.assertIn(field, case.expect, case.name)


class TestTheExampleShape(unittest.TestCase):
    def setUp(self):
        self.pack = load()

    def test_it_is_the_five_node_workflow_plus_its_endings(self):
        generate_nodes = [n for n in self.pack.nodes.values() if n.type == "generate"]
        self.assertEqual(
            sorted(node.name for node in generate_nodes),
            ["actions", "assign", "attendees", "decisions", "followup"],
        )
        self.assertEqual(
            sorted(n.name for n in self.pack.nodes.values() if n.type == "end"),
            ["needs_review", "tracked"],
        )

    def test_every_generate_node_has_a_prompt_and_a_closed_grammar(self):
        for node in self.pack.nodes.values():
            if node.type != "generate":
                continue
            self.assertTrue(node.prompt, node.name)
            self.assertTrue(node.grammar, node.name)
            self.assertIs(node.grammar.get("additionalProperties"), False, node.name)

    def test_the_typed_fields_are_enums_rather_than_free_strings(self):
        """A slot with a fixed set of answers is a slot a small model cannot fumble."""
        enums = {}

        def walk(schema, path):
            if "enum" in schema:
                enums[path] = schema["enum"]
            for name, sub in (schema.get("properties") or {}).items():
                walk(sub, "%s.%s" % (path, name) if path else name)
            if schema.get("items"):
                walk(schema["items"], path + "[]")

        for node in self.pack.nodes.values():
            if node.type == "generate":
                walk(node.grammar, node.name)
        self.assertIn("actions.action_items[].priority", enums)
        self.assertIn("actions.action_items[].due_bucket", enums)
        self.assertIn("actions.action_items[].owner_source", enums)
        self.assertEqual(enums["actions.action_items[].owner_source"],
                         ["explicit", "inferred", "unassigned"])
        self.assertIn("assign.fallback_basis", enums)
        self.assertIn("attendees.meeting_kind", enums)

    def test_only_the_extraction_node_pays_for_a_think_stage(self):
        """Reasoning is bought where notes are prose, not where state is already typed."""
        two_stage = [n.name for n in self.pack.nodes.values() if n.two_stage]
        self.assertEqual(two_stage, ["actions"])
        self.assertIn("reasoning step", self.pack.nodes["actions"].think_prompt)

    def test_the_nodes_that_can_contradict_themselves_declare_an_invariant(self):
        for name in ("actions", "assign", "decisions", "followup"):
            node = self.pack.nodes[name]
            self.assertTrue(node.assert_expr, name)
            self.assertEqual(node.on_fail, "needs_review", name)

    def test_the_branch_is_declared_before_the_fallthrough(self):
        outgoing = self.pack.edges_from("actions")
        self.assertEqual([e.target for e in outgoing], ["assign", "decisions"])
        self.assertEqual(outgoing[0].when, {"unowned": "some"})
        self.assertIsNone(outgoing[1].when)


class TestRunningOneMeeting(unittest.TestCase):
    def test_the_owned_path_skips_the_assign_node(self):
        pack = load()
        result = run(pack, scripted(pack), {"notes": NOTES})
        self.assertEqual(result.end_node, "tracked")
        self.assertNotIn("assign", result.path)
        self.assertNotIn("fallback_owner", result.output)
        self.assertEqual(result.output["action_count"], 2)
        self.assertEqual(result.output["unowned"], "none")
        self.assertEqual(result.output["action_items"][0]["owner"], "Ravi")
        self.assertEqual(result.output["action_items"][0]["owner_source"], "explicit")
        self.assertEqual(result.output["action_items"][1]["priority"], "p1")
        self.assertEqual(result.output["follow_up_needed"], False)

    def test_unowned_work_takes_the_conditional_edge_to_assign(self):
        pack = load()
        model = keyed(**{
            "actions step":
                '{"action_items": [{"task": "Write the migration doc", "owner": null,'
                ' "owner_source": "unassigned", "due_bucket": "no_date",'
                ' "priority": "p2"}], "action_count": 1, "unowned": "some"}',
        })
        result = run(pack, model, {"notes": "someone should write the migration doc"})
        self.assertIn("assign", result.path)
        self.assertEqual(result.end_node, "tracked")
        self.assertEqual(result.output["fallback_owner"], "Maya")
        self.assertEqual(result.output["fallback_basis"], "meeting_lead")

    def test_the_workflow_costs_five_model_calls_five_nodes_minus_a_skip_plus_a_think(self):
        pack = load()
        model = scripted(pack)
        run(pack, model, {"notes": NOTES})
        self.assertEqual(model.call_count, 5)
        self.assertIsNone(model.calls[1].grammar)  # the think stage is unconstrained
        self.assertIsNotNone(model.calls[2].grammar)

    def test_the_scratchpad_conditions_the_emit_and_stops_there(self):
        pack = load()
        model = scripted(pack)
        result = run(pack, model, {"notes": NOTES})
        notes_from_thinking = "picking up the CSV importer today"
        self.assertIn(notes_from_thinking, model.calls[2].prompt)
        self.assertNotIn("scratchpad", result.state)
        self.assertNotIn("made you call it a commitment", json.dumps(result.state))
        self.assertNotIn("made you call it a commitment", json.dumps(result.output))

    def test_a_meeting_that_decided_nothing_still_produces_empty_lists(self):
        pack = load()
        model = keyed(**{
            "actions step":
                '{"action_items": [], "action_count": 0, "unowned": "none"}',
            "decisions step": '{"decisions": [], "decision_count": 0}',
        })
        result = run(pack, model, {"notes": "all-hands, no commitments"})
        self.assertEqual(result.output["action_items"], [])
        self.assertEqual(result.output["decisions"], [])
        self.assertEqual(result.end_node, "tracked")


class TestTheInvariantsHold(unittest.TestCase):
    """Each assert is the deterministic half of a node a small model gets wrong."""

    def divert(self, **overrides):
        pack = load()
        result = run(pack, keyed(**overrides), {"notes": "n"})
        self.assertEqual(result.end_node, "needs_review")
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].attempts, 3)
        return result

    def test_a_count_that_does_not_match_its_own_list_is_caught(self):
        result = self.divert(**{
            "actions step":
                '{"action_items": [{"task": "Book time with legal", "owner": "Maya",'
                ' "owner_source": "explicit", "due_bucket": "next_week",'
                ' "priority": "p2"}], "action_count": 3, "unowned": "none"}',
        })
        self.assertEqual(result.failures[0].node, "actions")
        self.assertNotIn("action_items", result.state)
        self.assertNotIn("action_count", result.state)
        # what was verified before the failure still reaches the human
        self.assertEqual(result.output["attendees"], ["Maya"])

    def test_unowned_work_in_an_empty_list_is_caught(self):
        result = self.divert(**{
            "actions step":
                '{"action_items": [], "action_count": 0, "unowned": "some"}',
        })
        self.assertEqual(result.failures[0].node, "actions")

    def test_a_named_fallback_owner_on_an_unclear_basis_is_caught(self):
        result = self.divert(**{
            "actions step":
                '{"action_items": [{"task": "Write the migration doc", "owner": null,'
                ' "owner_source": "unassigned", "due_bucket": "no_date",'
                ' "priority": "p2"}], "action_count": 1, "unowned": "some"}',
            "assign step": '{"fallback_owner": "Maya", "fallback_basis": "unclear"}',
        })
        self.assertEqual(result.failures[0].node, "assign")
        self.assertNotIn("fallback_owner", result.state)

    def test_a_follow_up_that_is_not_needed_but_has_a_date_is_caught(self):
        result = self.divert(**{
            "followup step":
                '{"follow_up_needed": false, "follow_up_when": "next_week",'
                ' "follow_up_reason": "Not needed."}',
        })
        self.assertEqual(result.failures[0].node, "followup")
        self.assertNotIn("follow_up_needed", result.state)

    def test_an_answer_outside_the_enum_never_reaches_state(self):
        """The grammar half of verify-before-commit, on the same pack."""
        result = self.divert(**{
            "actions step":
                '{"action_items": [{"task": "Book time", "owner": "Maya",'
                ' "owner_source": "explicit", "due_bucket": "someday",'
                ' "priority": "p2"}], "action_count": 1, "unowned": "none"}',
        })
        self.assertEqual(result.failures[0].node, "actions")
        self.assertNotIn("action_items", result.state)


if __name__ == "__main__":
    unittest.main()
