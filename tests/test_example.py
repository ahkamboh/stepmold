"""T10 — the shipped example pack, scored end to end with no network."""

import json
import os
import subprocess
import sys
import unittest

from stepmold.cli import resolve_model
from stepmold.eval import evaluate
from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "examples", "support_triage")


def load():
    return load_pack(PACK)


def scripted(pack):
    return resolve_model(None, pack)


class TestTheExampleScores(unittest.TestCase):
    def test_it_scores_twelve_out_of_twelve(self):
        pack = load()
        report = evaluate(pack, scripted(pack))
        self.assertEqual((report.passed, report.total), (12, 12))
        self.assertTrue(report.passed_all)
        self.assertEqual(report.by_node, {})

    def test_the_cli_scores_it_too(self):
        completed = subprocess.run(
            [sys.executable, "-m", "stepmold", "eval", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("12/12", completed.stdout)

    def test_the_evalset_and_the_fake_script_have_not_drifted(self):
        pack = load()
        with open(os.path.join(PACK, "fakes", "script.json")) as handle:
            script = json.load(handle)
        self.assertEqual(len(pack.evalset), 12)
        for key, responses in script.items():
            self.assertEqual(len(responses), 12, "%s has %d responses" % (key, len(responses)))


class TestTheExampleShape(unittest.TestCase):
    def setUp(self):
        self.pack = load()

    def test_it_is_the_four_node_workflow_plus_its_endings(self):
        generate_nodes = [n for n in self.pack.nodes.values() if n.type == "generate"]
        self.assertEqual(
            sorted(node.name for node in generate_nodes),
            ["classify", "emit", "extract", "priority"],
        )

    def test_every_generate_node_has_a_prompt_and_a_grammar(self):
        for node in self.pack.nodes.values():
            if node.type == "generate":
                self.assertTrue(node.prompt, node.name)
                self.assertTrue(node.grammar, node.name)

    def test_the_priority_node_is_two_stage_with_its_own_think_prompt(self):
        priority = self.pack.nodes["priority"]
        self.assertTrue(priority.two_stage)
        self.assertIn("reasoning step", priority.think_prompt)

    def test_the_emit_node_declares_a_deterministic_invariant(self):
        self.assertIn("escalate", self.pack.nodes["emit"].assert_expr)
        self.assertEqual(self.pack.nodes["emit"].on_fail, "needs_human")


class TestRunningOneTicket(unittest.TestCase):
    def test_a_billing_ticket_routes_to_billing_ops(self):
        pack = load()
        result = run(pack, scripted(pack),
                     {"ticket": "I was charged twice for order A-1001, $49.99 both times."})
        self.assertEqual(result.output["category"], "billing")
        self.assertEqual(result.output["order_id"], "A-1001")
        self.assertEqual(result.output["queue"], "billing-ops")
        self.assertEqual(result.output["escalate"], False)
        self.assertEqual(result.end_node, "done")

    def test_the_workflow_costs_five_model_calls_four_nodes_plus_one_think(self):
        pack = load()
        model = scripted(pack)
        run(pack, model, {"ticket": "I was charged twice for order A-1001, $49.99 both times."})
        self.assertEqual(model.call_count, 5)
        self.assertIsNone(model.calls[2].grammar)  # the think stage is unconstrained

    def test_the_scratchpad_never_reaches_the_output(self):
        pack = load()
        model = scripted(pack)
        result = run(pack, model, {"ticket": "I was charged twice for order A-1001, $49.99 both times."})
        # the think output conditions the emit prompt (calls[3]) and stops there
        self.assertIn("Nothing is down", model.calls[3].prompt)
        self.assertNotIn("scratchpad", result.state)
        self.assertNotIn("Nothing is down", json.dumps(result.state))
        self.assertNotIn("Nothing is down", json.dumps(result.output))

    def test_a_p0_ticket_takes_the_escalation_edge(self):
        """Routing is the subject here, so the model is keyed rather than positional."""
        pack = load()
        model = FakeModel(
            {
                "classify step": '{"category": "technical"}',
                "extract step": '{"order_id": null, "amount_usd": null, "sentiment": "angry"}',
                "priority reasoning step": "their whole site is down",
                "priority step": '{"priority": "p0", "reason": "production is down"}',
                "emit step": '{"queue": "eng-support", "summary": "site down", "escalate": true}',
            }
        )
        result = run(pack, model, {"ticket": "Our entire production site is down."})
        self.assertEqual(result.output["priority"], "p0")
        self.assertTrue(result.output["escalate"])
        self.assertEqual(result.end_node, "escalated")


class TestTheInvariantHolds(unittest.TestCase):
    def test_an_inconsistent_escalate_flag_is_caught_and_routed_to_a_human(self):
        """The emit grammar is satisfied but the node's assert is not."""
        pack = load()
        model = FakeModel(
            {
                "classify step": '{"category": "billing"}',
                "extract step": '{"order_id": null, "amount_usd": null, "sentiment": "calm"}',
                "priority reasoning step": "some notes",
                "priority step": '{"priority": "p2", "reason": "not urgent"}',
                # valid against the schema, but escalate is true on a p2 ticket
                "emit step": '{"queue": "general", "summary": "s", "escalate": true}',
            }
        )
        result = run(pack, model, {"ticket": "how do I export invoices?"})
        self.assertEqual(result.end_node, "needs_human")
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].node, "emit")
        self.assertEqual(result.failures[0].attempts, 3)
        self.assertNotIn("escalate", result.state)
