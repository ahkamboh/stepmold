"""The incident-alert triage example pack, scored end to end with no network."""

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
PACK = os.path.join(ROOT, "examples", "incident_triage")

CASE_COUNT = 13

# A p0 alert, used wherever the subject is routing rather than classification.
OUTAGE = {
    "alert_id": "INC-1050",
    "service": "edge-router",
    "severity_raw": "page",
    "message": "All regions returning 502 since the 09:15 deploy",
    "error_text": "new build panics on startup",
}


def load():
    return load_pack(PACK)


def scripted(pack):
    return resolve_model(None, pack)


def keyed(**overrides):
    """A FakeModel keyed on the node headers, ignoring which alert it is given.

    A plain-string value answers every prompt that matches its key, so these models
    are stateless and can be reused across attempts — which is exactly what a retry
    ladder needs to burn through.
    """
    script = {
        "Task: intake check": '{"alert_status": "actionable", "defect": "none"}',
        "Task: cause category": '{"cause_category": "deploy_regression", "confidence": "high"}',
        "Task: severity notes": "every region is down since the deploy",
        "Task: severity call": '{"severity": "p2", "blast_radius": "many_customers", "reason": "degraded"}',
        "Task: route and summarise": '{"owner_team": "platform", "page_now": false, "summary": "s"}',
    }
    script.update(overrides)
    return FakeModel(script)


class TestTheExampleScores(unittest.TestCase):
    def test_it_scores_thirteen_out_of_thirteen(self):
        pack = load()
        report = evaluate(pack, scripted(pack))
        self.assertEqual((report.passed, report.total), (CASE_COUNT, CASE_COUNT))
        self.assertTrue(report.passed_all)
        self.assertEqual(report.by_node, {})

    def test_the_cli_scores_it_too(self):
        completed = subprocess.run(
            [sys.executable, "-m", "stepmold", "eval", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("%d/%d" % (CASE_COUNT, CASE_COUNT), completed.stdout)

    def test_the_cli_validates_it(self):
        completed = subprocess.run(
            [sys.executable, "-m", "stepmold", "validate", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("incident_triage", completed.stdout)

    def test_the_evalset_and_the_fake_script_have_not_drifted(self):
        """Every case must be answerable, and no key may sit there answering nothing."""
        pack = load()
        with open(os.path.join(PACK, "fakes", "script.json")) as handle:
            script = json.load(handle)
        self.assertEqual(len(pack.evalset), CASE_COUNT)

        wanted = set()
        for case in pack.evalset:
            alert_id = case.input["alert_id"]
            wanted.add("Task: intake check\nAlert: %s" % alert_id)
            if case.expect["alert_status"] == "actionable":
                for header in ("Task: cause category", "Task: severity notes",
                               "Task: severity call", "Task: route and summarise"):
                    wanted.add("%s\nAlert: %s" % (header, alert_id))
        self.assertEqual(set(script), wanted)

    def test_every_case_names_the_ending_it_reaches(self):
        """The case names carry the routing contract, so assert they are honest."""
        pack = load()
        model = scripted(pack)
        for case in pack.evalset:
            result = run(pack, model, dict(case.input))
            expected_end = case.name.split()[-1]
            self.assertEqual(result.end_node, expected_end, case.name)
            self.assertEqual(result.failures, [], case.name)

    def test_the_evalset_covers_every_ending_the_graph_can_reach_on_its_own(self):
        pack = load()
        endings = set(case.name.split()[-1] for case in pack.evalset)
        self.assertEqual(endings, {"paged", "queued", "tracked", "dropped"})


class TestTheExampleShape(unittest.TestCase):
    def setUp(self):
        self.pack = load()

    def test_it_is_the_four_node_workflow_plus_its_endings(self):
        generate_nodes = [n for n in self.pack.nodes.values() if n.type == "generate"]
        self.assertEqual(
            sorted(node.name for node in generate_nodes),
            ["cause", "intake", "route", "severity"],
        )

    def test_every_generate_node_has_a_prompt_and_a_grammar(self):
        for node in self.pack.nodes.values():
            if node.type == "generate":
                self.assertTrue(node.prompt, node.name)
                self.assertTrue(node.grammar, node.name)

    def test_every_generated_field_is_an_enum_or_a_typed_scalar(self):
        """No free-string slot except the two the workflow exists to write."""
        prose = {"reason", "summary"}
        for node in self.pack.nodes.values():
            if node.type != "generate":
                continue
            for field, schema in node.grammar["properties"].items():
                if field in prose:
                    continue
                self.assertTrue(
                    "enum" in schema or schema.get("type") == "boolean",
                    "%s.%s is an unconstrained %s" % (node.name, field, schema),
                )

    def test_the_severity_node_is_two_stage_with_its_own_think_prompt(self):
        severity = self.pack.nodes["severity"]
        self.assertTrue(severity.two_stage)
        self.assertIn("severity reasoning step", severity.think_prompt)

    def test_both_invariants_are_declared_with_a_failure_edge(self):
        self.assertIn("page_now", self.pack.nodes["route"].assert_expr)
        self.assertEqual(self.pack.nodes["route"].on_fail, "needs_human")
        self.assertIn("defect", self.pack.nodes["intake"].assert_expr)
        self.assertEqual(self.pack.nodes["intake"].on_fail, "unreadable")


class TestRouting(unittest.TestCase):
    def test_a_p0_alert_pages_someone(self):
        pack = load()
        result = run(pack, scripted(pack), dict(OUTAGE))
        self.assertEqual(result.output["severity"], "p0")
        self.assertTrue(result.output["page_now"])
        self.assertEqual(result.output["owner_team"], "platform")
        self.assertEqual(result.end_node, "paged")

    def test_a_p3_alert_is_queued_instead(self):
        pack = load()
        result = run(pack, scripted(pack), {
            "alert_id": "INC-1046", "service": "etl-nightly", "severity_raw": "info",
            "message": "Nightly ETL finished 20 minutes late", "error_text": "",
        })
        self.assertEqual(result.output["severity"], "p3")
        self.assertFalse(result.output["page_now"])
        self.assertEqual(result.end_node, "queued")

    def test_a_malformed_alert_is_dropped_before_any_triage_is_spent_on_it(self):
        pack = load()
        model = scripted(pack)
        result = run(pack, model, {
            "alert_id": "INC-1052", "service": "checkout-api",
            "severity_raw": "critical", "message": "", "error_text": "",
        })
        self.assertEqual(result.end_node, "dropped")
        self.assertEqual(result.output["defect"], "no_message")
        # one generation, not five: the cheap node fired the expensive ones
        self.assertEqual(model.call_count, 1)
        self.assertNotIn("severity", result.state)

    def test_an_unfamiliar_service_still_gets_triaged_and_lands_on_unknown(self):
        pack = load()
        result = run(pack, scripted(pack), {
            "alert_id": "INC-1047", "service": "legacy-fax-bridge",
            "severity_raw": "notice",
            "message": "Fax delivery receipts are delayed on the legacy bridge",
            "error_text": "",
        })
        self.assertEqual(result.output["alert_status"], "actionable")
        self.assertEqual(result.output["owner_team"], "unknown")


class TestTheThinkStage(unittest.TestCase):
    def test_the_workflow_costs_five_model_calls_four_nodes_plus_one_think(self):
        pack = load()
        model = scripted(pack)
        run(pack, model, dict(OUTAGE))
        self.assertEqual(model.call_count, 5)
        self.assertIsNone(model.calls[2].grammar)  # the think stage is unconstrained
        self.assertIsNotNone(model.calls[3].grammar)

    def test_the_scratchpad_conditions_the_emit_and_goes_no_further(self):
        pack = load()
        model = scripted(pack)
        result = run(pack, model, dict(OUTAGE))
        notes = "Every region is returning errors"
        self.assertIn(notes, model.calls[3].prompt)   # severity's own emit sees it
        self.assertNotIn(notes, model.calls[4].prompt)  # route does not
        self.assertNotIn("scratchpad", result.state)
        self.assertNotIn(notes, json.dumps(result.state))
        self.assertNotIn(notes, json.dumps(result.output))


class TestTheInvariantsHold(unittest.TestCase):
    def test_paging_on_a_p2_is_caught_and_routed_to_a_human(self):
        """The route grammar is satisfied but the node's assert is not."""
        pack = load()
        model = keyed(**{
            "Task: route and summarise":
                '{"owner_team": "platform", "page_now": true, "summary": "s"}',
        })
        result = run(pack, model, dict(OUTAGE))
        self.assertEqual(result.end_node, "needs_human")
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].node, "route")
        self.assertEqual(result.failures[0].attempts, 3)
        self.assertNotIn("page_now", result.state)
        # what survived is what the human needs to pick it up
        self.assertEqual(result.output["severity"], "p2")
        self.assertEqual(result.output["cause_category"], "deploy_regression")

    def test_a_verdict_that_argues_with_its_own_defect_never_reaches_the_graph(self):
        """'actionable' and a named defect cannot both be true."""
        pack = load()
        model = keyed(**{
            "Task: intake check":
                '{"alert_status": "actionable", "defect": "no_message"}',
        })
        result = run(pack, model, dict(OUTAGE))
        self.assertEqual(result.end_node, "unreadable")
        self.assertEqual(result.failures[0].node, "intake")
        self.assertNotIn("alert_status", result.state)
        self.assertNotIn("defect", result.state)
        # the ladder was spent on intake alone; nothing downstream ever ran
        self.assertEqual(model.call_count, 3)
        self.assertEqual(result.output, {"alert_id": "INC-1050",
                                         "service": "edge-router"})

    def test_a_malformed_verdict_with_a_defect_is_accepted(self):
        pack = load()
        model = keyed(**{
            "Task: intake check":
                '{"alert_status": "malformed", "defect": "not_an_alert"}',
        })
        result = run(pack, model, dict(OUTAGE))
        self.assertEqual(result.end_node, "dropped")
        self.assertEqual(result.failures, [])
