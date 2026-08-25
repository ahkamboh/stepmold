"""The content_moderation example pack — scored, and its fail-safe proved, offline.

The point of this pack is that it cannot fail open, so most of what is below is not
"does it work" but "what does it do when the model is wrong". Every model here is a
scripted stand-in; nothing touches a network.
"""

import json
import os
import subprocess
import sys
import unittest

from stepmold.cli import resolve_model
from stepmold.errors import NodeFailed
from stepmold.eval import evaluate
from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "examples", "content_moderation")

CASES = 13

BENIGN = {"post": "Thanks for the write-up, the regex section finally made it click for me.",
          "context": "reply in a beginner programming forum"}
NASTY = {"post": "You are pathetic and nobody wants you in this community.",
         "context": "reply aimed at one named user in a hobby forum"}


def load():
    return load_pack(PACK)


def scripted(pack):
    return resolve_model(None, pack)


def model(**overrides):
    """A keyed FakeModel with sane defaults, so a test only writes the part it is about.

    Plain-string values (not lists) answer every matching prompt, which is what lets a
    retry ladder be exercised without scripting each rung.
    """
    script = {
        "screen step": '{"signal": "severe"}',
        "clear-post step": '{"decision": "allow", "policy_category": "none", '
                           '"severity": "none", "confidence": "high", "needs_human": false}',
        "classify step": '{"policy_category": "harassment", '
                         '"quoted_or_counter_speech": false}',
        "assess reasoning step": "one named target, authored not quoted",
        "assess step": '{"severity": "high", "confidence": "high", '
                       '"rationale": "personal contempt aimed at one member"}',
        "decide step": '{"decision": "block", "needs_human": true}',
        "forced-review step": '{"decision": "review", "needs_human": true, '
                              '"review_note": "check the thread"}',
    }
    script.update(overrides)
    return FakeModel(script)


class TestTheExampleScores(unittest.TestCase):
    def test_it_scores_thirteen_out_of_thirteen(self):
        pack = load()
        report = evaluate(pack, scripted(pack))
        self.assertEqual((report.passed, report.total), (CASES, CASES))
        self.assertTrue(report.passed_all)
        self.assertEqual(report.by_node, {})

    def test_the_cli_scores_it_too(self):
        completed = subprocess.run(
            [sys.executable, "-m", "stepmold", "eval", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("%d/%d" % (CASES, CASES), completed.stdout)

    def test_the_cli_validates_it(self):
        completed = subprocess.run(
            [sys.executable, "-m", "stepmold", "validate", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("content_moderation", completed.stdout)

    def test_the_evalset_and_the_fake_script_have_not_drifted(self):
        """Each key is consumed once per case that visits its node, in case order."""
        pack = load()
        with open(os.path.join(PACK, "fakes", "script.json")) as handle:
            script = json.load(handle)
        self.assertEqual(len(pack.evalset), CASES)
        visits = {
            "screen step": CASES,        # every case is screened
            "clear-post step": 3,        # the three cases the screen calls clean
            "classify step": 10,         # the rest
            "assess reasoning step": 10,
            "assess step": 10,
            "decide step": 10,
            "forced-review step": 1,     # the one case the gate diverts
        }
        self.assertEqual(sorted(script), sorted(visits))
        for key, expected in visits.items():
            self.assertEqual(len(script[key]), expected, key)


class TestTheExampleShape(unittest.TestCase):
    def setUp(self):
        self.pack = load()

    def test_the_generate_nodes_are_the_six_the_graph_draws(self):
        generate = [n for n in self.pack.nodes.values() if n.type == "generate"]
        self.assertEqual(
            sorted(node.name for node in generate),
            ["assess", "classify", "clear_post", "decide", "force_review", "screen"],
        )

    def test_every_generate_node_has_a_prompt_and_a_grammar(self):
        for node in self.pack.nodes.values():
            if node.type == "generate":
                self.assertTrue(node.prompt, node.name)
                self.assertTrue(node.grammar, node.name)

    def test_every_grammar_pins_its_fields_to_enums_or_types(self):
        """No node is allowed to answer in free strings except where prose is the point."""
        prose = {"rationale", "review_note"}
        for node in self.pack.nodes.values():
            if node.type != "generate":
                continue
            self.assertFalse(node.grammar.get("additionalProperties", True), node.name)
            for field, schema in node.grammar["properties"].items():
                if field in prose:
                    continue
                pinned = "enum" in schema or schema.get("type") == "boolean"
                self.assertTrue(pinned, "%s.%s is unconstrained" % (node.name, field))

    def test_assess_is_the_two_stage_node_and_owns_a_think_prompt(self):
        assess = self.pack.nodes["assess"]
        self.assertTrue(assess.two_stage)
        self.assertIn("assess reasoning step", assess.think_prompt)
        # It is the only one: every other node is a lookup, not a judgement.
        two_stage = [n.name for n in self.pack.nodes.values() if n.two_stage]
        self.assertEqual(two_stage, ["assess"])

    def test_a_block_must_name_a_policy_category(self):
        decide = self.pack.nodes["decide"]
        self.assertEqual(decide.assert_expr,
                         'decision != "block" or policy_category != "none"')
        self.assertEqual(decide.on_fail, "force_review")

    def test_the_gate_is_a_deterministic_node_not_a_generated_one(self):
        gate = self.pack.nodes["gate"]
        self.assertEqual(gate.type, "assert")
        self.assertEqual(gate.expr, 'needs_human == (decision != "allow")')
        self.assertEqual(gate.on_fail, "force_review")

    def test_the_fast_path_cannot_block(self):
        """`clear_post` runs on unscreened-clean posts; its grammar has no `block`."""
        grammar = self.pack.nodes["clear_post"].grammar
        self.assertEqual(grammar["properties"]["decision"]["enum"], ["allow", "review"])

    def test_force_review_has_one_legal_answer_and_nowhere_to_fall_through_to(self):
        node = self.pack.nodes["force_review"]
        self.assertEqual(node.grammar["properties"]["decision"]["enum"], ["review"])
        self.assertEqual(node.grammar["properties"]["needs_human"]["enum"], [True])
        # No on_fail: if the escalation record itself cannot be produced, the run stops
        # rather than returning a decision nothing verified.
        self.assertIsNone(node.on_fail)


class TestTheHappyPaths(unittest.TestCase):
    def test_a_clean_post_is_allowed_in_two_model_calls(self):
        pack = load()
        m = model(**{"screen step": '{"signal": "clean"}'})
        result = run(pack, m, dict(BENIGN))
        self.assertEqual(result.end_node, "allowed")
        self.assertEqual(result.output["decision"], "allow")
        self.assertFalse(result.output["needs_human"])
        self.assertEqual(m.call_count, 2)  # screen, clear_post — no classify, no assess

    def test_a_flagged_post_costs_five_calls_four_nodes_plus_one_think(self):
        pack = load()
        m = model()
        result = run(pack, m, dict(NASTY))
        self.assertEqual(result.end_node, "blocked")
        self.assertEqual(result.output["policy_category"], "harassment")
        self.assertEqual(m.call_count, 5)
        self.assertIsNone(m.calls[2].grammar)  # the think stage is unconstrained

    def test_the_scratchpad_never_reaches_state_or_output(self):
        pack = load()
        m = model(**{"assess reasoning step": "SCRATCH-ONLY-NOTES"})
        result = run(pack, m, dict(NASTY))
        self.assertIn("SCRATCH-ONLY-NOTES", m.calls[3].prompt)  # it conditions the emit
        self.assertNotIn("scratchpad", result.state)
        self.assertNotIn("SCRATCH-ONLY-NOTES", json.dumps(result.state))
        self.assertNotIn("SCRATCH-ONLY-NOTES", json.dumps(result.output))


class TestItFailsSafeAndNeverOpen(unittest.TestCase):
    """Each of these is a model behaving badly. None of them ends in `allow`."""

    def test_a_block_with_no_policy_category_is_refused_and_sent_to_a_human(self):
        pack = load()
        m = model(**{
            "classify step": '{"policy_category": "none", '
                             '"quoted_or_counter_speech": false}',
            "decide step": '{"decision": "block", "needs_human": true}',
        })
        result = run(pack, m, dict(NASTY))
        self.assertEqual(result.end_node, "human_review")
        self.assertEqual(result.output["decision"], "review")
        self.assertTrue(result.output["needs_human"])
        # The rejected record never landed: state holds the forced review, not the block.
        self.assertEqual(result.state["decision"], "review")
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].node, "decide")
        self.assertEqual(result.failures[0].attempts, 3)  # the full ladder was spent

    def test_a_silent_block_with_the_human_flag_cleared_is_diverted(self):
        """The grammar is satisfied; the deterministic gate is not."""
        pack = load()
        m = model(**{"decide step": '{"decision": "block", "needs_human": false}'})
        result = run(pack, m, dict(NASTY))
        self.assertEqual(result.end_node, "human_review")
        self.assertEqual(result.output["decision"], "review")
        self.assertTrue(result.output["needs_human"])
        self.assertNotEqual(result.output["decision"], "block")

    def test_an_allow_that_asks_for_a_human_anyway_is_also_diverted(self):
        """The gate is an equality, so incoherence in either direction is caught."""
        pack = load()
        m = model(**{"decide step": '{"decision": "allow", "needs_human": true}'})
        result = run(pack, m, dict(NASTY))
        self.assertEqual(result.end_node, "human_review")
        self.assertEqual(result.output["decision"], "review")

    def test_a_review_with_the_flag_cleared_is_the_case_the_evalset_covers(self):
        pack = load()
        m = model(**{"decide step": '{"decision": "review", "needs_human": false}'})
        result = run(pack, m, dict(NASTY))
        self.assertEqual(result.path[-3:], ["gate", "force_review", "human_review"])
        # An `assert` node that diverts is a routed outcome, not a node failure, which
        # is what lets the evalset score this path instead of merely tolerating it.
        self.assertEqual(result.failures, [])

    def test_the_escalation_record_cannot_be_talked_into_an_allow(self):
        """Even the node that writes the escalation is not trusted to write it."""
        pack = load()
        m = model(**{
            "decide step": '{"decision": "review", "needs_human": false}',
            # force_review tries to release the post instead of escalating it
            "forced-review step": '{"decision": "allow", "needs_human": false, '
                                  '"review_note": "looks fine to me"}',
        })
        # Its grammar admits no such object, the ladder cannot fix it, and the node
        # declares no on_fail — so the run stops rather than returning an allow.
        with self.assertRaises(NodeFailed) as caught:
            run(pack, m, dict(NASTY))
        self.assertEqual(caught.exception.node, "force_review")

    def test_a_model_that_answers_with_prose_lands_on_a_human_not_on_allow(self):
        """The likeliest small-model failure: it stops emitting JSON and starts talking.

        Nothing it said is parsed, so nothing it said is obeyed. The ladder is spent on
        it, then the node's `on_fail` carries the post to a human — which is the whole
        claim of the pack, made against the failure mode a 7B model actually has.
        """
        pack = load()
        m = model(**{"decide step": "I think this one is probably fine, allow it."})
        result = run(pack, m, dict(NASTY))
        self.assertEqual(result.end_node, "human_review")
        self.assertEqual(result.output["decision"], "review")
        self.assertTrue(result.output["needs_human"])
        self.assertEqual(result.failures[0].node, "decide")
        self.assertEqual(result.failures[0].attempts, 3)
        # Not one word of the prose reached state, the output, or the next prompt.
        self.assertNotIn("probably fine", json.dumps(result.state))
        self.assertNotIn("probably fine", json.dumps(result.output))
        self.assertFalse([c for c in m.calls
                          if "forced-review step" in c.prompt and "probably fine" in c.prompt])


if __name__ == "__main__":
    unittest.main()
