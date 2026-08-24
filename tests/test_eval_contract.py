"""The evalset is the contract — so it has to be able to state the whole contract.

Two holes made "N/N cases passed" mean less than it sounds like, both found by agents
authoring real packs against the docs:

1. A case could not say which end node it reached. Inverting an incident-triage pack's
   entire routing policy — p0 outages queued, p3 noise paging someone at 3am — still
   scored 13/13. `expect` compares field values, and the branch a run took is not a field.

2. A case could not cover an `on_fail` path. `_run_case` failed any case whose run
   recorded a failure, so the rescue path a pack declares could never be a passing
   expectation. Authors covered it with hand-written unit tests instead, which is exactly
   the duplication the evalset exists to remove.

Both are tested here against sabotage, not by inspection.
"""

import json
import os
import shutil
import tempfile
import unittest

from jig.eval import evaluate
from jig.model import FakeModel
from jig.pack import PackError, load_pack


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"severity": {"type": "string", "enum": ["p0", "p3"]}},
    "required": ["severity"],
    "additionalProperties": False,
}

# A two-way graph: severity p0 pages a human, anything else is queued. The two endings
# project the same field, so nothing but the ending itself distinguishes them — which is
# precisely the case the old contract could not express.
GRAPH = """
nodes:
  classify:
    type: generate
    output: null
    retries: 0
  paged:
    type: end
  queued:
    type: end
edges:
  - from: classify
    to: paged
    when:
      severity: p0
  - from: classify
    to: queued
"""


def _write(root, graph=GRAPH, cases=(), script=None):
    os.makedirs(os.path.join(root, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(root, "grammars"), exist_ok=True)
    with open(os.path.join(root, "manifest.yaml"), "w") as handle:
        handle.write("name: routing\nversion: 1\nentry: classify\n")
    with open(os.path.join(root, "graph.yaml"), "w") as handle:
        handle.write(graph)
    with open(os.path.join(root, "grammars", "classify.json"), "w") as handle:
        json.dump(ROUTE_SCHEMA, handle)
    with open(os.path.join(root, "prompts", "classify.txt"), "w") as handle:
        handle.write("Classify: {alert}\n")
    with open(os.path.join(root, "evalset.jsonl"), "w") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    return root


class ACaseCanAssertWhichEndingItReached(unittest.TestCase):
    """The hole that let a policy inversion score 13/13."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _pack(self, graph=GRAPH):
        cases = [
            {"name": "outage", "input": {"alert": "site down"},
             "expect": {"severity": "p0"}, "end": "paged"},
            {"name": "noise", "input": {"alert": "disk 61% full"},
             "expect": {"severity": "p3"}, "end": "queued"},
        ]
        _write(self.root, graph=graph, cases=cases)
        return load_pack(self.root)

    def _model(self):
        return FakeModel(['{"severity": "p0"}', '{"severity": "p3"}'])

    def test_the_correct_routing_passes(self):
        report = evaluate(self._pack(), self._model())
        self.assertEqual(report.passed, 2, _explain(report))

    def test_inverting_the_routing_policy_now_FAILS(self):
        """The whole point. Before this, an inverted policy still scored full marks."""
        inverted = GRAPH.replace("      severity: p0", "      severity: p3")
        report = evaluate(self._pack(graph=inverted), self._model())
        self.assertEqual(report.passed, 0, _explain(report))

    def test_the_failure_names_the_ending_that_was_expected(self):
        inverted = GRAPH.replace("      severity: p0", "      severity: p3")
        report = evaluate(self._pack(graph=inverted), self._model())
        rendered = _explain(report)
        self.assertIn("paged", rendered)
        self.assertIn("queued", rendered)

    def test_a_case_without_an_end_still_scores_on_fields_alone(self):
        """`end` is optional — every pack written before it must keep working."""
        cases = [{"name": "outage", "input": {"alert": "x"}, "expect": {"severity": "p0"}}]
        _write(self.root, cases=cases)
        report = evaluate(load_pack(self.root), FakeModel(['{"severity": "p0"}']))
        self.assertEqual(report.passed, 1, _explain(report))

    def test_an_end_naming_a_node_that_is_not_an_ending_is_refused_at_load(self):
        """A typo must fail loudly at load, not silently never match."""
        cases = [{"input": {"alert": "x"}, "expect": {}, "end": "classify"}]
        _write(self.root, cases=cases)
        with self.assertRaises(PackError):
            load_pack(self.root)

    def test_an_end_naming_a_node_that_does_not_exist_is_refused_at_load(self):
        cases = [{"input": {"alert": "x"}, "expect": {}, "end": "nowhere"}]
        _write(self.root, cases=cases)
        with self.assertRaises(PackError):
            load_pack(self.root)


RESCUE_GRAPH = """
nodes:
  classify:
    type: generate
    output: null
    retries: 0
    on_fail: needs_human
  paged:
    type: end
  needs_human:
    type: end
edges:
  - from: classify
    to: paged
"""


class ACaseCanCoverAnOnFailPath(unittest.TestCase):
    """A declared rescue path must be expressible as a passing expectation.

    Without this, the one path a pack author most wants to prove — what happens when the
    model cannot do the job — is the one path the evalset cannot score.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _pack(self, cases):
        _write(self.root, graph=RESCUE_GRAPH, cases=cases)
        return load_pack(self.root)

    def test_a_case_that_declares_rescued_passes_when_the_ladder_burns(self):
        pack = self._pack([
            {"name": "garbage in", "input": {"alert": "???"}, "expect": {},
             "end": "needs_human", "rescued": True},
        ])
        report = evaluate(pack, FakeModel(["not json at all"]))
        self.assertEqual(report.passed, 1, _explain(report))

    def test_a_case_that_does_not_declare_rescued_still_fails_on_a_burnt_ladder(self):
        """The default must not change: an unexpected failure is still a failure."""
        pack = self._pack([
            {"name": "garbage in", "input": {"alert": "???"}, "expect": {}},
        ])
        report = evaluate(pack, FakeModel(["not json at all"]))
        self.assertEqual(report.passed, 0, _explain(report))

    def test_declaring_rescued_on_a_case_that_succeeds_FAILS(self):
        """`rescued` must not become a way to silence real failures.

        A case claiming to exercise the rescue path, whose run sailed through cleanly, is
        not testing what it says it tests.
        """
        pack = self._pack([
            {"name": "fine", "input": {"alert": "x"}, "expect": {"severity": "p0"},
             "end": "paged", "rescued": True},
        ])
        report = evaluate(pack, FakeModel(['{"severity": "p0"}']))
        self.assertEqual(report.passed, 0, _explain(report))

    def test_the_rescued_case_still_reports_which_node_burned(self):
        pack = self._pack([
            {"name": "garbage in", "input": {"alert": "???"}, "expect": {},
             "end": "needs_human", "rescued": True},
        ])
        report = evaluate(pack, FakeModel(["not json at all"]))
        case = report.cases[0]
        self.assertEqual(case.node, "classify")


def _explain(report):
    """Everything a failure knows, as one string — so assertion output is useful."""
    lines = ["%d/%d passed" % (report.passed, len(report.cases))]
    for case in report.cases:
        lines.append("  %s passed=%s node=%s error=%s" % (
            case.name, case.passed, case.node, case.error))
        for mismatch in case.mismatches:
            lines.append("    %s: expected %r got %r (%s)" % (
                mismatch.field, mismatch.expected, mismatch.actual, mismatch.note))
    return "\n".join(lines)


if __name__ == "__main__":
    unittest.main()
