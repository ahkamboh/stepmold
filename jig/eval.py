"""Scoring a pack against its evalset.

The evalset is the contract (docs/PLAN.md §1): prompts, grammars and graphs are all
regenerable build outputs, but the gold examples are the hand-maintained asset, and
"v3 passes 50/50" is the sentence a regulated buyer can actually audit.

What makes this more than a test runner is **attribution**. DSPy's known limitation is
that optimisation is end-to-end and a failure is a black box; because a jig graph gives
every node its own input/output contract and the walker records which node wrote which
state key, a failed case can name the node that caused it. `Report.by_node` is that
per-node signal — the thing a future `jig build` would optimise against.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import NodeFailed
from .graph import run

__all__ = ["CaseResult", "Mismatch", "Report", "evaluate"]

_MISSING = object()
UNKNOWN_NODE = "<unknown>"


@dataclass
class Mismatch:
    """One expected field that did not come back as promised."""

    field: str
    expected: Any
    actual: Any
    node: Optional[str] = None
    note: str = ""


@dataclass
class CaseResult:
    """One evalset case, run."""

    name: str
    passed: bool
    input: Dict[str, Any]
    expected: Dict[str, Any]
    actual: Dict[str, Any] = field(default_factory=dict)
    mismatches: List[Mismatch] = field(default_factory=list)
    node: Optional[str] = None
    error: Optional[str] = None
    run_id: Optional[str] = None


@dataclass
class Report:
    """The score, plus enough detail to act on it."""

    pack: str
    cases: List[CaseResult] = field(default_factory=list)
    by_node: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self):
        return len(self.cases)

    @property
    def passed(self):
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed(self):
        return self.total - self.passed

    @property
    def passed_all(self):
        return self.failed == 0 and self.total > 0

    def summary(self):
        """A short human report — what the CLI prints."""
        lines = ["%s: %d/%d cases passed" % (self.pack, self.passed, self.total)]
        for case in self.cases:
            if case.passed:
                continue
            lines.append("  FAIL %s [%s]" % (case.name, case.node or UNKNOWN_NODE))
            if case.error:
                lines.append("    error: %s" % case.error)
            for mismatch in case.mismatches:
                lines.append(
                    "    %s: expected %r, got %s"
                    % (
                        mismatch.field,
                        mismatch.expected,
                        mismatch.note or repr(mismatch.actual),
                    )
                )
        if self.by_node:
            lines.append(
                "  failures by node: %s"
                % ", ".join("%s=%d" % item for item in sorted(self.by_node.items()))
            )
        return "\n".join(lines)


def evaluate(pack, model, cases=None):
    """Run every evalset case through `pack` and score it.

    `model` is either a `Model` or a zero-argument factory returning one — a factory
    gives each case a fresh model, which is how an ordered `FakeModel` script stays
    readable across a dozen cases.
    """
    cases = list(pack.evalset if cases is None else cases)
    if not cases:
        raise ValueError(
            "pack %r has no evalset cases to run — an empty evalset is not a pass"
            % pack.name
        )

    report = Report(pack=pack.name)
    counts = OrderedDict()
    for index, case in enumerate(cases, start=1):
        result = _run_case(pack, model, case, index)
        report.cases.append(result)
        if not result.passed:
            node = result.node or UNKNOWN_NODE
            counts[node] = counts.get(node, 0) + 1
    report.by_node = dict(counts)
    return report


def _run_case(pack, model, case, index):
    name = case.name or "case %d" % index
    result = CaseResult(
        name=name, passed=False, input=dict(case.input), expected=dict(case.expect)
    )
    trace = _Trace(pack.entry)
    try:
        run_result = run(pack, _model_for(model), dict(case.input), store=trace)
    except NodeFailed as exc:
        result.node, result.error = exc.node, str(exc)
        return result
    except Exception as exc:  # a backend that falls over fails its case, not the suite
        result.node = trace.pending
        result.error = "%s: %s" % (type(exc).__name__, exc)
        return result

    result.run_id = run_result.run_id
    result.actual = dict(run_result.output)
    result.mismatches = _compare(case.expect, run_result)
    result.mismatches.extend(_compare_ending(case, run_result))
    diverted = run_result.failures[0] if run_result.failures else None

    # `rescued` says this case is meant to burn a ladder and take an on_fail edge. It is
    # checked in both directions: an undeclared failure still fails the case, and a case
    # that claims a rescue but sailed through is not testing what it says it tests — so
    # it cannot be used to silence a real failure.
    if case.rescued and diverted is None:
        result.node = run_result.end_node
        result.error = (
            "case declares rescued: true but the run completed with no failure — "
            "either the rescue path is not being exercised, or the flag is wrong"
        )
        return result
    unexpected = diverted is not None and not case.rescued

    if not result.mismatches and not unexpected:
        result.passed = True
        if diverted is not None:
            # Passing, but say which node needed rescuing: an operator reading a green
            # report should still see where the ladder was spent.
            result.node = diverted.node
        return result
    if unexpected:
        # A node that burned its ladder and was diverted is the cause, even when the
        # projected output happens to look right.
        result.node = diverted.node
        result.error = diverted.reason
    else:
        result.node = _earliest_node(result.mismatches, run_result.path)
    return result


def _compare_ending(case, run_result):
    """The branch a run took, as part of the contract.

    `expect` compares fields, and the ending is not a field — so a pack whose branches
    project the same shape could have its routing inverted and still score full marks.
    Naming the ending is what closes that.
    """
    if case.end is None:
        return []
    if run_result.end_node == case.end:
        return []
    return [
        Mismatch(
            field="<ending>",
            expected=case.end,
            actual=run_result.end_node,
            node=run_result.path[-2] if len(run_result.path) > 1 else None,
            # No note: the renderer substitutes a note FOR the actual value, and here the
            # actual ending is the whole point. "<ending>" already says what the field is.
        )
    ]


def _earliest_node(mismatches, path):
    """Blame the node the walker reached first, not the field listed first.

    `expect` is a JSON object typed by hand, so its key order carries no meaning; the
    run's visit order does. When two nodes both produced a wrong field, the earlier one
    is the one to fix — the later node consumed its mistake. A node can be visited more
    than once in a loop, so its first visit is what ranks it.
    """
    rank = {}
    for position, node in enumerate(path):
        rank.setdefault(node, position)

    best = None
    for mismatch in mismatches:
        if mismatch.node is None:
            continue  # an expectation on an input field: nobody wrote it
        if mismatch.node not in rank:
            continue  # provenance from outside this run's path; cannot be ordered
        if best is None or rank[mismatch.node] < rank[best]:
            best = mismatch.node
    if best is not None:
        return best
    return mismatches[0].node


def _compare(expect, run_result):
    """Exact match on the declared fields, each blamed on whoever wrote it."""
    mismatches = []
    for name, expected in expect.items():
        actual = _actual(name, run_result)
        if actual is _MISSING:
            mismatches.append(
                Mismatch(
                    field=name,
                    expected=expected,
                    actual=None,
                    node=run_result.provenance.get(name),
                    note="missing from output",
                )
            )
        elif actual != expected:
            mismatches.append(
                Mismatch(
                    field=name,
                    expected=expected,
                    actual=actual,
                    node=run_result.provenance.get(name),
                )
            )
    return mismatches


def _actual(name, run_result):
    """Look in the projected output first, then the full state.

    An `end` node that projects a subset should not stop an evalset from asserting on an
    intermediate field — per-node expectations are exactly the signal we want.
    """
    if name in run_result.output:
        return run_result.output[name]
    if name in run_result.state:
        return run_result.state[name]
    return _MISSING


def _model_for(model):
    if hasattr(model, "generate"):
        return model
    if callable(model):
        return model()
    raise TypeError(
        "evaluate() needs a Model or a factory returning one, got %s"
        % type(model).__name__
    )


class _Trace:
    """A store that keeps only the last checkpoint, to blame the right node on a crash."""

    def __init__(self, entry):
        self.pending = entry

    def save(self, **checkpoint):
        self.pending = checkpoint.get("next_node")
