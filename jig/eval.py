"""Scoring a pack against its evalset.

The evalset is the contract (docs/ARCHITECTURE.md §1): prompts, grammars and graphs are all
regenerable build outputs, but the gold examples are the hand-maintained asset, and
"v3 passes 50/50" is the sentence a regulated buyer can actually audit.

What makes this more than a test runner is **attribution**. DSPy's known limitation is
that optimisation is end-to-end and a failure is a black box; because a jig graph gives
every node its own input/output contract and the walker records which node wrote which
state key, a failed case can name the node that caused it. `Report.by_node` is that
per-node signal — the thing a future `jig build` would optimise against.

The second thing it reports is **the tier split**. A single pass rate is the wrong number
to ship on: it averages the cases the pack answered on its own together with the cases it
handed to a human, and those two populations are not comparable. What a buyer is owed
before anything is deployed is three numbers — how much ran automatically, how much went
to a person, and how right the automatic part was — because a pack that automates 60% at
99.9% is worth more than one that automates 95% at 92%, and one pass rate hides which is
which. So every case is classified:

    auto        the run finished with no node unsure and nothing diverted
    escalated   a node was unsure or ran out of ladder, and the pack routed it
    failed      nothing routed it; the run could not finish at all

and `Report.auto_accuracy` is scored over the auto bucket alone. The accuracy of the
escalated bucket is not the pack's claim to make — a human answers those.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import NodeFailed
from .graph import run

__all__ = [
    "CaseResult",
    "Escalation",
    "Mismatch",
    "Report",
    "TIERS",
    "TIER_AUTO",
    "TIER_ESCALATED",
    "TIER_FAILED",
    "evaluate",
]

_MISSING = object()
UNKNOWN_NODE = "<unknown>"

# The three tiers, in the order a report reads them: what the pack did by itself, what it
# handed over, and what it could not finish. They are strings rather than an enum because
# they cross into JSON and into a text report, and both want the word.
TIER_AUTO = "auto"
TIER_ESCALATED = "escalated"
TIER_FAILED = "failed"
TIERS = (TIER_AUTO, TIER_ESCALATED, TIER_FAILED)


@dataclass
class Mismatch:
    """One expected field that did not come back as promised."""

    field: str
    expected: Any
    actual: Any
    node: Optional[str] = None
    note: str = ""


@dataclass
class Escalation:
    """A node that handed its case to a human instead of answering it.

    `kind` is which signal did it: `"unsure"` for the confidence gate, `"failed"` for a
    node that ran out of retry ladder and took `on_fail`. Both end the same way — a
    person finishes the case — but an operator fixes them differently, so the report
    keeps them apart.
    """

    node: str
    kind: str
    reason: str = ""


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
    escalations: List[Escalation] = field(default_factory=list)
    # Defaulting to `failed` is not pessimism, it is the value the early returns want: a
    # case whose run raised never reaches the line that classifies it, and a run that did
    # not finish is exactly what this tier means. Nothing may be counted as automated
    # without a completed run to say so.
    tier: str = TIER_FAILED

    @property
    def escalated_by(self):
        """The node that escalated this case — the earliest one the walk reached."""
        return self.escalations[0].node if self.escalations else None


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

    # ------------------------------------------------------------------ the tier split
    #
    # Derived from the cases rather than tallied alongside them: there is then one place
    # a tier can be decided (`_run_case`), and a report assembled by hand — a subset
    # rerun, a report rebuilt from JSON — reports the same numbers as one `evaluate`
    # produced.

    @property
    def tier_counts(self):
        """How many cases landed in each tier. Always all three keys.

        A missing key reads as "not measured", and zero escalations is a result worth
        printing, not a gap.
        """
        counts = OrderedDict((tier, 0) for tier in TIERS)
        for case in self.cases:
            counts[case.tier] = counts.get(case.tier, 0) + 1
        return counts

    @property
    def auto_cases(self):
        """The cases the pack answered on its own — the only ones it may claim."""
        return [case for case in self.cases if case.tier == TIER_AUTO]

    @property
    def auto_passed(self):
        return sum(1 for case in self.auto_cases if case.passed)

    @property
    def auto_accuracy(self):
        """Correctness inside the auto bucket, as a fraction — the number that matters.

        `None` when nothing was automated, rather than 0.0 or 1.0: a rate over an empty
        bucket is not a low score, it is an absent one, and either digit would be a claim
        nobody measured.

        Escalated cases are deliberately not scored here even when they pass. A `rescued`
        case passing means the pack routed it as promised, not that it answered it — the
        human on the other end of that edge is the one who answers it.
        """
        cases = self.auto_cases
        if not cases:
            return None
        return sum(1 for case in cases if case.passed) / float(len(cases))

    @property
    def automation_rate(self):
        """The share of cases that went end to end with nobody's help. `None` if empty."""
        return self._rate(TIER_AUTO)

    @property
    def escalation_rate(self):
        return self._rate(TIER_ESCALATED)

    @property
    def failure_rate(self):
        return self._rate(TIER_FAILED)

    def _rate(self, tier):
        if not self.cases:
            return None
        return self.tier_counts[tier] / float(len(self.cases))

    @property
    def escalated_by(self):
        """Which node handed each escalated case over, tallied.

        This is the line an operator acts on: it names where to put the next `assert`, or
        which node's prompt is costing the automation rate.
        """
        return self._tally(
            TIER_ESCALATED, lambda case: case.escalated_by or UNKNOWN_NODE
        )

    @property
    def failed_by(self):
        """Which node the runs that could not finish died on.

        Distinct from `by_node`, which tallies *wrong answers* per node. A node here has
        no route out — it needs an `on_fail` (or an `on_unsure`) before this pack ships.
        """
        return self._tally(TIER_FAILED, lambda case: case.node or UNKNOWN_NODE)

    def _tally(self, tier, name_of):
        counts = OrderedDict()
        for case in self.cases:
            if case.tier != tier:
                continue
            node = name_of(case)
            counts[node] = counts.get(node, 0) + 1
        return counts

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

    def tier_summary(self):
        """The automation rate, as three lines, printed by `jig eval --tiers`.

        Separate from `summary()` on purpose. `summary()` is what every existing script
        and transcript already parses, and it does not change; this is an operator asking
        one more question of the same report.
        """
        if not self.cases:
            return "%s: no cases" % self.pack
        counts = self.tier_counts
        return "\n".join([
            "%s: %d case%s — %d auto, %d escalated, %d failed"
            % (self.pack, self.total, "" if self.total == 1 else "s",
               counts[TIER_AUTO], counts[TIER_ESCALATED], counts[TIER_FAILED]),
            "  auto       %s   %s" % (_percent(self.automation_rate), self._accuracy()),
            "  escalated  %s%s" % (_percent(self.escalation_rate),
                                   _blamed(self.escalated_by)),
            "  failed     %s%s" % (_percent(self.failure_rate), _blamed(self.failed_by)),
        ])

    def _accuracy(self):
        """The auto bucket's own score, spelled out so it cannot be read as the total."""
        accuracy = self.auto_accuracy
        if accuracy is None:
            return "accuracy n/a — no case ran without escalating"
        return "accuracy %s (%d/%d correct)" % (
            _percent(accuracy).strip(), self.auto_passed, len(self.auto_cases)
        )


def evaluate(pack, model, cases=None, tools=None):
    """Run every evalset case through `pack` and score it.

    `model` is either a `Model` or a zero-argument factory returning one — a factory
    gives each case a fresh model, which is how an ordered `FakeModel` script stays
    readable across a dozen cases.

    `tools` is the host's `ToolRegistry` (see `jig.tools`), for a pack whose graph has
    `tool` nodes. It is handed to the walker only when it was given, so a pack with no
    tools — and a runtime whose walker predates them — scores exactly as it did before.
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
        result = _run_case(pack, model, case, index, tools)
        report.cases.append(result)
        if not result.passed:
            node = result.node or UNKNOWN_NODE
            counts[node] = counts.get(node, 0) + 1
    report.by_node = dict(counts)
    return report


def _run_case(pack, model, case, index, tools=None):
    name = case.name or "case %d" % index
    result = CaseResult(
        name=name, passed=False, input=dict(case.input), expected=dict(case.expect)
    )
    trace = _Trace(pack.entry)
    # Passed through only when there is something to pass. A walker that takes no `tools`
    # keyword must still run every pack that needs none, which is all of them today.
    extra = {"tools": tools} if tools is not None else {}
    try:
        run_result = run(pack, _model_for(model), dict(case.input), store=trace, **extra)
    except NodeFailed as exc:
        result.node, result.error = exc.node, str(exc)
        return result
    except Exception as exc:  # a backend that falls over fails its case, not the suite
        result.node = trace.pending
        result.error = "%s: %s" % (type(exc).__name__, exc)
        return result

    result.run_id = run_result.run_id
    result.actual = dict(run_result.output)

    # The run finished, so this case is the pack's to claim or to hand over. Conservative
    # on purpose: `auto` requires that nothing in the run hesitated. An unsure node the
    # pack chose not to route still ran unsupervised — but the auto bucket is a promise
    # about answers nobody checked, and an answer the node itself would not stand behind
    # does not belong in it. Anything short of certain leaves the bucket.
    result.escalations = _escalations(run_result)
    result.tier = TIER_ESCALATED if result.escalations else TIER_AUTO
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
    rank = _rank(path)

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


def _rank(path):
    """Each node's position at its *first* visit — the order blame is decided in.

    Shared by the two things that have to agree about which node comes first: the wrong
    field a case is blamed on, and the node an escalation is attributed to. A node
    entered twice in a loop is ranked by the first time the walk reached it.
    """
    rank = {}
    for position, node in enumerate(path):
        rank.setdefault(node, position)
    return rank


def _escalations(run_result):
    """Everything in this run that put a human in the loop, earliest node first.

    Two signals, produced at two layers. `failures` is the walker's own record of a node
    that ran out of retry ladder and was diverted by `on_fail` — the walker only records
    one when there was somewhere to divert to, so a failure here always means the case
    was routed rather than dropped. `unsure` is the confidence gate's record of a node
    that produced an answer it could not stand behind.
    """
    records = [
        Escalation(node=failure.node, kind="failed",
                   reason=getattr(failure, "reason", "") or "")
        for failure in getattr(run_result, "failures", None) or ()
    ]
    records.extend(_unsure(run_result))
    # The same rule that blames a wrong field: the walk's order decides, not the order
    # two separate lists happen to be in. The sort is stable, so a node's own records
    # keep the order the run recorded them in.
    rank = _rank(getattr(run_result, "path", None) or [])
    records.sort(key=lambda record: rank.get(record.node, len(rank)))
    return records


def _unsure(run_result):
    """This run's confidence-gate records, if the runtime it ran on has a gate.

    Read with `getattr` and shaped loosely on purpose. The gate lives in the walker; a
    `RunResult` replayed from a checkpoint written before it existed has no such field,
    and a pack that sets no tiers leaves it empty. None of those may stop a report being
    scored. All eval needs from a record is the node's name, so a dataclass, a mapping
    and a bare node name are all accepted — whatever the walker publishes is readable.

    A record naming no node is still an escalation: it is counted, and attributed to
    `<unknown>`. Losing the attribution is a smaller error than quietly counting the case
    as automated.
    """
    for record in getattr(run_result, "unsure", None) or ():
        yield Escalation(
            node=str(_record_field(record, "node") or UNKNOWN_NODE),
            kind="unsure",
            reason=str(_record_field(record, "reason") or ""),
        )


def _record_field(record, name):
    if isinstance(record, str):
        return record if name == "node" else ""
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


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


def _percent(rate):
    """A rate as a right-aligned percentage, so the three tier lines form a column."""
    if rate is None:
        return "   n/a"
    return "%5.1f%%" % (rate * 100.0)


def _blamed(counts):
    """`   at classify=2, confirm=1`, or nothing at all when the tier is empty.

    Sorted by how many cases each node cost, not alphabetically: this list exists to be
    read top-down by someone deciding where the next `assert` goes.
    """
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "   at %s" % ", ".join("%s=%d" % item for item in ordered)


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
