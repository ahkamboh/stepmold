"""The thesis test: does jig actually remove error compounding over a long horizon?

docs/ARCHITECTURE.md §0 sells jig on one number — *"2%/step = 33% failure at 20 steps"* — and §3
claims the fix is structural: bounded steps, a grammar per node, verify-before-commit,
and a rejected generation that never re-enters context. Every other test in this repo
checks a mechanism. This one checks the *claim*, end to end, at N = 5, 20 and 50, against
a seeded model that fails the way small models fail (`longpack.FlakyModel`).

Read this file as an experiment report, not a checklist. Each class states what it
measures, the assertions pin the measurement, and the numbers in the comments are the
values observed when it was written — they are exact, because every draw is a hash of
`(seed, node, attempt)` and nothing here is wall-clock or RNG-order dependent.

    python3 -m tests.production.test_longhorizon     # prints the full tables

What it found, in one paragraph. jig's ladder+verification is worth an enormous amount
when the model's re-samples are *independent* draws — at N=50, p=0.10 it turns a 0.5%
analytical survival rate into 96.5%, an effective per-step error of 0.07% against a real
one of 10%. Three things broke that, and two of them have since been fixed here. A model
whose re-sample reproduces the same mistake — a greedy decoder — got *nothing* from the
ladder, because nothing in the protocol could ask for a different draw; the ladder now
sends a per-call sampling hint (`codegen.Sampling`) and a backend that honours it scores
97.3% where the same model scored 0.7%. A `two_stage` node got nothing either, because
`verify.run_node` reused the scratchpad from the rejected attempt and never re-ran
`think`; it now discards it, and that arm goes from 0.0% to 96.0% at N=50 p=0.10. What
remains is the third: verification with retries turned off scores *below* no
verification at all, which is a trade rather than a bug. Details in the classes below.
"""

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass

from jig.codegen import accepts_sampling
from jig.errors import RunError
from jig.graph import run
from jig.model import FakeModel
from jig.pack import Node
from jig.state import Store
from jig.verify import Rejected, extract_json, run_node, sampling_for, verify

from tests.production import longpack as L


# Sample size. 200 seeds resolves a 5-point difference in success rate well enough to
# assert on, and the whole sweep — 4 arms x 3 horizons x 4 error rates — runs in a few
# seconds because `FlakyModel` is arithmetic, not a network.
TRIALS = 200
HORIZONS = (5, 20, 50)
ERROR_RATES = (0.0, 0.02, 0.10, 0.30)

_ROOT = None
_PACKS = {}
_CELLS = {}


def setUpModule():
    global _ROOT
    _ROOT = tempfile.mkdtemp(prefix="jig-longhorizon-")


def tearDownModule():
    if _ROOT:
        shutil.rmtree(_ROOT, ignore_errors=True)


# --------------------------------------------------------------------- the harness


@dataclass
class Cell:
    """One (horizon, error rate, arm) measurement over `TRIALS` seeds."""

    n: int
    p: float
    arm: str
    trials: int = 0
    successes: int = 0        # finished at `done` with the analytically correct answer
    silent_wrong: int = 0     # finished at `done` with a wrong answer — the bad outcome
    gave_up: int = 0          # diverted to the arm's `on_fail` end node
    raised: int = 0           # RunError escaped: a node spent its ladder with no on_fail
    calls: int = 0
    calls_on_success: int = 0
    retries: int = 0          # generations spent on attempts after the first

    @property
    def rate(self):
        return self.successes / float(self.trials)

    @property
    def analytic(self):
        """ARCHITECTURE.md's own model of a naive run: (1 - p) ** N."""
        return (1.0 - self.p) ** self.n

    @property
    def effective_p(self):
        """The per-step error rate that would explain the measured end-to-end rate.

        This is the number the whole product is about. jig cannot make the model better,
        so if the thesis holds, `effective_p` is far below `p` and `naive`'s is not.
        """
        if self.rate <= 0.0:
            return 1.0
        return 1.0 - self.rate ** (1.0 / self.n)

    @property
    def call_overhead(self):
        """Cost of the ladder: generations per *successful* run, over the ideal N."""
        if not self.successes:
            return float("nan")
        return (self.calls_on_success / float(self.successes)) / self.n


def pack_for(n, arm):
    key = (n, arm.name)
    if key not in _PACKS:
        _PACKS[key] = L.build_pack(_ROOT, n, arm)
    return _PACKS[key]


class StubbornUnlessAsked(L.FlakyModel):
    """A greedy backend that honours the ladder's per-call sampling hint.

    `FlakyModel(stubborn=True)` is a server at temperature 0: ask it the same question
    twice and it makes the same mistake twice, which is what made jig's first rung dead
    weight. This one is that same server, with the one difference the fix introduces —
    a request carrying a `Sampling` is a *different* request, so its draw moves with the
    rung. It declares the optional `sampling` keyword, which is how `codegen._generate`
    knows the hint is worth sending at all.

    It lives here rather than in `longpack` because it is the fix's measurement
    instrument, not a way small models fail.
    """

    def generate(self, prompt, grammar=None, max_tokens=512, sampling=None):
        # Stubborn exactly while nothing has asked it to vary.
        self.stubborn = sampling is None
        return L.FlakyModel.generate(
            self, prompt, grammar=grammar, max_tokens=max_tokens
        )


def trial(n, arm, seed, p, stubborn=False, model_type=L.FlakyModel):
    """One run. Returns (outcome, model, result-or-error).

    `v0` is derived from the seed, so two arms sharing a seed share an input as well as
    a fault schedule — that is what makes the arm comparison paired rather than merely
    averaged. `model_type` swaps the backend for one that answers the sampling hint,
    which is the only thing separating the two independent-draw columns below.
    """
    pack = pack_for(n, arm)
    model = model_type(seed=seed, p=p, stubborn=stubborn)
    v0 = seed % L.STEP_MOD
    try:
        result = run(pack, model, inputs={"v0": v0})
    except RunError as exc:
        return "raised", model, exc
    if result.end_node != "done":
        return "gave_up", model, result
    if result.output.get("v%d" % n) == L.expected_final(v0, n):
        return "success", model, result
    return "silent_wrong", model, result


def cell(n, p, arm, trials=TRIALS, stubborn=False, model_type=L.FlakyModel):
    """Measure one (n, p, arm) cell, memoised — several tests read the same numbers."""
    key = (n, p, arm.name, trials, stubborn, model_type.__name__)
    if key in _CELLS:
        return _CELLS[key]
    out = Cell(n=n, p=p, arm=arm.name, trials=trials)
    for seed in range(trials):
        outcome, model, _ = trial(n, arm, seed, p, stubborn=stubborn,
                                  model_type=model_type)
        setattr(out, outcome if outcome != "success" else "successes",
                getattr(out, outcome if outcome != "success" else "successes") + 1)
        out.calls += model.call_count
        out.retries += model.wasted_calls()
        if outcome == "success":
            out.calls_on_success += model.call_count
    _CELLS[key] = out
    return out


# ------------------------------------------------------------------ harness sanity


class TheHarnessMeasuresWhatItClaims(unittest.TestCase):
    """Before believing any number below, the apparatus has to be sound."""

    def test_a_perfect_model_finishes_every_horizon_in_every_arm(self):
        """p=0 must be 100% everywhere, or the generated pack is the thing failing."""
        for n in HORIZONS:
            for arm in L.ARMS:
                measured = cell(n, 0.0, arm, trials=20)
                self.assertEqual(
                    measured.successes, 20,
                    "arm %s at N=%d fails with a perfect model — the pack is broken, "
                    "not the model" % (arm.name, n),
                )
                # Exactly one generation per node: no retries, no hidden calls.
                self.assertEqual(measured.calls, 20 * n)

    def test_a_wrong_answer_is_unambiguous(self):
        """The chain has to be checkable, or 'success' is a matter of opinion."""
        self.assertEqual(L.expected_final(42, 1), L.step_value(42))
        self.assertEqual(L.expected_final(42, 3),
                         L.step_value(L.step_value(L.step_value(42))))
        # A single corrupted link never rejoins the correct chain.
        for start in range(0, 1000, 37):
            self.assertNotEqual(L.chain_answer(start, 50),
                                L.chain_answer((start + 1) % 1000, 50))

    def test_arms_share_a_fault_schedule_so_the_comparison_is_paired(self):
        """Node 1 must misbehave identically in every arm at the same seed.

        This is the validity condition for every 'jig vs baseline' number in this file.
        A sequential RNG would desynchronise as soon as one arm made a different number
        of calls, and the comparison would be measuring stream drift.
        """
        for seed in range(40):
            faults = set()
            for arm in L.ARMS:
                _, model, _ = trial(20, arm, seed, 0.30)
                first = [e for e in model.emissions if e.node == "n1" and e.attempt == 0]
                self.assertEqual(len(first), 1)
                faults.add(first[0].fault)
            self.assertEqual(len(faults), 1,
                             "seed %d produced different first faults per arm" % seed)

    def test_the_fault_mix_is_a_distribution(self):
        self.assertAlmostEqual(sum(weight for _, weight in L.FAULT_MIX), 1.0, places=9)


# ------------------------------------------------------- compounding, and its removal


class CompoundingIsRealWithoutVerification(unittest.TestCase):
    """ARCHITECTURE.md §0's premise. If this does not reproduce, jig is solving nothing.

    Measured (TRIALS=200), naive arm — success rate vs the analytical (1-p)^N:

        N=5   p=0.02   88.5%  (90.4%)     N=20  p=0.02  68.0%  (66.8%)
        N=5   p=0.10   63.5%  (59.0%)     N=20  p=0.10  16.0%  (12.2%)
        N=5   p=0.30   28.0%  (16.8%)     N=20  p=0.30   0.5%  ( 0.1%)
        N=50  p=0.02   48.0%  (36.4%)     N=50  p=0.10   2.0%  ( 0.5%)
        N=50  p=0.30    0.0%  ( 0.0%)

    It sits slightly *above* the analytical curve because two of the six fault kinds
    (`prose_wrapped`, `extra_key`) do not damage the answer — 80% of faults are harmful,
    so the honest comparison is against (1 - 0.8p)^N, which it tracks closely.
    """

    def test_the_naive_baseline_decays_geometrically(self):
        for p in (0.02, 0.10):
            for n in HORIZONS:
                measured = cell(n, p, L.NAIVE)
                harmful = (1.0 - 0.8 * p) ** n
                self.assertLess(
                    abs(measured.rate - harmful), 0.12,
                    "naive at N=%d p=%.2f measured %.3f, geometric model says %.3f"
                    % (n, p, measured.rate, harmful),
                )

    def test_plan_md_s_headline_number_reproduces(self):
        """'2%/step = 33% failure at 20 steps' — the sentence the pitch rests on."""
        measured = cell(20, 0.02, L.NAIVE)
        self.assertLess(measured.rate, 0.80)      # measured 68.0%
        self.assertGreater(measured.rate, 0.55)
        # And the failure rate is in the neighbourhood ARCHITECTURE.md names.
        self.assertGreater(1.0 - measured.rate, 0.20)

    def test_the_naive_baseline_returns_wrong_answers_silently(self):
        """The failure mode that matters: a completed run holding a wrong number.

        A run that raises is a run an operator can see. `naive` mostly does not raise;
        it finishes, and hands back a confident wrong answer.
        """
        measured = cell(20, 0.10, L.NAIVE)
        self.assertGreater(measured.silent_wrong, 30)   # measured 68 of 200
        self.assertGreater(measured.silent_wrong, measured.successes)


class JigBeatsTheAnalyticalCurve(unittest.TestCase):
    """The headline. Same seeds, same faults, verification and the ladder switched on.

    Measured (TRIALS=200), jig arm vs naive vs the analytical (1-p)^N:

        N     p      jig     naive   (1-p)^N   jig - analytic
        5     0.02  100.0%   88.5%    90.4%     +9.6
        5     0.10   99.5%   63.5%    59.0%    +40.5
        5     0.30   90.5%   28.0%    16.8%    +73.7
        20    0.02  100.0%   68.0%    66.8%    +33.2
        20    0.10   99.0%   16.0%    12.2%    +86.8
        20    0.30   68.5%    0.5%     0.1%    +68.4
        50    0.02  100.0%   48.0%    36.4%    +63.6
        50    0.10   96.5%    2.0%     0.5%    +96.0
        50    0.30   40.0%    0.0%     0.0%    +40.0
    """

    def test_jig_finishes_a_fifty_node_run_perfectly_at_the_plan_s_error_rate(self):
        """N=50 at p=0.02: the analytical model says 36% survive. jig says all of them."""
        measured = cell(50, 0.02, L.JIG)
        self.assertEqual(measured.successes, TRIALS)
        self.assertGreater(measured.rate - measured.analytic, 0.55)

    def test_jig_beats_the_curve_by_a_growing_margin_as_the_horizon_grows(self):
        """The margin has to grow with N, or jig is not attacking *compounding*."""
        margins = [cell(n, 0.10, L.JIG).rate - cell(n, 0.10, L.JIG).analytic
                   for n in HORIZONS]
        self.assertEqual(margins, sorted(margins),
                         "margin over the analytical curve must grow with N: %s" % margins)
        self.assertGreater(margins[-1], 0.90)     # measured +96.0 points at N=50

    def test_jig_beats_the_naive_baseline_on_every_cell_it_can(self):
        for n in HORIZONS:
            for p in (0.02, 0.10, 0.30):
                jig, naive = cell(n, p, L.JIG), cell(n, p, L.NAIVE)
                self.assertGreater(
                    jig.rate, naive.rate,
                    "jig lost to the naive baseline at N=%d p=%.2f (%.3f vs %.3f)"
                    % (n, p, jig.rate, naive.rate),
                )

    def test_jig_collapses_the_effective_per_step_error_rate(self):
        """The mechanism, expressed as the number a buyer would care about.

        Measured effective per-step error (the p that would explain the end-to-end rate):

            N=50 p=0.02 -> 0.000  (jig)  vs 0.0146 (naive)
            N=50 p=0.10 -> 0.0007 (jig)  vs 0.0757 (naive)   ~106x reduction
            N=50 p=0.30 -> 0.0182 (jig)  vs 1.0    (naive)    ~16x reduction
        """
        jig = cell(50, 0.10, L.JIG)
        naive = cell(50, 0.10, L.NAIVE)
        self.assertLess(jig.effective_p, 0.002)
        self.assertGreater(naive.effective_p, 0.05)
        self.assertGreater(naive.effective_p / max(jig.effective_p, 1e-9), 50.0)

    def test_jig_never_returns_a_silently_wrong_answer(self):
        """The claim `naive` cannot make: if jig finishes, the answer is right.

        Verify-before-commit means every committed link satisfied its assert, so a run
        that reaches `done` cannot be carrying a corrupted chain. Across 2,400 runs at
        every horizon and error rate, not one.
        """
        for n in HORIZONS:
            for p in ERROR_RATES:
                measured = cell(n, p, L.JIG)
                self.assertEqual(
                    measured.silent_wrong, 0,
                    "jig finished a run at N=%d p=%.2f with a wrong answer" % (n, p),
                )

    def test_the_ladder_costs_little_when_it_is_not_needed(self):
        """Generations per successful run, over the ideal N: 1.02x / 1.09x / 1.32x."""
        self.assertLess(cell(50, 0.02, L.JIG).call_overhead, 1.05)
        self.assertLess(cell(50, 0.10, L.JIG).call_overhead, 1.15)
        self.assertLess(cell(50, 0.30, L.JIG).call_overhead, 1.45)


class WhereJigStopsRescuingTheRun(unittest.TestCase):
    """The question the brief actually asks: at what (N, p) does it break down?

    Measured (TRIALS=200), jig arm with the default `retries: 2`:

        p \\ N      5       20      50
        0.02    100.0%  100.0%  100.0%
        0.10     99.5%   99.0%   96.5%
        0.30     90.5%   68.5%   40.0%

    The knee is p=0.30. It is a *ladder-depth* limit, not a structural one: three
    attempts leave a residual per-node failure of ~(0.9p)^3 = 2.0%, and 2% compounded
    over 50 nodes is exactly the 40% that survives. Raising `retries` to 5 restores
    N=50 p=0.30 to 98.0% for 1.37x the generations.
    """

    def test_the_knee_is_at_p_0_30(self):
        self.assertGreater(cell(50, 0.10, L.JIG).rate, 0.90)
        self.assertLess(cell(50, 0.30, L.JIG).rate, 0.60)

    def test_the_residual_matches_a_three_attempt_ladder(self):
        """40% at N=50 p=0.30 is (1 - (0.9p)^3)^50 — the ladder's own arithmetic."""
        residual = (0.9 * 0.30) ** 3
        predicted = (1.0 - residual) ** 50
        self.assertLess(abs(cell(50, 0.30, L.JIG).rate - predicted), 0.10)

    def test_a_deeper_ladder_buys_the_run_back(self):
        two = cell(50, 0.30, L.JIG)
        five = cell(50, 0.30, L.JIG_LADDER_5)
        self.assertGreater(five.rate - two.rate, 0.50)     # 98.0% vs 40.0%
        self.assertGreater(five.rate, 0.95)
        # And it is not free: the extra rungs are generations.
        self.assertGreater(five.call_overhead, two.call_overhead)


# ----------------------------------------------------------- the gaps, and their fixes


class TheLadderNeedsAnIndependentResample(unittest.TestCase):
    """FIXED: the ladder asks for a different draw, and a backend that can, gives one.

    The defect this class was written for: every re-sample was the same request plus a
    line of feedback, because the `Model` protocol carried no sampling parameter and
    `OpenAICompatModel` pins temperature at 0.0. A greedy server answering the same
    question the same way is the *expected* case, not an exotic one — and
    `FlakyModel(stubborn=True)` is exactly that model. Measured then (TRIALS=150), jig
    arm, independent draws vs stubborn:

        N=20 p=0.10   99.3%  ->  12.7%
        N=50 p=0.02  100.0%  ->  40.0%
        N=50 p=0.10   97.3%  ->   0.7%
        N=50 p=0.30   42.0%  ->   0.0%

    The stubborn column is the naive baseline's curve: every point jig scored over the
    analytical prediction was bought by an assumption nothing in jig made true.

    The fix is a per-call hint (`codegen.Sampling`) that `verify.sampling_for` climbs
    across the rungs, sent only to a model whose `generate` declares it. Same seeds,
    same faults, same stubborn server — now told that rung 2 wants a different draw
    (`StubbornUnlessAsked`):

        N     p     stubborn   + honours the hint   independent draws
        20    0.02    60.7%          100.0%              100.0%
        20    0.10    12.7%           99.3%               99.3%
        50    0.02    40.0%          100.0%              100.0%
        50    0.10     0.7%           97.3%               97.3%
        50    0.30     0.0%           42.0%               42.0%

    A backend that honours the hint scores *identically* to a model that re-draws on its
    own, which is the whole claim: the ladder's arithmetic was always right, and what
    was missing was a way to ask. A backend that ignores the hint still gets nothing —
    that residual is now the backend's property rather than a hole in jig, and it is the
    one number in this file an operator can fix by changing a server flag.
    """

    def test_a_backend_that_ignores_the_hint_still_gets_nothing(self):
        """`FlakyModel` does not declare `sampling`, so it is called exactly as before."""
        free = cell(50, 0.10, L.JIG, trials=150)
        stuck = cell(50, 0.10, L.JIG, trials=150, stubborn=True)
        self.assertGreater(free.rate, 0.90)
        self.assertLess(stuck.rate, 0.05)

    def test_honouring_the_hint_restores_the_whole_of_the_ladder(self):
        hinted = cell(50, 0.10, L.JIG, trials=150, stubborn=True,
                      model_type=StubbornUnlessAsked)
        free = cell(50, 0.10, L.JIG, trials=150)
        self.assertGreater(hinted.rate, 0.95)             # 97.3%, from 0.7%
        self.assertEqual(hinted.successes, free.successes)

    def test_the_rescue_is_the_ladder_and_not_a_luckier_model(self):
        """Proof the rungs did the work: successful runs cost more than N generations.

        The un-hinted stubborn arm's successes cost exactly N — they are the runs where
        nothing went wrong, never the runs the ladder rescued.
        """
        stuck = cell(50, 0.02, L.JIG, trials=150, stubborn=True)
        hinted = cell(50, 0.02, L.JIG, trials=150, stubborn=True,
                      model_type=StubbornUnlessAsked)
        self.assertEqual(stuck.calls_on_success, stuck.successes * 50)
        self.assertGreater(hinted.calls_on_success, hinted.successes * 50)

    def test_the_first_attempt_is_never_perturbed(self):
        """Rung 0 asks for nothing, so a run that never stumbles is byte-identical."""
        self.assertIsNone(sampling_for(0))
        clean = cell(50, 0.0, L.JIG, trials=20, model_type=StubbornUnlessAsked)
        self.assertEqual(clean.successes, 20)
        self.assertEqual(clean.calls, 20 * 50)

    def test_the_hint_climbs_with_the_rung(self):
        first, second = sampling_for(1), sampling_for(2)
        self.assertGreater(second.temperature, first.temperature)
        self.assertNotEqual(second.seed, first.seed)
        # A deeper ladder stays at the top of the measured range rather than inventing
        # temperatures nobody has run.
        self.assertLessEqual(sampling_for(9).temperature, 1.0)

    def test_a_model_that_does_not_ask_for_the_hint_is_called_as_it_always_was(self):
        """The compatibility claim, checked rather than asserted in a docstring."""
        self.assertFalse(accepts_sampling(FakeModel(["{}"])))
        self.assertTrue(accepts_sampling(StubbornUnlessAsked(seed=0, p=0.0)))
        model = FakeModel(['{"v1": 0}', '{"v1": 0}'])
        node = Node(name="n1", type="generate", prompt="p", grammar=L.node_schema(1),
                    assert_expr="v1 == 1", retries=1)
        with self.assertRaises(RunError):
            run_node(node, {}, model)
        self.assertEqual([call.max_tokens for call in model.calls], [512, 512])


class TwoStageRetriesReThinkInsteadOfReEmitting(unittest.TestCase):
    """FIXED: a rejected `two_stage` answer takes its reasoning down with it.

    The defect: `verify.run_node` kept `scratchpad = candidate.scratchpad` across rungs
    and `codegen.generate_once` only calls `think` when the scratchpad is None, so the
    think stage ran exactly once per node and every re-sample re-read the same
    reasoning — including the reasoning that produced the answer just rejected. For a
    model whose emit follows its own notes, which is what conditioning on a scratchpad
    means and the exact case ARCHITECTURE.md Bug 2 introduced the think stage to serve, the
    ladder had no rung that could reach the error.

    `README.md` and `codegen.py` both describe the scratchpad as thrown away and never
    committed, and that is true of *state*. It was not true of the retry prompt.

    Measured (TRIALS=150), the two_stage arm before and after `run_node` learned to drop
    the scratchpad on a rejection:

        N     p     two_stage (before)   (1-p)^N   two_stage (after)    jig
        20    0.02        64.7%           66.8%         100.0%        100.0%
        20    0.10        14.0%           12.2%          96.7%         99.3%
        50    0.02        34.0%           36.4%         100.0%        100.0%
        50    0.10         0.0%            0.5%          96.0%         97.3%
        50    0.30         0.0%            0.0%          24.0%         42.0%

    The before column *is* the analytical curve: for this class of error a two-stage
    node was a naive loop with double the token bill. The after column is a two-stage
    node behaving like a single-stage one, at the two calls per rung it costs.

    It stays a little behind the single-stage `jig` arm, and that is not noise: a
    re-thought rung re-rolls two draws instead of one, so it has two chances to go wrong
    per attempt. That is the honest price of a think stage, not a defect in the ladder.
    """

    SCHEMA = {
        "type": "object",
        "properties": {"v": {"type": "integer"}},
        "required": ["v"],
        "additionalProperties": False,
    }
    NOTES = "SCRATCH-NOTE-4b1c: I am confident the answer is 99"

    def _node(self):
        return Node(
            name="step", type="generate", prompt="Do the step.",
            grammar=self.SCHEMA, two_stage=True, retries=2,
            assert_expr="v == 1",
        )

    def test_a_rejected_rung_re_thinks_before_it_re_emits(self):
        model = FakeModel([self.NOTES, '{"v": 99}', "second thoughts", '{"v": 1}'])

        run_node(self._node(), {}, model)

        unconstrained = [call for call in model.calls if call.grammar is None]
        self.assertEqual(
            len(unconstrained), 2,
            "a node whose error is in its reasoning needs a rung that reaches the "
            "reasoning",
        )

    def test_the_rejected_attempt_s_notes_reach_no_later_prompt(self):
        """The anti-self-conditioning invariant, extended to the notes behind an answer."""
        model = FakeModel([self.NOTES, '{"v": 99}', "second thoughts", '{"v": 1}'])

        run_node(self._node(), {}, model)

        for call in model.calls[2:]:
            self.assertNotIn(
                self.NOTES, call.prompt,
                "ARCHITECTURE.md §3 wants a rejected attempt out of the model's context, and "
                "the scratchpad that produced it is part of that attempt",
            )
        # The notes that were *not* rejected are still there to condition the emit.
        self.assertIn("second thoughts", model.calls[3].prompt)

    def test_end_to_end_a_two_stage_pack_now_beats_the_analytical_curve(self):
        measured = cell(50, 0.02, L.JIG_TWO_STAGE, trials=150)
        self.assertEqual(measured.successes, 150)
        self.assertGreater(measured.rate - measured.analytic, 0.60)

    def test_the_ladder_is_what_is_doing_it(self):
        """Successful runs cost more than the 2N calls a retry-free run would."""
        measured = cell(50, 0.10, L.JIG_TWO_STAGE, trials=150)
        self.assertGreater(measured.rate, 0.90)           # 96.0%, from 0.0%
        self.assertGreater(measured.calls_on_success, measured.successes * 100)

    def test_a_re_thought_rung_costs_two_calls_not_one(self):
        """The price of the fix, stated: ~2.2 generations per node per successful run."""
        measured = cell(50, 0.10, L.JIG_TWO_STAGE, trials=150)
        self.assertGreater(measured.call_overhead, 2.0)
        self.assertLess(measured.call_overhead, 2.5)

    def test_a_stubborn_model_is_a_separate_problem_with_a_separate_fix(self):
        """Re-thinking cannot help a model that repeats itself; the hint can.

        The two defects were independent, and so are their fixes: dropping the
        scratchpad re-rolls a draw, and a stubborn backend's re-roll is the same draw
        until something asks it to vary.
        """
        free = cell(50, 0.02, L.JIG_TWO_STAGE, trials=150)
        stuck = cell(50, 0.02, L.JIG_TWO_STAGE, trials=150, stubborn=True)
        hinted = cell(50, 0.02, L.JIG_TWO_STAGE, trials=150, stubborn=True,
                      model_type=StubbornUnlessAsked)
        self.assertEqual(free.successes, 150)
        self.assertLess(stuck.rate, 0.40)                 # 34.0% — the analytical curve
        self.assertEqual(hinted.successes, free.successes)



class VerificationWithoutALadderScoresBelowNoVerification(unittest.TestCase):
    """FINDING: The design's layer table implies each defence only adds. It does not.

    Measured (TRIALS=200) success rate by arm — each arm adds one defence, retries off
    until the last:

        N     p     naive  grammar_only  verify_only    jig
        5     0.30  28.0%     22.5%         22.5%      90.5%
        20    0.10  16.0%     11.5%         11.5%      99.0%
        50    0.02  48.0%     43.0%         43.0%     100.0%

    Turning the grammar on *lowers* end-to-end success whenever there is no rung behind
    it. The mechanism is `additionalProperties: false`: a model that answers correctly
    and adds a `"confidence"` field has produced a usable answer, and a strict schema with
    `retries: 0` converts it into a dead run. Small models add fields constantly.

    This is arguably the right trade — a loud failure beats a silent wrong answer, and
    the `silent_wrong` column below shows the trade being made — but it is not what the
    layer table says, and a reader who sets `retries: 0` for cost reasons will get a
    worse pack, not a cheaper one.
    """

    def test_adding_a_strict_grammar_without_retries_lowers_success(self):
        for n, p in ((5, 0.30), (20, 0.10), (50, 0.02)):
            naive = cell(n, p, L.NAIVE)
            strict = cell(n, p, L.GRAMMAR_ONLY)
            self.assertLess(
                strict.rate, naive.rate,
                "expected the documented regression at N=%d p=%.2f, got %.3f >= %.3f"
                % (n, p, strict.rate, naive.rate),
            )

    def test_what_it_buys_instead_is_the_elimination_of_silent_wrong_answers(self):
        naive = cell(20, 0.10, L.NAIVE)
        verified = cell(20, 0.10, L.VERIFY_ONLY)
        self.assertGreater(naive.silent_wrong, 30)
        self.assertEqual(verified.silent_wrong, 0)

    def test_the_semantic_assert_adds_nothing_over_the_grammar_without_a_ladder(self):
        """With retries: 0 the two arms are identical — every rejection is fatal anyway.

        Worth stating because it means a pack author cannot read `verify_only == jig`'s
        assert as 'the assert is what saves me'. The assert only pays once there is a
        rung behind it; before that it is just a different way to die.
        """
        for n, p in ((20, 0.10), (50, 0.02)):
            self.assertEqual(cell(n, p, L.GRAMMAR_ONLY).rate,
                             cell(n, p, L.VERIFY_ONLY).rate)


# --------------------------------------------------- the invariants, at fifty nodes


class RejectedGenerationsNeverEnterALaterPrompt(unittest.TestCase):
    """ARCHITECTURE.md §3's anti-self-conditioning invariant, at scale rather than in a unit.

    `tests/test_invariants.py` proves it for one node and three rungs. This proves it
    across a 50-node run at p=0.30, where hundreds of generations are rejected and every
    surviving one is read by the next node's prompt — the only place a leak could hide.

    Every faulted generation `FlakyModel` produces carries a tracer tag outside its JSON.
    Measured: 570 rejected generations across 60 runs, zero tags in any prompt, any
    state, or any output.
    """

    def _tags(self, model):
        return {"%s-%s-%d" % (L.POISON_MARK, e.node, e.attempt)
                for e in model.emissions
                if e.fault is not None and e.fault != "wrong_value"}

    def test_no_rejected_generation_reaches_a_prompt_across_fifty_nodes(self):
        rejected = 0
        for seed in range(60):
            _, model, result = trial(50, L.JIG, seed, 0.30)
            tags = self._tags(model)
            rejected += len(tags)
            blob = json.dumps(result.state) if hasattr(result, "state") else ""
            for tag in tags:
                for index, call in enumerate(model.calls):
                    self.assertNotIn(
                        tag, call.prompt,
                        "seed %d: a rejected generation reached call %d's prompt"
                        % (seed, index),
                    )
                self.assertNotIn(tag, blob)
        self.assertGreater(rejected, 200, "the test never saw a rejection — vacuous")

    def test_the_invariant_is_bought_by_verification_and_not_for_free(self):
        """The same model against the permissive arm *does* poison state.

        Without `additionalProperties: false` a model that appends an invented field
        gets that field merged into run state, provenance and all. 26 of the 97 runs
        that completed carried a tracer tag in their final state. Verify-before-commit
        is what makes the invariant true, not the graph shape.
        """
        poisoned = completed = 0
        for seed in range(200):
            outcome, model, result = trial(20, L.NAIVE, seed, 0.10)
            if outcome not in ("success", "silent_wrong"):
                continue
            completed += 1
            if L.POISON_MARK in json.dumps(result.state):
                poisoned += 1
        self.assertGreater(completed, 50)
        self.assertGreater(poisoned, 10)


class TheHorizonStaysBounded(unittest.TestCase):
    """ARCHITECTURE.md §3: 'the small model never plans; short, fresh context per node.'

    The structural claim underneath every number above. If context grew with N, a
    50-node run would be a long-context problem and small-model reliability would decay
    for a reason no ladder can fix.
    """

    def test_prompt_size_does_not_grow_with_the_horizon(self):
        _, model, _ = trial(50, L.JIG, 5, 0.0)
        lengths = [len(call.prompt) for call in model.calls]
        self.assertEqual(len(lengths), 50)
        # Measured 182 -> 189 bytes across fifty nodes; the growth is the field name
        # going from "v1" to "v50", nothing else.
        self.assertLess(max(lengths) - min(lengths), 20)

    def test_a_fifty_node_prompt_is_no_larger_than_a_five_node_one(self):
        _, short, _ = trial(5, L.JIG, 5, 0.0)
        _, long_run, _ = trial(50, L.JIG, 5, 0.0)
        self.assertLess(max(len(c.prompt) for c in long_run.calls),
                        max(len(c.prompt) for c in short.calls) + 20)

    def test_even_a_retry_prompt_stays_bounded(self):
        """The one thing that legitimately grows a prompt is the feedback block."""
        _, model, _ = trial(50, L.JIG, 11, 0.30)
        lengths = [len(call.prompt) for call in model.calls]
        self.assertLess(max(lengths), 2 * min(lengths))


class OnFailRoutingSurvivesFiftyNodes(unittest.TestCase):
    """A spent node 40 links deep still takes its declared failure edge."""

    def test_a_spent_ladder_diverts_instead_of_raising(self):
        measured = cell(50, 0.30, L.JIG_ON_FAIL, trials=150)
        self.assertEqual(measured.raised, 0)
        self.assertGreater(measured.gave_up, 0)
        self.assertEqual(measured.successes + measured.gave_up, measured.trials)

    def test_routing_does_not_change_the_answer_only_how_the_run_ends(self):
        """Same seeds, same correct-answer count — `on_fail` is presentation, not rescue."""
        for p in (0.10, 0.30):
            self.assertEqual(cell(50, p, L.JIG_ON_FAIL, trials=150).successes,
                             cell(50, p, L.JIG, trials=150).successes)

    def test_the_failure_is_attributed_to_the_node_that_spent_its_ladder(self):
        for seed in range(150):
            outcome, model, result = trial(50, L.JIG_ON_FAIL, seed, 0.30)
            if outcome != "gave_up":
                continue
            self.assertEqual(len(result.failures), 1)
            failure = result.failures[0]
            self.assertEqual(failure.node, result.path[-2])
            self.assertEqual(failure.attempts, 3)
            self.assertNotIn(L.POISON_MARK, json.dumps(result.state))
            return
        self.fail("no run gave up — the test is vacuous")


# ------------------------------------------------------ what a long run can now tell you


class ARescuedRunSaysHowCloseItCame(unittest.TestCase):
    """FIXED: retries used to leave no trace, so a rescued run looked like a clean one.

    README.md sells auditability: *"we can't prove what our AI did"* is problem 4. But
    `RunResult` records a node's failure only when the ladder is *spent*, so a 50-node
    run in which 40 nodes were rejected once and recovered reported `failures == []`,
    `steps == 51`, and a provenance map identical to a run where nothing went wrong. The
    two things an operator most wants from a long run — how close it came to failing,
    and what it cost — were in neither the result nor the checkpoint.

    `RunResult.attempts` now carries the generations each `generate` node spent. A run
    where every value is 1 is a run that never stumbled; a node sitting at `retries + 1`
    is a node that was one rung from taking its `on_fail` edge, which is the signal that
    a pack's reliability is degrading *before* the run that fails.

    Measured at N=50 p=0.30 over the 33 of 80 seeds that finished after spending at
    least five re-samples: the median rescued run has 14 of its 50 nodes above one
    attempt, 3 of them at the ladder's ceiling, and 67 generations against an ideal of
    50. None of that appears in `failures`, because nothing failed.
    """

    def _rescued(self):
        for seed in range(80):
            outcome, model, result = trial(50, L.JIG, seed, 0.30)
            if outcome == "success" and model.wasted_calls() >= 5:
                return model, result
        self.fail("no run needed retries — the test would be vacuous")

    def test_a_run_that_needed_retries_says_which_nodes_needed_them(self):
        model, result = self._rescued()

        retried = {name: spent for name, spent in result.attempts.items() if spent > 1}

        self.assertEqual(result.failures, [], "nothing here was diverted")
        self.assertTrue(retried, "the run was rescued and the result does not show it")
        self.assertEqual(len(result.attempts), 50, "every generate node is counted")

    def test_the_counts_are_the_generations_that_were_actually_spent(self):
        """Checked against the model's own call log, not against the walker's arithmetic."""
        model, result = self._rescued()
        self.assertEqual(sum(result.attempts.values()), model.call_count)

    def test_a_clean_run_is_visibly_clean(self):
        _, _, result = trial(50, L.JIG, 5, 0.0)
        self.assertEqual(sorted(set(result.attempts.values())), [1])

    def test_no_node_reports_more_attempts_than_its_ladder_allows(self):
        for seed in range(20):
            outcome, _, result = trial(50, L.JIG, seed, 0.30)
            if outcome == "raised":     # a RunError, not a RunResult
                continue
            self.assertLessEqual(max(result.attempts.values()), 3)

    def test_a_store_that_records_them_is_offered_them(self):
        """The walker hands the counts to any store whose `save` accepts them.

        `jig.state.Store` predates the field and does not, so it is offered nothing and
        keeps working — the column is the store's to add, and this is the assertion that
        will notice when it does.
        """
        seen = []

        class RecordingStore:
            def save(self, run_id, step, node, next_node, state, path=None,
                     provenance=None, failures=None, output=None, pack=None,
                     pack_version=None, attempts=None):
                seen.append(attempts)

        run(pack_for(20, L.JIG), L.FlakyModel(seed=7, p=0.30), inputs={"v0": 7},
            run_id="recorded", store=RecordingStore())

        self.assertTrue(seen)
        self.assertGreater(max(sum(record.values()) for record in seen), 20)

    def test_a_store_that_predates_the_field_still_checkpoints(self):
        store = Store(os.path.join(_ROOT, "audit.sqlite"))
        try:
            model = L.FlakyModel(seed=7, p=0.30)
            run(pack_for(20, L.JIG), model, inputs={"v0": 7}, run_id="audited",
                store=store)
            self.assertGreater(model.wasted_calls(), 0)
            self.assertEqual(len(store.history("audited")), 21)
        finally:
            store.close()


class CheckpointStorageGrowsWithTheSquareOfTheHorizon(unittest.TestCase):
    """FINDING (low): every checkpoint holds the whole of state, so cost is O(N^2).

    `graph._checkpoint` writes the full state dict after every node, and this chain's
    state grows by one key per node. Measured total persisted state bytes for one clean
    run: N=5 -> 260, N=20 -> 2,631, N=50 -> 14,961 — a 10x horizon costs 58x the storage.

    Harmless at N=50 with three-digit values. It is not harmless for the workflows
    ARCHITECTURE.md targets (invoice extraction, ticket triage), where a node commits a document
    rather than an integer and every later checkpoint re-writes every earlier document.
    """

    def _persisted(self, n):
        store = Store(os.path.join(_ROOT, "growth%d.sqlite" % n))
        try:
            run(pack_for(n, L.JIG), L.FlakyModel(seed=3, p=0.0),
                inputs={"v0": 7}, run_id="g%d" % n, store=store)
            return sum(len(json.dumps(cp.state)) for cp in store.history("g%d" % n))
        finally:
            store.close()

    def test_persisted_bytes_grow_quadratically_not_linearly(self):
        small, large = self._persisted(5), self._persisted(50)
        linear = small * (50 / 5.0)
        self.assertGreater(large, 3 * linear,
                           "persisted %d bytes at N=50 vs %d if growth were linear"
                           % (large, linear))
        # Quadratic within a constant factor: 50^2 / 5^2 = 100, measured 57.5x.
        self.assertLess(large, small * 150)


class ExtractJsonPrefersTheObjectTheModelAuthored(unittest.TestCase):
    """FIXED: an echoed format example used to be committed in preference to the answer.

    `verify.extract_json` is deliberately forgiving about *finding* an object in prose,
    and it used to take the first balanced `{...}` span. A small model that restates the
    requested shape before answering — one of the most common small-model habits there
    is — got its restatement committed instead of its answer. In a pack with a semantic
    `assert:` the ladder caught it and the only cost was generations; in a pack without
    one the wrong object was committed having passed "verification", which is the one
    thing verify-before-commit exists to prevent.

    The scan now walks the balanced spans from the end. What a model says last is its
    answer; what comes before it is preamble, quoted input, or a format example, and
    none of those are things the node asked it to produce.

    Selection is deliberately blind to the node's schema. "Prefer the object that
    validates" reads better until the model's own answer is the imperfect one: the echo
    then outranks it, and a rejection the ladder could have repaired becomes a silent
    commit of quoted input — the same defect wearing a schema.
    """

    def test_the_real_answer_wins_over_the_echoed_example(self):
        text = 'The format is {"v7": 0}. My answer: {"v7": 913}'
        self.assertEqual(extract_json(text), {"v7": 913})

    def test_a_pack_without_an_assert_commits_the_answer_and_not_the_echo(self):
        node = Node(name="n7", type="generate", prompt="x",
                    grammar=L.node_schema(7, strict=True))
        self.assertEqual(verify(node, 'Format: {"v7": 0}\nAnswer: {"v7": 913}', {}),
                         {"v7": 913})

    def test_a_fence_after_prose_is_reached_now_that_the_scan_runs_backwards(self):
        """`_unfence` only fires on a leading fence; prose-then-fence is the common shape."""
        text = 'user said {"v7": 0}\n```json\n{"v7": 913}\n```'
        self.assertEqual(extract_json(text), {"v7": 913})

    def test_an_unparseable_trailing_span_falls_back_instead_of_failing(self):
        """Preferring the last object must not mean giving up at the last object."""
        text = 'Answer: {"v7": 913}. Schema for reference: {v7: integer}'
        self.assertEqual(extract_json(text), {"v7": 913})

    def test_an_unparseable_leading_span_no_longer_costs_a_rung(self):
        """The old scan returned one candidate; a recoverable generation cost a retry."""
        self.assertEqual(
            extract_json('Using the schema {v7: integer} I get {"v7": 913}'),
            {"v7": 913},
        )

    def test_text_with_no_object_in_it_is_still_rejected(self):
        with self.assertRaises(Rejected):
            extract_json("I could not work out step 7, sorry.")


# ------------------------------------------------------------------------- the report


def report():
    """Print every table in this file. Run as a module to see the measurement."""
    setUpModule()
    try:
        print("\n== end-to-end success rate, %d seeds per cell ==\n" % TRIALS)
        header = "  N    p     " + "".join("%-14s" % arm.name for arm in L.ARMS)
        print(header + "(1-p)^N")
        for n in HORIZONS:
            for p in ERROR_RATES:
                row = "%3d  %.2f   " % (n, p)
                for arm in L.ARMS:
                    row += "%-14s" % ("%.1f%%" % (100 * cell(n, p, arm).rate))
                print(row + "%.1f%%" % (100 * (1 - p) ** n))
            print()

        print("== jig vs the analytical curve, and the effective per-step error ==\n")
        print("  N    p    jig     naive   (1-p)^N  jig-analytic  p_eff(jig)  p_eff(naive)")
        for n in HORIZONS:
            for p in ERROR_RATES[1:]:
                j, b = cell(n, p, L.JIG), cell(n, p, L.NAIVE)
                print("%3d  %.2f  %5.1f%%  %5.1f%%  %5.1f%%     %+6.1f       %.4f      %.4f"
                      % (n, p, 100 * j.rate, 100 * b.rate, 100 * j.analytic,
                         100 * (j.rate - j.analytic), j.effective_p, b.effective_p))
            print()

        print("== where it breaks down, and what fixes it ==\n")
        for n, p in ((20, 0.30), (50, 0.10), (50, 0.30)):
            two = cell(n, p, L.JIG)
            five = cell(n, p, L.JIG_LADDER_5, trials=150)
            stuck = cell(n, p, L.JIG, trials=150, stubborn=True)
            hinted = cell(n, p, L.JIG, trials=150, stubborn=True,
                          model_type=StubbornUnlessAsked)
            stage = cell(n, p, L.JIG_TWO_STAGE, trials=150)
            print("N=%2d p=%.2f  retries=2 %5.1f%%   retries=5 %5.1f%%   "
                  "stubborn %5.1f%%   stubborn+hint %5.1f%%   two_stage %5.1f%%   "
                  "calls/success %.2fx"
                  % (n, p, 100 * two.rate, 100 * five.rate, 100 * stuck.rate,
                     100 * hinted.rate, 100 * stage.rate, two.call_overhead))
    finally:
        tearDownModule()


if __name__ == "__main__":
    report()
