"""T10b — the seventh example pack: the one that acts.

`tests/test_example.py` proves the shipped pack *decides*. This one proves `refund_desk`
decides and then *does*, and holds the two properties that make doing safe:

* **exactly once, in the window the code actually protects** — a run killed after the
  call was *written down* and before the walk left its node is resumed, and the money moves
  one time. Asserted on the host's own ledger, because committed state records what a call
  returned and never that it happened.

  The qualifier is load-bearing and was added because a reviewer falsified the sentence
  without it. The record lands after the call returns, so a process killed inside that gap
  — call done, row not yet durable — resumes and calls again. jig cannot close that window
  from this side of the boundary: a tool that has already sent an email cannot be un-sent
  by anything jig writes afterwards. A tool whose repetition is harmless should say so with
  `idempotent=True`; one whose repetition is not should be written to be idempotent on its
  own key.
* **gated first** — the node that approves the refund is upstream of the node that issues
  it, so draws that disagree take `on_unsure` and nothing is issued at all.

Everything here is offline: a scripted `FakeModel` from the pack, and an in-memory order
store from `examples/refund_desk/tools.py`.
"""

import dataclasses
import importlib.util
import json
import logging
import os
import subprocess
import sys
import unittest

from jig.cli import resolve_model
from jig.eval import evaluate
from jig.graph import run
from jig.model import FakeModel
from jig.pack import GraphError, Node, load_pack
from jig.state import Store, resume
from jig.tools import ToolNotRegistered

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "examples", "refund_desk")


def desk_module():
    """Import the pack's host module the way `--tools <path>.py` does, without sys.path."""
    spec = importlib.util.spec_from_file_location(
        "_refund_desk_tools", os.path.join(PACK, "tools.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = desk_module()


def fresh_desk():
    """A desk nobody has refunded anything on yet — one per test, so state cannot leak."""
    return TOOLS.RefundDesk()


def load(desk=None):
    """Load the pack, and check its tool wiring when a desk is supplied.

    `load_pack(path, tools=registry)` is the only way `jig.pack.check_tools` runs; the
    `jig validate` subcommand takes no `--tools` flag (`jig/cli.py:_add_tools_option`
    adds it to `run` and `eval` only), so a pack's tool wiring is not checked from the
    command line at all today.
    """
    return load_pack(PACK, tools=desk.registry()) if desk else load_pack(PACK)


def scripted(pack):
    return resolve_model(None, pack)


DAMAGED = {"message": "The mug arrived smashed in three pieces.", "order_id": "R-1001"}


# --------------------------------------------------------------------------- scoring


class TestTheRefundDeskScores(unittest.TestCase):
    def test_it_scores_twelve_out_of_twelve(self):
        desk = fresh_desk()
        pack = load(desk)
        report = evaluate(pack, scripted(pack), tools=desk.registry())
        self.assertEqual((report.passed, report.total), (12, 12))
        self.assertTrue(report.passed_all)

    def test_the_cli_scores_it_with_the_hosts_tools(self):
        completed = subprocess.run(
            [sys.executable, "-m", "jig", "eval", PACK,
             "--tools", os.path.join(PACK, "tools.py")],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("12/12", completed.stdout)

    def test_without_the_tools_flag_the_pack_cannot_run_at_all(self):
        """The security model, from the operator's side: no flag, no actions.

        Documented as a limit in the pack's README because it is the first thing a reader
        copying `python3 -m jig eval examples/refund_desk` will hit.
        """
        completed = subprocess.run(
            [sys.executable, "-m", "jig", "eval", PACK],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("this pack needs tools; pass tools= to run()", completed.stdout)
        self.assertIn("failures by node: lookup=10", completed.stdout)

    def test_five_of_the_twelve_cases_actually_move_money(self):
        """Scoring a pack with a tool node is not a dry run — the evalset really acts."""
        desk = fresh_desk()
        pack = load(desk)
        evaluate(pack, scripted(pack), tools=desk.registry())
        self.assertEqual(
            sorted(entry["order_id"] for entry in desk.ledger),
            ["R-1001", "R-1004", "R-1006", "R-1009", "R-1012"],
        )

    def test_the_evalset_covers_every_ending_and_both_rescues(self):
        cases = load().evalset
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(1 for case in cases if case.end == "done"), 8)
        self.assertEqual(sum(1 for case in cases if case.end == "needs_human"), 4)
        self.assertEqual(
            sorted(case.input["order_id"] for case in cases if case.rescued),
            ["R-1010", "R-9999"],
        )


# ----------------------------------------------------------------------- pack shape


class TestTheShapeThatMakesItSafe(unittest.TestCase):
    def setUp(self):
        self.pack = load()

    def test_the_two_tool_nodes_name_the_two_registered_actions(self):
        tool_nodes = {node.name: node.tool
                      for node in self.pack.nodes.values() if node.type == "tool"}
        self.assertEqual(tool_nodes, {"lookup": "fetch_order", "refund": "issue_refund"})

    def test_the_read_is_idempotent_and_the_write_is_not(self):
        registry = fresh_desk().registry()
        self.assertTrue(registry.get("fetch_order").idempotent)
        self.assertFalse(registry.get("issue_refund").idempotent)

    def test_the_only_edge_into_refund_comes_from_approve(self):
        """The ordering claim, read off the graph rather than off the prose."""
        into_refund = [(edge.source, edge.when)
                       for edge in self.pack.edges if edge.target == "refund"]
        self.assertEqual(into_refund, [("approve", {"approved": True})])

    def test_approve_routes_both_of_its_doubts_to_a_human(self):
        approve = self.pack.nodes["approve"]
        self.assertEqual(approve.on_fail, "needs_human")
        self.assertEqual(approve.on_unsure, "needs_human")
        self.assertEqual(approve.assert_expr, "justified or not approved")

    def test_a_tool_the_host_never_registered_is_refused_at_load(self):
        """The allowlist, from the pack's side: an empty registry refuses this pack."""
        from jig.tools import ToolRegistry

        with self.assertRaises(ToolNotRegistered) as caught:
            load_pack(PACK, tools=ToolRegistry())
        self.assertIn("fetch_order", str(caught.exception))


# ------------------------------------------------------------------- exactly once


class Crash(Exception):
    """Not a `JigError`: the worker died, it did not fail. Nothing catches this."""


class DyingStore(Store):
    """A real store whose worker is killed at a save of the caller's choosing.

    The window that matters is a few microseconds wide — between `issue_refund`
    returning and the walk leaving the `refund` node — and the store is the only thing
    called inside it. Killing from there leaves on disk exactly what a real crash would.
    """

    def __init__(self):
        Store.__init__(self, ":memory:")
        self.die_on = None

    def save(self, **kwargs):
        if self.die_on is not None and self.die_on(kwargs):
            raise Crash("worker killed at node %r" % kwargs.get("node"))
        return Store.save(self, **kwargs)


class TestTheRefundIsIssuedExactlyOnce(unittest.TestCase):
    """Run it, kill it mid-refund, resume it: one refund, and it is the recorded one."""

    def setUp(self):
        self.desk = fresh_desk()
        self.registry = self.desk.registry()
        self.store = DyingStore()
        self.addCleanup(self.store.close)
        # After the call has returned and been written down, before the walk has left it.
        self.store.die_on = lambda kw: (kw.get("node") == "refund"
                                        and kw.get("next_node") == "done")

    def crash(self):
        pack = load(self.desk)
        with self.assertRaises(Crash):
            run(pack, scripted(pack), dict(DAMAGED), run_id="r", store=self.store,
                tools=self.registry)

    def finish(self):
        self.store.die_on = None
        pack = load(self.desk)
        return resume(pack, scripted(pack), "r", self.store, tools=self.registry)

    def test_the_money_moved_once_before_the_crash(self):
        self.crash()
        self.assertEqual(len(self.desk.refunds_for("R-1001")), 1)

    def test_the_call_is_on_disk_with_its_arguments_and_its_result(self):
        self.crash()
        [record] = self.store.latest("r").tool_calls
        self.assertEqual(record["node"], "refund")
        self.assertEqual(record["tool"], "issue_refund")
        self.assertEqual(record["args"], {"order_id": "R-1001", "order_total": 49.99})
        self.assertEqual(record["result"],
                         {"refund_id": "RF-R-1001", "refund_amount": 49.99})

    def test_the_checkpoint_still_points_at_the_refund_node(self):
        """So the resume lands back on it — and replays instead of calling."""
        self.crash()
        self.assertEqual(self.store.latest("r").next_node, "refund")

    def test_resuming_finishes_the_run_without_moving_the_money_again(self):
        self.crash()
        result = self.finish()
        self.assertEqual(len(self.desk.refunds_for("R-1001")), 1)
        self.assertEqual(len(self.desk.ledger), 1)
        self.assertEqual(result.end_node, "done")
        self.assertEqual(result.output["refund_id"], "RF-R-1001")

    def test_the_replayed_result_is_committed_as_if_the_call_had_just_happened(self):
        self.crash()
        result = self.finish()
        self.assertEqual(result.state["refund_amount"], 49.99)
        self.assertEqual(result.provenance["refund_id"], "refund")

    def test_the_record_is_cleared_once_the_node_has_been_left(self):
        self.store.die_on = None
        pack = load(self.desk)
        run(pack, scripted(pack), dict(DAMAGED), run_id="clean", store=self.store,
            tools=self.registry)
        for checkpoint in self.store.history("clean"):
            self.assertEqual(checkpoint.tool_calls, [])

    def test_a_second_call_would_have_been_loud(self):
        """The guard that proves the assertion above is not vacuous.

        `issue_refund` raises `AlreadyRefunded` on a repeat, so if the replay ever stopped
        working the resume would fail rather than quietly double-refund. Nothing in this
        file triggers it — this test is what says so on purpose.
        """
        self.crash()
        self.finish()
        with self.assertRaises(TOOLS.AlreadyRefunded):
            self.desk.issue_refund("R-1001", 49.99)

    def test_the_idempotent_lookup_is_not_recorded_at_all(self):
        """`fetch_order` declares itself safe to repeat, so it buys a cheaper checkpoint."""
        self.store.die_on = lambda kw: (kw.get("node") == "lookup"
                                        and kw.get("next_node") == "assess")
        pack = load(self.desk)
        with self.assertRaises(Crash):
            run(pack, scripted(pack), dict(DAMAGED), run_id="reads", store=self.store,
                tools=self.registry)
        self.assertEqual(self.store.latest("reads").tool_calls, [])


# ------------------------------------------------------------------- the gate first


@dataclasses.dataclass(frozen=True)
class GatedNode(Node):
    """A `Node` carrying the gate's keys, for as long as `jig.pack.Node` does not.

    The same stand-in `tests/test_verify.py` uses, and for the same reason: `verify.
    gate_for` reads `samples`/`agree` with `getattr`, so the runtime gate is real while
    the pack format cannot yet spell it. `examples/refund_desk/gate_demo.py` is the
    runnable version of everything below.
    """

    samples: int = 1
    agree: int = 0


AGREE = '{"approved": true, "rationale": "Damage three days after delivery is ours."}'
DISSENT = '{"approved": false, "rationale": "The customer may have dropped it."}'
AGREE_OTHER_WORDS = '{"approved": true, "rationale": "Cheaper to refund than to argue."}'


def gated_pack(desk, samples=3, agree=2):
    pack = load(desk)
    node = pack.nodes["approve"]
    fields = {f.name: getattr(node, f.name) for f in dataclasses.fields(node)}
    pack.nodes["approve"] = GatedNode(samples=samples, agree=agree, **fields)
    return pack


def gate_model(draws):
    return FakeModel({
        "Refund desk / classify / order R-1001": '{"kind": "refund"}',
        "Refund desk / assess / order R-1001":
            '{"justified": true, "grounds": "Delivered 3 days ago and arrived damaged."}',
        "Refund desk / approve / order R-1001": list(draws),
    })


class TestNothingIsIssuedOnACoinFlip(unittest.TestCase):
    def run_with(self, draws):
        desk = fresh_desk()
        result = run(gated_pack(desk), gate_model(draws), dict(DAMAGED),
                     tools=desk.registry())
        return result, desk

    def test_agreeing_draws_reach_the_refund_and_issue_it(self):
        result, desk = self.run_with([AGREE, AGREE, DISSENT])
        self.assertEqual(result.path,
                         ["classify", "lookup", "assess", "approve", "refund", "done"])
        self.assertEqual(len(desk.ledger), 1)

    def test_the_third_draw_is_never_paid_for_once_two_agree(self):
        """`samples: 3` is a ceiling on the bill, not the bill."""
        model = gate_model([AGREE, AGREE, DISSENT])
        desk = fresh_desk()
        run(gated_pack(desk), model, dict(DAMAGED), tools=desk.registry())
        approve_calls = [call for call in model.calls
                         if "Refund desk / approve" in call.prompt]
        self.assertEqual(len(approve_calls), 2)

    def test_disagreeing_draws_never_reach_the_refund_node(self):
        result, desk = self.run_with([AGREE, DISSENT, AGREE_OTHER_WORDS])
        self.assertNotIn("refund", result.path)
        self.assertEqual(result.end_node, "needs_human")
        self.assertEqual(desk.ledger, [])
        self.assertFalse(desk.is_refunded("R-1001"))

    def test_agreement_is_on_the_whole_object_not_on_the_decision(self):
        """The surprise worth knowing before you put a gate on a wide node.

        All three draws say `approved: true`. The gate compares the object it would
        commit (`verify._canonical`), the rationale differs three ways, and the refund is
        not issued. A gate that should be about the decision needs a node that commits
        the decision alone.
        """
        result, desk = self.run_with([
            AGREE,
            AGREE_OTHER_WORDS,
            '{"approved": true, "rationale": "The photo shows a clean break."}',
        ])
        self.assertEqual(result.end_node, "needs_human")
        self.assertEqual(desk.ledger, [])

    def test_an_unsure_case_is_escalated_rather_than_claimed(self):
        """`jig eval --tiers` would put this case in the escalated bucket, not the auto one."""
        from jig.eval import evaluate as score

        desk = fresh_desk()
        pack = gated_pack(desk)
        pack.evalset[:] = [
            type(pack.evalset[0])(
                name="unsure", input=dict(DAMAGED),
                expect={"kind": "refund"}, end="needs_human",
            )
        ]
        report = score(pack, gate_model([AGREE, DISSENT, AGREE_OTHER_WORDS]),
                       tools=desk.registry())
        self.assertEqual(report.tier_counts["escalated"], 1)
        self.assertEqual(desk.ledger, [])


class TestTheShippedScriptsRun(unittest.TestCase):
    """The two runnable demos README.md quotes. Their output is in the README verbatim,
    so a change that breaks them breaks a documented transcript."""

    def script(self, name):
        completed = subprocess.run(
            [sys.executable, os.path.join(PACK, name)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_gate_demo_prints_one_ledger_entry_and_two_empty_ones(self):
        out = self.script("gate_demo.py")
        self.assertEqual(out.count("refunded?   True"), 1)
        self.assertEqual(out.count("refunded?   False"), 2)
        self.assertIn("node.samples.blind", out)
        self.assertIn("node.agreed node=approve agreed=2 of=2 required=2 asked=3 "
                      "generations=2", out)
        self.assertIn("edge.on_unsure", out)

    def test_once_only_crashes_resumes_and_refunds_once(self):
        out = self.script("once_only.py")
        self.assertIn("checkpoint next_node      refund", out)
        self.assertIn("resumed to                done", out)
        self.assertIn("refunds for R-1001        1", out)


# ------------------------------------------------------------------ documented limits


class TestTheLimitsTheReadmeClaims(unittest.TestCase):
    """Every limit README.md states, pinned here so it cannot rot into a false claim."""

    def test_the_gate_keys_cannot_be_written_in_graph_yaml(self):
        import shutil
        import tempfile

        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        copy = os.path.join(directory, "refund_desk")
        shutil.copytree(PACK, copy)
        graph = os.path.join(copy, "graph.yaml")
        with open(graph) as handle:
            text = handle.read()
        with open(graph, "w") as handle:
            handle.write(text.replace("    on_unsure: needs_human",
                                      "    samples: 3\n    agree: 2\n"
                                      "    on_unsure: needs_human"))
        with self.assertRaises(GraphError) as caught:
            load_pack(copy)
        self.assertEqual(
            str(caught.exception),
            "graph.yaml: node 'approve' has unknown key(s): agree, samples",
        )

    def test_the_scripted_model_takes_no_sampling_hint_and_jig_says_so(self):
        """`node.samples.blind` — the warning that stops a gate reporting confidence it
        never measured. It fires here because `FakeModel.generate` has no `sampling`
        keyword, so on a real greedy backend the extra draws would be the first draw
        repeated. This pack's draws differ only because the script says they do."""
        from jig.codegen import accepts_sampling

        self.assertFalse(accepts_sampling(gate_model([AGREE])))

        # A handler on jig's own logger rather than `jig.log.configure`, which is the
        # CLI's switch and has no off: this must not leave logging on for the rest of
        # the suite. `event` is gated on `isEnabledFor(WARNING)`, which is already true
        # by default, so nothing about the run has to change to see this line.
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger = logging.getLogger("jig.verify")
        logger.addHandler(handler)
        try:
            desk = fresh_desk()
            run(gated_pack(desk), gate_model([AGREE, AGREE, DISSENT]), dict(DAMAGED),
                tools=desk.registry())
        finally:
            logger.removeHandler(handler)
        blind = [record for record in records
                 if getattr(record, "jig_event", None) == "node.samples.blind"]
        self.assertEqual(len(blind), 1)
        self.assertEqual(blind[0].jig_fields["node"], "approve")
        self.assertEqual(blind[0].jig_fields["samples"], 3)
        self.assertEqual(blind[0].jig_fields["model"], "FakeModel")

    def test_the_fake_script_is_keyed_on_the_order_not_on_case_order(self):
        """Which is what makes `jig run` of any one case answer that case."""
        with open(os.path.join(PACK, "fakes", "script.json")) as handle:
            script = json.load(handle)
        self.assertEqual(len(script), 30)
        self.assertTrue(all(key.startswith("Refund desk / ") for key in script))
        orders = {case.input["order_id"] for case in load().evalset}
        self.assertEqual(len(orders), 12)


class TestTheReadmeSaysWhatTheDirectoryHolds(unittest.TestCase):
    """The README quotes real transcripts; these are the ones cheap enough to re-check."""

    def setUp(self):
        with open(os.path.join(PACK, "README.md")) as handle:
            self.readme = handle.read()

    def test_the_file_table_lists_every_file_in_the_pack(self):
        on_disk = set()
        for directory, _, names in os.walk(PACK):
            for name in names:
                if name.endswith(".pyc"):
                    continue
                relative = os.path.relpath(os.path.join(directory, name), PACK)
                on_disk.add(relative)
        undocumented = sorted(
            name for name in on_disk
            if name != "README.md"
            and "`%s`" % name not in self.readme
            and "`%s/*%s`" % (os.path.dirname(name),
                              os.path.splitext(name)[1]) not in self.readme
        )
        self.assertEqual(undocumented, [])

    def test_it_quotes_the_refusal_the_gate_keys_actually_get(self):
        self.assertIn(
            "jig: pack error: graph.yaml: node 'approve' has unknown key(s): "
            "agree, samples",
            self.readme,
        )

    def test_it_quotes_the_score_the_pack_actually_gets(self):
        self.assertIn("refund_desk: 12/12 cases passed", self.readme)
        self.assertIn("refund_desk: 12 cases \u2014 10 auto, 2 escalated, 0 failed",
                      self.readme)

    def test_it_names_the_flag_without_which_nothing_runs(self):
        self.assertIn("--tools examples/refund_desk/tools.py", self.readme)
        self.assertIn("this pack needs tools; pass tools= to run()", self.readme)


if __name__ == "__main__":
    unittest.main()
