"""The confidence gate, on this pack, with the money in view.

Run it:

    python3 examples/refund_desk/gate_demo.py

`approve` is the node the refund hangs off. This script runs it three times against the same
order and the same graph, changing one thing: whether the model's independent draws agree.

    draws agree      -> approve commits -> refund -> done         ledger: 1 entry
    draws disagree   -> verify raises Unsure -> on_unsure         ledger: 0 entries

That is the ordering the whole design is about. The gate is spent on the *decision*, and
the action is downstream of it, so a coin flip never reaches `issue_refund` at all — there
is nothing to undo, because nothing was done.

WHY THIS IS A SCRIPT AND NOT A LINE IN graph.yaml
-------------------------------------------------
`samples:` and `agree:` are not in `stepmold.pack._NODE_KEYS`, so a graph.yaml that names them
is refused at load — `stepmold.pack.Node` has no field for them yet. `stepmold.verify.gate_for`
reads them off the node with `getattr`, which is what lets this script attach them from
Python and get the real, shipped gate. `tests/test_verify.py:GatedNode` does exactly the
same thing, and for the same stated reason.

`on_unsure:` IS a pack key today (`stepmold/pack.py:_NODE_KEYS`, and the walker routes it at
`stepmold/graph.py`, the `except Unsure` clause), so the edge this demo takes is the one
graph.yaml already declares. Only the entrance to the gate is missing.
"""

import dataclasses
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from stepmold.graph import run                       # noqa: E402
from stepmold.log import configure                   # noqa: E402
from stepmold.model import FakeModel                 # noqa: E402
from stepmold.pack import Node, load_pack            # noqa: E402

from tools import RefundDesk                    # noqa: E402

ORDER = "R-1001"


def gate(pack, node_name, samples, agree):
    """Dial one node's gate to a chosen setting.

    graph.yaml already asks for samples: 3 / agree: 2 on `approve`, so calling this with
    those numbers changes nothing — it stays so the demo can vary them. It used to build a
    `Node` subclass, because `samples:` and `agree:` were not loader keys and a pack could
    not spell its own gate.
    """
    pack.nodes[node_name] = dataclasses.replace(
        pack.nodes[node_name], samples=samples, agree=agree)
    return pack


def model_for(approve_draws):
    """The scripted desk, with `approve` answering a list of draws in order."""
    return FakeModel({
        "Refund desk / classify / order %s" % ORDER: '{"kind": "refund"}',
        "Refund desk / assess / order %s" % ORDER:
            '{"justified": true, "grounds": "Delivered 3 days ago and arrived damaged."}',
        "Refund desk / approve / order %s" % ORDER: list(approve_draws),
    })


# stepmold is a library and logs nothing until an application switches it on. This script is
# the application, and the gate's own lines -- node.agreed, node.unsure, and the
# node.samples.blind warning below -- are most of what there is to see.
LOG = io.StringIO()
configure(level="info", stream=LOG)

GATE_EVENTS = ("node.samples.blind", "node.agreed", "node.unsure", "edge.on_unsure")


def gate_log():
    """The lines this attempt wrote about the gate, with the timestamp trimmed off."""
    lines = LOG.getvalue().splitlines()
    LOG.seek(0)
    LOG.truncate(0)
    return [line.split(" ", 1)[1] for line in lines
            if any(name in line for name in GATE_EVENTS)]


def attempt(label, approve_draws):
    desk = RefundDesk()
    pack = gate(load_pack(HERE), "approve", samples=3, agree=2)
    result = run(
        pack,
        model_for(approve_draws),
        {"message": "The mug arrived smashed in three pieces.", "order_id": ORDER},
        tools=desk.registry(),
    )
    print("%s" % label)
    print("  draws       %s" % ", ".join(
        json.dumps(json.loads(draw)["approved"]) for draw in approve_draws))
    for line in gate_log():
        print("  log         %s" % line)
    print("  path        %s" % " -> ".join(result.path))
    print("  end node    %s" % result.end_node)
    print("  ledger      %r" % (desk.ledger,))
    print("  refunded?   %s" % desk.is_refunded(ORDER))
    print()


AGREE = '{"approved": true, "rationale": "Damage three days after delivery is ours."}'
DISSENT = '{"approved": false, "rationale": "The customer may have dropped it."}'
AGREE_OTHER_WORDS = '{"approved": true, "rationale": "Cheaper to refund than to argue."}'
AGREE_THIRD_WORDS = '{"approved": true, "rationale": "The photo shows a clean break."}'


if __name__ == "__main__":
    # Two of three draws match, so the gate opens on the second draw and never pays for
    # the third: `verify.run_node` short-circuits the moment the answer cannot change.
    # Read `generations=2` in the node.agreed line -- `samples: 3` is a ceiling, not a bill.
    attempt("AGREED  (draw 1 == draw 2)", [AGREE, AGREE, DISSENT])

    # Three valid answers, three different objects, no group of two. Every draw was a
    # legal answer -- that is why this is `Unsure` and not `NodeFailed`, and why it takes
    # `on_unsure` rather than `on_fail`.
    attempt("UNSURE  (a real disagreement)", [AGREE, DISSENT, AGREE_OTHER_WORDS])

    # The one that will surprise you, and the reason this script prints three attempts.
    # All three draws say `approved: true`. The gate still calls it unsure, because it
    # compares the whole object it would commit (`verify._canonical`) and the rationale
    # differs three ways. Look at `distinct=3` in the node.unsure line.
    #
    # This is not a bug to route around; it is the cost of a wide node. A node whose
    # gate should be about the decision alone must commit the decision alone -- move
    # `rationale` to a node of its own, or drop it from the grammar.
    attempt("UNSURE  (same decision, three wordings)",
            [AGREE, AGREE_OTHER_WORDS, AGREE_THIRD_WORDS])
