"""Exactly-once, with the money in view.

Run it:

    python3 examples/refund_desk/once_only.py

The run is killed inside the few microseconds that matter: after `issue_refund` has
returned and before the walk has left the `refund` node. Then it is resumed. The ledger
has one entry both times.

Committed state cannot answer "did the money move?" — it records what a call *returned*,
never that it *happened*, so a resumed run that trusted state alone would call again. What
stops it is a row the walker writes before the commit (`jig/graph.py`, the `tool` branch:
"Written down before the commit and before the edge, with `next_node` still this node
because the walk has not left it"), carrying the node, the tool, the arguments it was
given, and the result. The resume lands back on the node, finds the row, and replays it.

`fetch_order` gets none of this, and does not want it: it is declared `idempotent=True`,
so its checkpoint stays empty and a resumed run simply reads the order again. The
bookkeeping is the price of not being repeatable, and only the tool that is not
repeatable pays it.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from jig.cli import resolve_model                # noqa: E402
from jig.graph import run                        # noqa: E402
from jig.pack import load_pack                   # noqa: E402
from jig.state import Store, resume              # noqa: E402

from tools import RefundDesk                     # noqa: E402

ORDER = "R-1001"
MESSAGE = {"message": "The mug arrived smashed in three pieces.", "order_id": ORDER}


class Crash(Exception):
    """Not a jig error. The worker died; it did not fail, and nothing catches this."""


class DyingStore(Store):
    """A real store whose worker is killed at the save that would settle the refund.

    A checkpointed run cannot be interrupted convincingly from outside — the window is a
    few microseconds wide and the store is the only thing called inside it. Staging the
    kill from here leaves on disk exactly what a real crash would have left.
    """

    def save(self, **checkpoint):
        if checkpoint.get("node") == "refund" \
                and checkpoint.get("next_node") == "done":
            raise Crash("worker killed after the refund, before the edge")
        return Store.save(self, **checkpoint)


def main():
    desk = RefundDesk()
    pack = load_pack(HERE, tools=desk.registry())
    model = resolve_model(None, pack)
    store = DyingStore(":memory:")

    try:
        run(pack, model, dict(MESSAGE), run_id="r", store=store,
            tools=desk.registry())
    except Crash as exc:
        print("crashed:", exc)

    checkpoint = store.latest("r")
    print("ledger after the crash    %r" % (desk.ledger,))
    print("checkpoint next_node      %s" % checkpoint.next_node)
    print("checkpoint tool_calls     %r" % (checkpoint.tool_calls,))
    print()

    # The supervisor brings a live worker up against the same store.
    store.__class__ = Store
    result = resume(pack, model, "r", store, tools=desk.registry())

    print("resumed to                %s" % result.end_node)
    print("ledger after the resume   %r" % (desk.ledger,))
    print("refunds for %s        %d" % (ORDER, len(desk.refunds_for(ORDER))))
    print("output                    %r" % (result.output,))
    store.close()


if __name__ == "__main__":
    main()
