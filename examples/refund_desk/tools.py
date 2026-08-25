"""The host half of the refund desk: an order store, and the two actions a pack may take.

A pack is text and cannot contain code (`stepmold/tools.py`), so everything the desk can
actually *do* lives here, in the host's own module, and is handed to the runtime per run:

    python3 -m stepmold eval examples/refund_desk --tools examples/refund_desk/tools.py

`stepmold/cli.py:_load_registry` imports this file and looks for an attribute named `registry`
or `REGISTRY`. Nothing in graph.yaml points here — the pack names `fetch_order` and
`issue_refund` and the operator decides, on the command line, what those two names resolve
to. That is the whole allowlist: a name this module never registered cannot be reached by
any pack, and `stepmold.pack.check_tools` refuses the pack at load rather than at step four.

The two tools are deliberately not the same kind of thing, because that difference is what
the `idempotent=` flag is for:

| tool | idempotent | recorded in the checkpoint | a resumed run |
| --- | --- | --- | --- |
| `fetch_order` | yes | no | reads again |
| `issue_refund` | no | yes, before the commit | replays the recorded result |

`issue_refund` mutates two things a reader (and a test) can watch: it appends to
`RefundDesk.ledger`, one entry per *call*, and it flips the order's `refunded` flag. The
ledger counts calls rather than refunds on purpose — committed state records what a call
returned, never that it happened, so counting the entries is the only way to prove
exactly-once from the outside.
"""

from stepmold.tools import ToolRegistry


class OrderNotFound(LookupError):
    """No such order. Raised out of `fetch_order`, so the tool node's `on_fail` gets it.

    A plain exception, not a stepmold one: `Tool.invoke` wraps whatever the host's function
    raises in `ToolFailed`, and the walker routes that like any other node failure.
    """


class AlreadyRefunded(RuntimeError):
    """This order has been refunded once already.

    Here so that "the refund went out twice" is a loud failure rather than a quiet second
    row in the ledger. Nothing in the shipped evalset or in tests/test_example_refund.py
    ever triggers it — that is the point. If stepmold's exactly-once record ever stopped
    working, this is what a resumed run would hit.
    """


# The store. Eleven orders, one per evalset case; R-9999 is
# deliberately absent, which is the case that exercises the `lookup` node's `on_fail`.
SEED_ORDERS = {
    "R-1001": {"total": 49.99, "status": "delivered", "days_since_delivery": 3},
    "R-1002": {"total": 120.00, "status": "delivered", "days_since_delivery": 45},
    "R-1004": {"total": 32.00, "status": "lost", "days_since_delivery": 0},
    "R-1005": {"total": 18.50, "status": "delivered", "days_since_delivery": 1},
    "R-1006": {"total": 75.00, "status": "delivered", "days_since_delivery": 9},
    "R-1007": {"total": 210.00, "status": "delivered", "days_since_delivery": 60},
    "R-1008": {"total": 64.25, "status": "delivered", "days_since_delivery": 2},
    "R-1009": {"total": 15.00, "status": "delivered", "days_since_delivery": 6},
    "R-1010": {"total": 88.00, "status": "delivered", "days_since_delivery": 4},
    "R-1011": {"total": 42.00, "status": "in_transit", "days_since_delivery": 0},
    "R-1012": {"total": 149.99, "status": "delivered", "days_since_delivery": 2},
}


class RefundDesk:
    """The host's own state, and the registry that closes over it.

    A class rather than module-level dicts so a test can build a fresh desk per test and
    a reader can see where the state a tool touches actually lives. `registry()` returns a
    new `ToolRegistry` bound to *this* desk; two desks share nothing.
    """

    def __init__(self, orders=None):
        self.orders = {
            order_id: dict(row)
            for order_id, row in (SEED_ORDERS if orders is None else orders).items()
        }
        # One entry per call to `issue_refund`, appended before anything else can fail.
        self.ledger = []

    # ------------------------------------------------------------------ the actions

    def fetch_order(self, order_id):
        """Read one order out of the store. No side effect, so safe to repeat."""
        order = self.orders.get(order_id)
        if order is None:
            raise OrderNotFound(
                "no order %r in the store (%d known)" % (order_id, len(self.orders))
            )
        return {
            "order_total": order["total"],
            "order_status": order["status"],
            "days_since_delivery": order["days_since_delivery"],
        }

    def issue_refund(self, order_id, order_total):
        """Move the money. The one thing in any example pack that is not reversible."""
        order = self.orders.get(order_id)
        if order is None:
            raise OrderNotFound("no order %r in the store" % order_id)
        if order.get("refunded"):
            raise AlreadyRefunded(
                "order %r was already refunded (%d call(s) in the ledger)"
                % (order_id, len(self.refunds_for(order_id)))
            )
        refund_id = "RF-%s" % order_id
        self.ledger.append({"order_id": order_id, "refund_id": refund_id,
                            "amount": order_total})
        order["refunded"] = True
        return {"refund_id": refund_id, "refund_amount": order_total}

    # ------------------------------------------------------------------ observation

    def refunds_for(self, order_id):
        """Every ledger entry for one order. Length is the exactly-once assertion."""
        return [entry for entry in self.ledger if entry["order_id"] == order_id]

    def is_refunded(self, order_id):
        return bool(self.orders.get(order_id, {}).get("refunded"))

    # -------------------------------------------------------------------- the wiring

    def registry(self):
        """Everything this host is willing to let a pack do, and nothing else.

        `reads` is the tool's whole argument list — a tool is called with exactly the
        state it declared and nothing else — and `writes` is the contract the graph is
        built around, checked on the way back out (`stepmold/tools.py:Tool._checked`). Both are
        declared rather than inferred here so `stepmold validate --tools ...` can catch a tool
        wired to a field this graph never produces, before a run has done half a job.
        """
        registry = ToolRegistry()
        registry.add(
            "fetch_order", self.fetch_order,
            reads=["order_id"],
            writes=["order_total", "order_status", "days_since_delivery"],
            idempotent=True,
            description="Read one order from the order store.",
        )
        registry.add(
            "issue_refund", self.issue_refund,
            reads=["order_id", "order_total"],
            writes=["refund_id", "refund_amount"],
            idempotent=False,
            description="Refund an order. Moves money; not repeatable.",
        )
        return registry


# What `--tools examples/refund_desk/tools.py` picks up. One desk per process, which is
# what a CLI invocation wants; a test that needs isolation builds its own `RefundDesk`.
DESK = RefundDesk()
registry = DESK.registry()
