"""Every error a *run* can raise, in one place.

The walker, the codegen, the verifier and the eval runner all raise these, and several
of them import each other. Keeping the hierarchy in a leaf module is what stops that from
becoming an import cycle. Pack-loading errors are different — they belong to `jig.pack`,
because they happen before a run exists.
"""

__all__ = [
    "AssertFailed",
    "DanglingEdge",
    "DeadEnd",
    "ExprError",
    "JigError",
    "MaxStepsExceeded",
    "MissingVariable",
    "NodeFailed",
    "RunError",
]


class JigError(Exception):
    """Base class for everything jig raises at run time."""


class RunError(JigError):
    """A run could not continue."""


class MissingVariable(RunError):
    """A prompt referenced a state variable that is not there."""


class ExprError(RunError):
    """An assert expression is unsupported, or references something missing."""


class DeadEnd(RunError):
    """No outgoing edge matched the current state."""


class DanglingEdge(RunError):
    """An edge points at a node that does not exist."""


class MaxStepsExceeded(RunError):
    """The walk ran longer than the graph's step budget — probably an unbroken loop."""


class NodeFailed(RunError):
    """A node exhausted its retry ladder and has no `on_fail` edge.

    `node` is the node name, `attempts` the number of generations spent, and `reason`
    the last verification failure.
    """

    def __init__(self, node, reason, attempts=0):
        self.node = node
        self.reason = reason
        self.attempts = attempts
        RunError.__init__(
            self, "node %r failed after %d attempt(s): %s" % (node, attempts, reason)
        )


class AssertFailed(RunError):
    """An `assert` node's expression was false and the node has no `on_fail`."""

    def __init__(self, node, expression):
        self.node = node
        self.expression = expression
        RunError.__init__(self, "assert node %r failed: %s" % (node, expression))
