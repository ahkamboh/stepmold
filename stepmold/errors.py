"""Every error a *run* can raise, in one place.

The walker, the codegen, the verifier and the eval runner all raise these, and several
of them import each other. Keeping the hierarchy in a leaf module is what stops that from
becoming an import cycle. Pack-loading errors are different — they belong to `stepmold.pack`,
because they happen before a run exists.
"""

__all__ = [
    "AssertFailed",
    "BackendError",
    "DanglingEdge",
    "DeadEnd",
    "ExprError",
    "StepmoldError",
    "MaxStepsExceeded",
    "MissingVariable",
    "NodeFailed",
    "RunError",
    "RunIdInUse",
    "UnknownRun",
]


class StepmoldError(Exception):
    """Base class for everything stepmold raises at run time."""


class RunError(StepmoldError):
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

    `feedback` is the *safe half* of that reason — `verify.Rejected.feedback`, which says
    what was wrong without quoting what the model said. It exists because this exception
    is now read by two audiences with different rights: an operator debugging a pack, who
    gets `reason` in `RunResult.failures`, in the checkpoint and in a DEBUG log line; and
    a default-level log line, which an operator may ship to a collector and which must
    therefore carry no model output at all. Optional, and None when nobody supplied one —
    the walker says "detail at DEBUG" rather than guessing which half it is holding.
    """

    def __init__(self, node, reason, attempts=0, feedback=None):
        self.node = node
        self.reason = reason
        self.attempts = attempts
        self.feedback = feedback
        RunError.__init__(
            self, "node %r failed after %d attempt(s): %s" % (node, attempts, reason)
        )


class AssertFailed(RunError):
    """An `assert` node's expression was false and the node has no `on_fail`."""

    def __init__(self, node, expression):
        self.node = node
        self.expression = expression
        RunError.__init__(self, "assert node %r failed: %s" % (node, expression))


class RunIdInUse(RunError):
    """A fresh run was started under a run id that already has checkpoints.

    Welding two runs into one chain lets `resume` hand back the earlier run's output —
    one caller's data returned as another's, with a zero exit code. Refuse instead.
    """


class UnknownRun(RunError):
    """A run id with no checkpoint behind it."""


class BackendError(RunError):
    """A real inference backend could not be reached, or answered with nonsense."""
