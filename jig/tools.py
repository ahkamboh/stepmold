"""Tools: the only way a pack can reach outside itself.

A pack is text. It has no code in it and never will — that is what lets a pack be
reviewed in a diff, copied to a machine that cannot install anything, and read by a
runtime written in another language. So a pack cannot *contain* an action. It can only
*name* one.

The host supplies the actions:

    registry = ToolRegistry()

    @registry.register("lookup_order", reads=["order_id"], writes=["order_total"])
    def lookup_order(order_id):
        return {"order_total": db.total_for(order_id)}

    run(pack, model, inputs, tools=registry)

and a `tool` node in the graph names one:

    fetch:
      type: tool
      tool: lookup_order

Two properties fall out of that shape, and both are the reason it is shaped that way.

**A pack can only call what the host already allowed.** There is no import, no dotted
path, no eval — a name that was never registered is refused at load, before the run
starts. A pack you did not write cannot reach anything you did not hand it. This is an
allowlist by construction rather than a sandbox that has to hold.

**A tool call is a side effect, and side effects must not happen twice.** A run that
sends an email, then crashes, then resumes must not send it again. Every completed call
is recorded in the checkpoint with the arguments it was given, and a resumed run replays
the recorded result instead of calling again. A tool that is genuinely safe to repeat can
say so with `idempotent=True` and skip the bookkeeping.

Nothing here calls a model. A tool node is deterministic: same state in, same call out.
That is what makes an action auditable in a way a generation is not.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .errors import RunError

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolError",
    "ToolNotRegistered",
    "ToolFailed",
    "ToolContract",
]


class ToolError(RunError):
    """A tool could not be run, or its result could not be trusted."""


class ToolNotRegistered(ToolError):
    """A pack named a tool the host did not supply.

    Raised at load time wherever possible. A run that discovers this halfway through has
    already done part of a job it cannot finish.
    """


class ToolFailed(ToolError):
    """The tool raised. Carries the node so the failure can be routed like any other."""

    def __init__(self, node, tool, cause):
        self.node = node
        self.tool = tool
        self.cause = cause
        ToolError.__init__(self, "tool %r on node %r failed: %s: %s"
                           % (tool, node, type(cause).__name__, cause))


class ToolContract(ToolError):
    """The tool returned something its declaration said it would not.

    Checked for the same reason a generation is checked: a value that does not match what
    the graph was built around is a bad result, whether a model or a function produced it.
    """


@dataclass(frozen=True)
class Tool:
    """One action the host has made available to a pack."""

    name: str
    call: Callable
    reads: List[str] = field(default_factory=list)
    writes: List[str] = field(default_factory=list)
    idempotent: bool = False
    description: str = ""

    def invoke(self, state, node_name):
        """Run the tool against `state` and return the object to commit.

        `reads` is the whole argument list: a tool is given exactly the state it declared
        it needs and nothing else. That keeps a tool's dependencies visible in the graph
        instead of hidden in its body, and stops a tool quietly coupling to a field the
        author did not intend.
        """
        missing = [key for key in self.reads if key not in state]
        if missing:
            raise ToolContract(
                "tool %r on node %r needs %s, which state does not have (it has: %s)"
                % (self.name, node_name, ", ".join(sorted(missing)),
                   ", ".join(sorted(state)) or "nothing")
            )
        arguments = {key: state[key] for key in self.reads}
        try:
            result = self.call(**arguments)
        except Exception as exc:  # noqa: BLE001 - any failure is the node's failure
            raise ToolFailed(node_name, self.name, exc)
        return self._checked(result, node_name)

    def _checked(self, result, node_name):
        if result is None:
            result = {}
        if not isinstance(result, dict):
            raise ToolContract(
                "tool %r on node %r returned %s; a tool must return a dict of the state "
                "it writes, or None for nothing"
                % (self.name, node_name, type(result).__name__)
            )
        if self.writes:
            missing = [key for key in self.writes if key not in result]
            if missing:
                raise ToolContract(
                    "tool %r on node %r declares it writes %s but returned %s"
                    % (self.name, node_name, ", ".join(sorted(self.writes)),
                       ", ".join(sorted(result)) or "nothing")
                )
            extra = [key for key in result if key not in self.writes]
            if extra:
                raise ToolContract(
                    "tool %r on node %r returned undeclared key(s) %s. A tool's `writes` "
                    "is the contract the graph is built around — widen it deliberately "
                    "rather than by accident."
                    % (self.name, node_name, ", ".join(sorted(extra)))
                )
        return dict(result)


class ToolRegistry:
    """The set of actions a host is willing to let a pack take.

    Deliberately not a module-level default. There is no ambient registry a pack could
    reach by existing; a host that wants a pack to be able to act has to say so, per run.
    """

    def __init__(self):
        self._tools = {}

    def register(self, name, reads=(), writes=(), idempotent=False, description=""):
        """Decorator: expose one function to packs under `name`.

        `reads` names the state this tool needs and `writes` names what it produces, both
        checked at load time against the graph. Declaring them is not bureaucracy — it is
        what lets `jig validate` catch a tool wired to a field that does not exist yet,
        before a run has done half a job.
        """

        def decorate(function):
            if name in self._tools:
                raise ToolError("tool %r is already registered" % name)
            reads_list = list(reads)
            if not reads_list:
                # A tool that named nothing gets its parameter list read off the function,
                # which is right often enough to be worth doing and always visible in
                # `jig validate` when it is not.
                try:
                    signature = inspect.signature(function)
                    reads_list = [
                        parameter.name
                        for parameter in signature.parameters.values()
                        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD,
                                              parameter.KEYWORD_ONLY)
                    ]
                except (TypeError, ValueError):
                    reads_list = []
            self._tools[name] = Tool(
                name=name, call=function, reads=reads_list, writes=list(writes),
                idempotent=bool(idempotent),
                description=description or (function.__doc__ or "").strip().split("\n")[0],
            )
            return function

        return decorate

    def add(self, name, function, **kwargs):
        """Register without the decorator, for a function you did not define."""
        return self.register(name, **kwargs)(function)

    def get(self, name, node_name=None):
        tool = self._tools.get(name)
        if tool is None:
            where = (" on node %r" % node_name) if node_name else ""
            raise ToolNotRegistered(
                "no tool named %r%s. A pack can only call what the host registered "
                "(available: %s). Register it before the run, or remove the node."
                % (name, where, ", ".join(sorted(self._tools)) or "none")
            )
        return tool

    def has(self, name):
        return name in self._tools

    @property
    def names(self):
        return sorted(self._tools)

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name):
        return name in self._tools
