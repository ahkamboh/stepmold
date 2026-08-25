"""The types every stage of `stepmold build` passes to the next.

The compiler is a pipeline, and these are the only things that cross between its stages:

    task.md + examples.jsonl  --analyze-->  TaskSpec
    TaskSpec                  --induce--->  GraphPlan
    TaskSpec + GraphPlan      --scribe--->  {node: prompt}
    TaskSpec + GraphPlan      --script--->  {key: answer}      (the offline model)
    all of the above          --assemble->  a pack directory

Two stages need a model and three do not, which is the whole reason the split is drawn
here: schema induction and fake-script generation are arithmetic over the gold examples,
and arithmetic does not need a frontier model, a GPU, or a network.

Frozen dataclasses throughout. A stage returns a new plan rather than editing the one it
was given, so a compile that fails halfway leaves nothing half-written — the same
verify-before-commit discipline the runtime uses, applied to the compiler.

Standard library only, like everything else in stepmold. `stepmold build` may cost more at build
time than `stepmold run` does at run time, but it must not drag a dependency into the package
that ships to a client box.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "FieldSpec",
    "TaskSpec",
    "NodePlan",
    "GraphPlan",
    "BuildError",
]


class BuildError(Exception):
    """A pack could not be compiled. Raised with what the author has to change."""


@dataclass(frozen=True)
class FieldSpec:
    """One field the finished pack must produce, as observed in the gold examples.

    `enum` is set when every observed value came from a small closed set — which is the
    single most valuable thing a compiler can infer, because an enum is what lets
    constrained decoding make a wrong answer unrepresentable rather than merely unlikely.
    """

    name: str
    type: str                                   # string | integer | number | boolean
    enum: Optional[List[Any]] = None
    optional: bool = False                      # absent or null in at least one example
    examples: List[Any] = field(default_factory=list)   # a few observed values, for prompts

    @property
    def schema(self):
        """This field as the JSON-Schema fragment stepmold's grammar subset accepts."""
        out = {"type": self.type}
        if self.enum:
            out["enum"] = list(self.enum)
        return out


@dataclass(frozen=True)
class TaskSpec:
    """Everything the compiler learned before it asked a model anything.

    `inputs` are the keys a run is given; `fields` are the keys it must end up producing.
    `cases` is the gold set verbatim — the contract the compiled pack is measured against,
    and the thing the compiler is never allowed to edit.
    """

    name: str
    description: str
    inputs: List[str]
    fields: List[FieldSpec]
    cases: List[dict]

    def field_named(self, name):
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise BuildError("no field named %r in the examples" % name)


@dataclass(frozen=True)
class NodePlan:
    """One generate node the compiler intends to emit.

    `writes` is the subset of the task's fields this node is responsible for. Every field
    must be written by exactly one node — a field written twice is ambiguous, and a field
    written by none never appears in the output. `assemble` enforces both.
    """

    name: str
    writes: List[str]
    purpose: str                # one line, in the imperative, for the prompt writer
    two_stage: bool = False
    reads: List[str] = field(default_factory=list)   # earlier fields this node needs


@dataclass(frozen=True)
class GraphPlan:
    """The decomposition, before any prompt has been written.

    Endings are separate from nodes because an end node writes nothing: it only projects
    what state already holds. Keeping them apart stops a planner from quietly assigning a
    field to a node that cannot produce one.
    """

    entry: str
    nodes: List[NodePlan]
    endings: List[str]
    edges: List[Dict[str, Any]] = field(default_factory=list)

    def node_named(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        raise BuildError("no node named %r in the plan" % name)

    @property
    def written_fields(self):
        seen = []
        for node in self.nodes:
            seen.extend(node.writes)
        return seen
