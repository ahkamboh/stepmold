"""`jig build` — compile a pack from a task description and gold examples.

The runtime half of jig runs a pack. This half writes one.

Nothing here is imported by `jig run`, `jig eval` or `jig validate`. That separation is
deliberate and enforced by a test: the runtime ships to a client box and must stay
dependency-free and small, while the compiler runs once on a machine you control.
"""

from .spec import BuildError, FieldSpec, GraphPlan, NodePlan, TaskSpec

__all__ = ["BuildError", "FieldSpec", "GraphPlan", "NodePlan", "TaskSpec"]
