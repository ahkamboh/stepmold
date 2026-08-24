"""Loading and validating a JigPack.

A pack is a directory, not a database — it is meant to be read in a diff, checked into
a client's repo, and shipped as text (docs/PLAN.md §7.2):

    <pack>/
      manifest.yaml        name, version, entry node, model hint
      graph.yaml           nodes + edges (the compiled "plan")
      prompts/<node>.txt   one prompt template per generate node
      prompts/<node>.think.txt   optional think-stage template (T5)
      grammars/<node>.json one JSON Schema per generate node
      evalset.jsonl        the contract: {"input": {...}, "expect": {...}} per line

Everything a run needs is validated here, at load time, so the walker never has to ask
"does this node exist?" mid-run. Errors name the offending file and node.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .grammar import SchemaError, check_schema
from .yamlish import YamlError, parse as parse_yaml

__all__ = [
    "Edge",
    "EvalCase",
    "EvalsetError",
    "GrammarError",
    "GraphError",
    "ManifestError",
    "MissingArtifactError",
    "Node",
    "Pack",
    "PackError",
    "RESERVED_STATE_NAMES",
    "UnsafePath",
    "load_pack",
]

NODE_TYPES = ("generate", "assert", "end")

# Names jig binds in a run's scope for its own purposes. `codegen.think` renders the
# think template with a `scratchpad` of its own, so anything else that lands under that
# name is served to the model in the slot the prompt labels "your notes from thinking
# this through" — the most persuasive position in the whole prompt, filled with text that
# is not the model's reasoning at all.
#
# A pack cannot claim the name: `_build_node` refuses a node whose `output` is one of
# these. A *run input* with that name is the same hole from the other side, and closing
# it belongs where inputs enter a run (`graph.run`), not here — this tuple is public so
# that check has one list to read.
RESERVED_STATE_NAMES = ("scratchpad",)
DEFAULT_MAX_STEPS = 100
DEFAULT_MAX_TOKENS = 512
DEFAULT_THINK_MAX_TOKENS = 256
DEFAULT_RETRIES = 2

_NODE_KEYS = {
    "type", "output", "two_stage", "max_tokens", "think_max_tokens", "retries",
    "on_fail", "expr", "assert", "prompt", "grammar", "description",
}
_EDGE_KEYS = {"from", "to", "when", "description"}


class PackError(Exception):
    """Anything wrong with a pack on disk."""


class UnsafePath(PackError):
    """A pack referenced a file outside its own directory.

    A pack is untrusted input the moment it leaves the machine that compiled it
    (docs/PLAN.md §6 plans a registry, §7.2 describes copying packs between hosts), so
    every artifact reference must resolve inside the pack root.
    """


class MissingArtifactError(PackError):
    """A file the pack declares it needs is not there."""


class ManifestError(PackError):
    """`manifest.yaml` is missing a key or holds the wrong kind of value."""


class GraphError(PackError):
    """`graph.yaml` describes a graph that cannot be walked."""


class GrammarError(PackError):
    """A node's grammar file is not a schema jig can enforce."""


class EvalsetError(PackError):
    """`evalset.jsonl` has a line that is not a usable case."""


@dataclass(frozen=True)
class Node:
    """One step of the workflow."""

    name: str
    type: str
    prompt: Optional[str] = None
    think_prompt: Optional[str] = None
    grammar: Optional[dict] = None
    output: Any = None
    two_stage: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    think_max_tokens: int = DEFAULT_THINK_MAX_TOKENS
    retries: int = DEFAULT_RETRIES
    on_fail: Optional[str] = None
    expr: Optional[str] = None
    assert_expr: Optional[str] = None


@dataclass(frozen=True)
class Edge:
    """A transition, taken when `when` matches the current state (or is unset)."""

    source: str
    target: str
    when: Optional[dict] = None


@dataclass(frozen=True)
class EvalCase:
    """One line of `evalset.jsonl` — the hand-maintained contract."""

    input: dict
    expect: dict
    name: Optional[str] = None


@dataclass(frozen=True)
class Pack:
    """A loaded, validated pack. Immutable: a run never edits its own pack."""

    path: str
    name: str
    version: Any
    entry: str
    model: Optional[str]
    nodes: Dict[str, Node]
    edges: List[Edge]
    evalset: List[EvalCase] = field(default_factory=list)
    max_steps: int = DEFAULT_MAX_STEPS
    manifest: dict = field(default_factory=dict)

    def edges_from(self, node_name):
        """Outgoing edges for `node_name`, in the order they were declared."""
        return [edge for edge in self.edges if edge.source == node_name]


def load_pack(path):
    """Read the pack at `path`, validate it, and return a `Pack`."""
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        # Clipped: this argument is whatever was on the command line, and pasting a
        # ticket where a pack path belongs is a common enough slip that the message
        # must not be a megabyte long.
        raise MissingArtifactError("pack directory not found: %s" % _clip(path))

    manifest = _load_yaml(path, "manifest.yaml", ManifestError)
    graph = _load_yaml(path, "graph.yaml", GraphError)

    name = _require(manifest, "name", "manifest.yaml", str)
    entry = _require(manifest, "entry", "manifest.yaml", str)
    version = manifest.get("version", 1)
    model = manifest.get("model")
    if model is not None and not isinstance(model, str):
        raise ManifestError("manifest.yaml: 'model' must be a string, got %s"
                            % type(model).__name__)

    max_steps = graph.get("max_steps", DEFAULT_MAX_STEPS)
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise GraphError("graph.yaml: 'max_steps' must be a positive integer")

    nodes = _load_nodes(path, graph)
    edges = _load_edges(graph, nodes)

    if entry not in nodes:
        raise GraphError(
            "manifest.yaml: entry node %r is not defined in graph.yaml (nodes: %s)"
            % (entry, ", ".join(sorted(nodes)) or "none")
        )
    _check_reachable_targets(nodes, edges)

    return Pack(
        path=path,
        name=name,
        version=version,
        entry=entry,
        model=model,
        nodes=nodes,
        edges=edges,
        evalset=_load_evalset(path),
        max_steps=max_steps,
        manifest=manifest,
    )


# ------------------------------------------------------------------------ manifest


def _load_yaml(path, filename, error_type):
    full = os.path.join(path, filename)
    if not os.path.isfile(full):
        raise MissingArtifactError("%s: required file is missing (%s)" % (filename, full))
    with open(full, "r") as handle:
        text = handle.read()
    try:
        document = parse_yaml(text, filename=filename)
    except YamlError as exc:
        raise error_type("%s: %s" % (filename, exc))
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise error_type("%s: must be a mapping at the top level" % filename)
    return document


def _require(mapping, key, filename, kind):
    if key not in mapping or mapping[key] is None:
        raise ManifestError("%s: missing required key %r" % (filename, key))
    value = mapping[key]
    if not isinstance(value, kind):
        raise ManifestError(
            "%s: %r must be a %s, got %s"
            % (filename, key, kind.__name__, type(value).__name__)
        )
    return value


# --------------------------------------------------------------------------- nodes


def _load_nodes(path, graph):
    declared = graph.get("nodes")
    if not declared:
        raise GraphError("graph.yaml: declares no nodes")
    if not isinstance(declared, dict):
        raise GraphError("graph.yaml: 'nodes' must be a mapping of name -> node")

    nodes = {}
    for node_name, spec in declared.items():
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            raise GraphError("graph.yaml: node %r must be a mapping" % node_name)
        unknown = set(spec) - _NODE_KEYS
        if unknown:
            raise GraphError(
                "graph.yaml: node %r has unknown key(s): %s"
                % (node_name, ", ".join(sorted(unknown)))
            )
        node_type = spec.get("type")
        if node_type not in NODE_TYPES:
            raise GraphError(
                "graph.yaml: node %r has unknown type %r (expected one of %s)"
                % (node_name, node_type, ", ".join(NODE_TYPES))
            )
        nodes[node_name] = _build_node(path, node_name, node_type, spec)
    return nodes


def _build_node(path, node_name, node_type, spec):
    prompt = think_prompt = grammar = None
    if node_type == "generate":
        prompt = _read_text(path, spec.get("prompt") or "prompts/%s.txt" % node_name)
        grammar_relative = spec.get("grammar") or "grammars/%s.json" % node_name
        grammar = _read_json(path, grammar_relative)
        try:
            check_schema(grammar)
        except SchemaError as exc:
            raise GrammarError("%s: %s" % (grammar_relative, exc))
        think_relative = "prompts/%s.think.txt" % node_name
        if os.path.isfile(_resolve_inside(path, think_relative)):
            think_prompt = _read_text(path, think_relative)

    if spec.get("output") in RESERVED_STATE_NAMES:
        raise GraphError(
            "graph.yaml: node %r has output %r, which is a name jig reserves for its "
            "own scope — committing there would write the node's answer into the think "
            "stage's notes slot" % (node_name, spec.get("output"))
        )

    if node_type == "assert" and not spec.get("expr"):
        raise GraphError("graph.yaml: assert node %r needs an 'expr'" % node_name)

    return Node(
        name=node_name,
        type=node_type,
        prompt=prompt,
        think_prompt=think_prompt,
        grammar=grammar,
        output=spec.get("output"),
        two_stage=bool(spec.get("two_stage", False)),
        max_tokens=_positive_int(spec, "max_tokens", node_name, DEFAULT_MAX_TOKENS),
        think_max_tokens=_positive_int(
            spec, "think_max_tokens", node_name, DEFAULT_THINK_MAX_TOKENS
        ),
        retries=_positive_int(spec, "retries", node_name, DEFAULT_RETRIES, floor=0),
        on_fail=spec.get("on_fail"),
        expr=spec.get("expr"),
        assert_expr=spec.get("assert"),
    )


def _positive_int(spec, key, node_name, default, floor=1):
    value = spec.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < floor:
        raise GraphError(
            "graph.yaml: node %r: %r must be an integer >= %d" % (node_name, key, floor)
        )
    return value


def _clip(text, limit=120):
    """Keep a message bounded — the offending text is usually not ours."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _resolve_inside(path, relative):
    """Resolve `relative` against the pack root, refusing anything that escapes it.

    Rejects absolute paths (os.path.join silently discards the root for those), `..`
    traversal, and symlinks pointing outside — realpath resolves links on both sides, so
    a symlinked artifact is caught by the same containment check.
    """
    if not isinstance(relative, str) or not relative:
        raise UnsafePath("artifact reference must be a non-empty string")
    if os.path.isabs(relative) or os.path.splitdrive(relative)[0]:
        raise UnsafePath(
            "%s: absolute paths are not allowed in a pack" % _clip(relative)
        )
    root = os.path.realpath(path)
    full = os.path.realpath(os.path.join(root, relative))
    if full != root and not full.startswith(root + os.sep):
        raise UnsafePath("%s: resolves outside the pack directory" % _clip(relative))
    return full


def _read_text(path, relative):
    full = _resolve_inside(path, relative)
    if not os.path.isfile(full):
        raise MissingArtifactError(
            "%s: required file is missing (%s)" % (_clip(relative), _clip(full))
        )
    with open(full, "r") as handle:
        return handle.read()


def _read_json(path, relative):
    text = _read_text(path, relative)
    try:
        return json.loads(text)
    except ValueError as exc:
        raise PackError("%s: is not valid JSON (%s)" % (relative, exc))


# --------------------------------------------------------------------------- edges


def _load_edges(graph, nodes):
    declared = graph.get("edges") or []
    if not isinstance(declared, list):
        raise GraphError("graph.yaml: 'edges' must be a list")

    edges = []
    for index, spec in enumerate(declared, start=1):
        if not isinstance(spec, dict):
            raise GraphError("graph.yaml: edge %d must be a mapping" % index)
        unknown = set(spec) - _EDGE_KEYS
        if unknown:
            raise GraphError(
                "graph.yaml: edge %d has unknown key(s): %s"
                % (index, ", ".join(sorted(unknown)))
            )
        source, target = spec.get("from"), spec.get("to")
        for role, value in (("from", source), ("to", target)):
            if not isinstance(value, str) or not value:
                raise GraphError("graph.yaml: edge %d is missing %r" % (index, role))
        for role, value in (("from", source), ("to", target)):
            if value not in nodes:
                raise GraphError(
                    "graph.yaml: edge %s -> %s points at undefined node %r"
                    % (source, target, value)
                )
        when = spec.get("when")
        if when is not None and not isinstance(when, dict):
            raise GraphError(
                "graph.yaml: edge %s -> %s: 'when' must be a mapping of "
                "state key -> expected value" % (source, target)
            )
        if nodes[source].type == "end":
            raise GraphError(
                "graph.yaml: end node %r cannot have an outgoing edge" % source
            )
        edges.append(Edge(source=source, target=target, when=when))
    return edges


def _check_reachable_targets(nodes, edges):
    for node in nodes.values():
        if node.on_fail is not None and node.on_fail not in nodes:
            raise GraphError(
                "graph.yaml: node %r has on_fail %r, which is not a defined node"
                % (node.name, node.on_fail)
            )
        if node.type == "end":
            continue
        if not any(edge.source == node.name for edge in edges):
            raise GraphError(
                "graph.yaml: node %r has no outgoing edge and is not an end node"
                % node.name
            )


# ------------------------------------------------------------------------ evalset


def _load_evalset(path):
    full = os.path.join(path, "evalset.jsonl")
    if not os.path.isfile(full):
        return []
    cases = []
    with open(full, "r") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise EvalsetError("evalset.jsonl:%d: not valid JSON (%s)" % (number, exc))
            if not isinstance(raw, dict):
                raise EvalsetError("evalset.jsonl:%d: each line must be an object" % number)
            for key in ("input", "expect"):
                if not isinstance(raw.get(key), dict):
                    raise EvalsetError(
                        "evalset.jsonl:%d: missing or non-object %r" % (number, key)
                    )
            cases.append(
                EvalCase(
                    input=raw["input"],
                    expect=raw["expect"],
                    name=raw.get("name") or "case %d" % number,
                )
            )
    return cases
