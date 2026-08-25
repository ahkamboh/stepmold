"""Loading and validating a JigPack.

A pack is a directory, not a database — it is meant to be read in a diff, checked into
a client's repo, and shipped as text (docs/ARCHITECTURE.md §7.2):

    <pack>/
      manifest.yaml        name, version, entry node, model hint
      graph.yaml           nodes + edges (the compiled "plan")
      prompts/<node>.txt   one prompt template per generate node
      prompts/<node>.think.txt   optional think-stage template (T5)
      grammars/<node>.json one JSON Schema per generate node
      evalset.jsonl        the contract: {"input": {...}, "expect": {...}} per line

A `tool` node has no files of its own: it names a function the *host* registered
(`jig/tools.py`), so there is nothing in the pack to read for it. Pass the registry —
`load_pack(path, tools=registry)` — and the name it calls and the state that tool reads
are checked here too; leave it out and the pack still loads, unchecked on that one point.

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
    "ToolWiringError",
    "UnsafePath",
    "check_tools",
    "load_pack",
]

NODE_TYPES = ("generate", "assert", "tool", "end")

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
    "on_fail", "on_unsure", "expr", "assert", "prompt", "grammar", "tool",
    "description",
}

# Keys that belong to a generate or an assert node and mean nothing on a tool node. Each
# is refused by name, with the reason, rather than ignored: a key that reads as if it
# does something and does nothing is how a pack ends up meaning something other than it
# says — and on a tool node the thing it silently does not do guards a side effect.
_TOOL_FORBIDDEN_KEYS = {
    "prompt": "a tool node calls a function, not a model, so there is no prompt "
              "to render",
    "grammar": "a tool node's contract is the tool's own `writes`, declared by the host "
               "in its registry, not a grammar file in the pack",
    "two_stage": "a tool node never generates, so there is no think stage to run",
    "retries": "a re-run tool is a side effect done twice; route the failure with "
               "`on_fail` instead of re-attempting it",
    "max_tokens": "a tool node never generates",
    "think_max_tokens": "a tool node never generates",
    "assert": "`assert:` gates a *generation* before it is committed; a tool node has no "
              "retry ladder for a rejection to spend",
    "expr": "`expr` is the assert node's branch condition",
}
_EDGE_KEYS = {"from", "to", "when", "description"}


class PackError(Exception):
    """Anything wrong with a pack on disk."""


class UnsafePath(PackError):
    """A pack referenced a file outside its own directory.

    A pack is untrusted input the moment it leaves the machine that compiled it
    (docs/ARCHITECTURE.md §6 plans a registry, §7.2 describes copying packs between hosts), so
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


class ToolWiringError(GraphError):
    """A tool node is wired to state that nothing in this graph produces.

    The alternative is a run that dies at step four because the tool wanted a field
    nobody wrote — by which point the run has already done part of a job it cannot
    finish, and part of a job with side effects in it is the worst place to stop.
    """


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
    on_unsure: Optional[str] = None
    expr: Optional[str] = None
    assert_expr: Optional[str] = None
    # The registered name a `type: tool` node calls. A pack names an action; it never
    # contains one (see jig/tools.py) — so this is a key into the host's registry and
    # nothing else: no import, no dotted path, no default.
    tool: Optional[str] = None


@dataclass(frozen=True)
class Edge:
    """A transition, taken when `when` matches the current state (or is unset)."""

    source: str
    target: str
    when: Optional[dict] = None


@dataclass(frozen=True)
class EvalCase:
    """One line of `evalset.jsonl` — the hand-maintained contract.

    `expect` compares fields, which is not the whole contract a graph makes. Two fields
    exist because authoring real packs proved that:

    * `end` — the ending the run must reach. Field comparison cannot see routing, so a
      pack whose branches project the same shape scored full marks with its policy
      inverted. Naming the ending makes the branch part of the contract.
    * `rescued` — this case is *supposed* to burn a node's ladder and take its `on_fail`
      edge. Without it a declared rescue path could never be a passing expectation, so
      the one path an author most wants to prove was the one the evalset could not score.
      It is checked both ways: a case that claims a rescue and does not get one fails
      too, so it cannot be used to silence a real failure.
    """

    input: dict
    expect: dict
    name: Optional[str] = None
    end: Optional[str] = None
    rescued: bool = False


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


def load_pack(path, tools=None):
    """Read the pack at `path`, validate it, and return a `Pack`.

    `tools` is the host's `jig.tools.ToolRegistry`, and passing it turns on the one
    check this loader cannot do alone: that every `type: tool` node names something the
    host actually registered, and that what each tool declares it `reads` is a field
    this graph will have by the time the node runs. See `check_tools`.

    It is optional on purpose. `jig validate` on a machine where the tools live in
    somebody else's process must still be able to say the pack is well formed — a check
    that cannot run is not the same as a check that failed.
    """
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

    _declared_input_names(manifest)

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

    evalset = _load_evalset(path)
    _check_case_endings(evalset, nodes)

    pack = Pack(
        path=path,
        name=name,
        version=version,
        entry=entry,
        model=model,
        nodes=nodes,
        edges=edges,
        evalset=evalset,
        max_steps=max_steps,
        manifest=manifest,
    )
    if tools is not None:
        check_tools(pack, tools)
    return pack


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
    tool_name = _tool_key(node_name, node_type, spec)

    prompt = think_prompt = grammar = None
    # Only a generate node reads artifacts off disk. A tool node deliberately does not
    # fall into this branch: it has no prompt and no grammar, so requiring
    # `prompts/<node>.txt` of it would refuse a pack that is perfectly well formed.
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
        on_unsure=spec.get("on_unsure"),
        expr=spec.get("expr"),
        assert_expr=spec.get("assert"),
        tool=tool_name,
    )


def _tool_key(node_name, node_type, spec):
    """Validate the `tool:` key both ways round, and return the name a tool node calls.

    Both halves are load-time refusals rather than run-time surprises. A tool node
    without a name has nothing to call, and a `tool:` on a generate node is a key the
    walker never reads — it would sit in the pack looking like an action that is never
    taken, which is the most expensive kind of silence a pack can hold.
    """
    if node_type != "tool":
        if "tool" in spec:
            raise GraphError(
                "graph.yaml: node %r is type %r but carries 'tool: %s'. Only a tool node "
                "names a tool — set 'type: tool', or drop the key."
                % (node_name, node_type, _clip(spec.get("tool")))
            )
        return None

    forbidden = sorted(set(spec) & set(_TOOL_FORBIDDEN_KEYS))
    if forbidden:
        raise GraphError(
            "graph.yaml: tool node %r carries %s. Those keys belong to a generate or "
            "an assert node and nothing would read them here — remove them, or make "
            "this a node type that uses them. %s"
            % (node_name, ", ".join("%r" % key for key in forbidden),
               " ".join("%r: %s." % (key, _TOOL_FORBIDDEN_KEYS[key])
                        for key in forbidden))
        )

    name = spec.get("tool")
    if not isinstance(name, str) or not name.strip():
        raise GraphError(
            "graph.yaml: tool node %r needs a 'tool:' naming the registered tool it "
            "calls (got %r). A pack names an action; the host registers it."
            % (node_name, name)
        )

    output = spec.get("output")
    if output is not None and (not isinstance(output, str) or not output):
        # Same shape as a generate node's `output:`, for the same reason: it is the one
        # state key the result is committed under. A list here would read like an end
        # node's projection and commit under no key at all.
        raise GraphError(
            "graph.yaml: tool node %r: 'output' must be a single state key to commit "
            "the tool's result under (a string), got %r" % (node_name, output)
        )
    return name.strip()


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
        # `on_fail` and `on_unsure` are both edges the walk can take without an entry in
        # `edges:`, so both are checked here or nowhere.
        for key, target in (("on_fail", node.on_fail), ("on_unsure", node.on_unsure)):
            if target is not None and target not in nodes:
                raise GraphError(
                    "graph.yaml: node %r has %s %r, which is not a defined node"
                    % (node.name, key, target)
                )
        if node.type == "end":
            continue
        if not any(edge.source == node.name for edge in edges):
            raise GraphError(
                "graph.yaml: node %r has no outgoing edge and is not an end node"
                % node.name
            )


# --------------------------------------------------------------------------- tools


def check_tools(pack, tools):
    """Check every `type: tool` node in `pack` against the host's registry.

    Two questions, both of which have a right answer before the run starts:

    * **Is the tool there?** A name the host never registered is refused here rather
      than at the step that would have called it. `jig/tools.py` builds the allowlist
      that way on purpose — a pack can only call what the host already handed it — and
      an allowlist that is checked halfway through a job is not one.
    * **Can the tool's `reads` be satisfied?** A tool is invoked with exactly the state
      it declared it needs (`Tool.invoke`), so a field that no earlier node writes and
      that no caller supplies is a wiring mistake with a name. Finding it at load turns
      "the run died at step 4 because the tool wanted `order_id`" into "this pack is
      wired wrong, here is the field".

    What counts as "the graph will have it by then", at any earlier node:

    | Source | The state keys it contributes |
    | --- | --- |
    | a node with `output:` | the one key it commits under |
    | a generate node without one | its grammar's property names (merge mode) |
    | a tool node without one | the registered tool's `writes` |
    | the run's own inputs | keys an evalset case supplies, or the manifest's `inputs:` |

    "Earlier" is any node the walk can reach this one from — down any branch, round any
    loop, along any rescue path. That is deliberately generous: this check exists to
    catch the field *nobody* writes, and reporting a field that merely *might* not be
    written on some branch would make it the kind of check people switch off. The one
    node it will not credit is one whose `on_fail` leads here, because a node that
    failed committed nothing at all (`_links`).

    For the same reason it stays quiet where it cannot be sure. A pack that declares no
    inputs anywhere (no evalset, no manifest `inputs:`) says nothing about what the
    caller passes, and an earlier node whose grammar declares no properties can write
    anything at all; in either case an unknown field is unproven, not wrong.

    Raises `jig.tools.ToolNotRegistered` for a name the registry does not hold, and
    `ToolWiringError` for a field nothing produces.
    """
    tool_nodes = [node for node in pack.nodes.values() if node.type == "tool"]
    if not tool_nodes:
        return
    if not (hasattr(tools, "get") and hasattr(tools, "has")):
        # A plain dict would answer `get(name, node_name)` with the node name and sail
        # straight past the registration check, so refuse the shape rather than trust it.
        raise TypeError(
            "tools must be a jig.tools.ToolRegistry (got %s)" % type(tools).__name__
        )

    # Every name first: an unregistered tool is the more basic mistake, and reporting it
    # before a wiring complaint keeps the two from being confused for each other.
    registered = {node.name: tools.get(node.tool, node.name) for node in tool_nodes}

    declared_inputs = _declared_inputs(pack)
    committed, lost = _links(pack)
    live = _walk(pack.entry, _merged(committed, lost))
    preceding, rescued_from = _reverse(committed), _reverse(lost)

    for node in tool_nodes:
        if node.name not in live:
            # Unreachable from the entry node, so it never runs and has no "earlier"
            # to speak of. That is a graph problem, not a tool problem.
            continue
        tool = registered[node.name]
        if not tool.reads:
            continue
        earlier = _earlier(node.name, preceding, rescued_from) & live
        available, complete = _available_before(pack, earlier, registered)
        if not complete or declared_inputs is None:
            continue
        missing = [field for field in tool.reads
                   if field not in available and field not in declared_inputs]
        if missing:
            raise ToolWiringError(
                "graph.yaml: tool node %r calls tool %r, which reads %s — and nothing "
                "writes %s before this node runs. Earlier nodes write: %s. The run "
                "inputs this pack declares are: %s. Give an earlier node an 'output:' "
                "that names the field, add it to the pack's inputs (an evalset case, or "
                "manifest 'inputs:'), or call a tool that reads what this graph has."
                % (node.name, tool.name, ", ".join(repr(f) for f in sorted(missing)),
                   "it" if len(missing) == 1 else "them",
                   ", ".join(sorted(available)) or "nothing",
                   ", ".join(sorted(declared_inputs)) or "none")
            )


def _declared_inputs(pack):
    """Every state key the caller is known to supply, or None if the pack never says.

    Two sources, both of them the pack's own text: the manifest's optional `inputs:`
    list, and the keys of each evalset case's `input` object. None — "this pack declares
    nothing" — is not the same as an empty set, and only the empty set is evidence.
    """
    names = _declared_input_names(pack.manifest)
    if names is None and not pack.evalset:
        return None
    declared = set(names or ())
    for case in pack.evalset:
        declared.update(case.input)
    return declared


def _declared_input_names(manifest):
    """The manifest's optional `inputs:` list, validated.

    Unknown manifest keys are kept rather than refused (docs/pack-format.md), but this
    one is read, so a value of the wrong shape would quietly declare nothing — and the
    whole point of declaring inputs is to be believed.
    """
    names = manifest.get("inputs")
    if names is None:
        return None
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ManifestError(
            "manifest.yaml: 'inputs', when present, must be a list of the state key "
            "names a caller supplies to a run, got %s" % _clip(names)
        )
    return list(names)


def _links(pack):
    """The graph's transitions, split by whether the node they leave got to commit.

    Two maps, both node name -> the nodes it can move to:

    * **committed** — ordinary `edges:`, and `on_unsure`. A node reached this way ran to
      completion, so whatever it writes is in state.
    * **lost** — `on_fail`. A node whose ladder ran out committed *nothing*: the
      rejected output is dropped and never touches state (`graph.run`). Its fields are
      not available to what it diverts to, though its own predecessors' fields are.

    `on_unsure` sits with the first group deliberately. Being unsure about a value is not
    the same as not having produced one, and assuming a loss here would invent wiring
    errors in packs that route a low-confidence result onward for review.
    """
    committed = dict((name, set()) for name in pack.nodes)
    lost = dict((name, set()) for name in pack.nodes)
    for edge in pack.edges:
        committed[edge.source].add(edge.target)
    for node in pack.nodes.values():
        if node.on_unsure in pack.nodes:
            committed[node.name].add(node.on_unsure)
        if node.on_fail in pack.nodes:
            lost[node.name].add(node.on_fail)
    return committed, lost


def _merged(committed, lost):
    return dict((name, committed[name] | lost[name]) for name in committed)


def _reverse(following):
    reversed_map = dict((name, set()) for name in following)
    for name, targets in following.items():
        for target in targets:
            reversed_map[target].add(name)
    return reversed_map


def _earlier(node_name, preceding, rescued_from):
    """Every node whose writes are in state by the time `node_name` runs.

    Walked backwards from the node's predecessors, not from the node itself — so a node
    inside a loop counts as its own ancestor (its earlier pass really did write those
    fields) while a node outside one does not (its own writes come too late).

    A node reached backwards through its `on_fail` is walked *past* rather than counted:
    the run took that edge precisely because the node produced nothing.
    """
    counted, seen = set(), set()
    queue = [(node_name, False)]
    while queue:
        name, writes_landed = queue.pop()
        if (name, writes_landed) in seen:
            continue
        seen.add((name, writes_landed))
        if writes_landed:
            counted.add(name)
        queue.extend((previous, True) for previous in preceding.get(name, ()))
        queue.extend((previous, False) for previous in rescued_from.get(name, ()))
    return counted


def _walk(start, links):
    """Everything reachable from `start` through `links`, `start` itself included."""
    seen, queue = set(), [start]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        queue.extend(links.get(name, ()))
    return seen


def _available_before(pack, earlier, registered):
    """(state keys those nodes write, whether that set is the whole story)."""
    available = set()
    complete = True
    for name in earlier:
        keys, known = _node_writes(pack.nodes[name], registered)
        available.update(keys)
        complete = complete and known
    return available, complete


def _node_writes(node, registered):
    """What one node commits into state, and whether that can be known from the pack.

    A generate node with no `output:` merges its generated object into state, so its
    state keys are its grammar's property names — unless the grammar declares no
    properties, in which case the node may write anything and this returns "unknown"
    rather than "nothing". Same for a tool node whose tool declares no `writes`.
    """
    if node.type not in ("generate", "tool"):
        # assert and end nodes cannot write state at all (docs/graph.md, "Node types").
        return set(), True
    if isinstance(node.output, str) and node.output:
        return {node.output}, True
    if node.output:
        # An `output:` of a shape `commit` cannot use as a state key. The CLI refuses
        # that pack outright (`cli._check_output_shapes`); here it simply means this
        # node's writes are not something to draw a conclusion from.
        return set(), False
    if node.type == "generate":
        properties = (node.grammar or {}).get("properties")
        if isinstance(properties, dict) and properties:
            return set(properties), True
        return set(), False
    tool = registered.get(node.name)
    if tool is not None and tool.writes:
        return set(tool.writes), True
    return set(), False


# ------------------------------------------------------------------------ evalset


def _case_end(raw, number):
    """The ending a case must reach, or None. Validated against the graph by the caller."""
    value = raw.get("end")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EvalsetError(
            "evalset.jsonl line %d: 'end' must be the name of an end node" % number
        )
    return value


def _case_flag(raw, key, number):
    value = raw.get(key, False)
    if not isinstance(value, bool):
        raise EvalsetError(
            "evalset.jsonl line %d: %r must be true or false" % (number, key)
        )
    return value


def _check_case_endings(cases, nodes):
    """Every `end:` a case names must be a real ending in this graph.

    A typo would otherwise never match and quietly fail every run of that case, which is
    the same silent-wrongness the field exists to remove.
    """
    for case in cases:
        if case.end is None:
            continue
        node = nodes.get(case.end)
        if node is None:
            raise EvalsetError(
                "evalset.jsonl: case %r expects ending %r, which is not a node in "
                "graph.yaml" % (case.name, case.end)
            )
        if node.type != "end":
            raise EvalsetError(
                "evalset.jsonl: case %r expects ending %r, but that node is type %r, "
                "not 'end'" % (case.name, case.end, node.type)
            )


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
                    end=_case_end(raw, number),
                    rescued=_case_flag(raw, "rescued", number),
                )
            )
    return cases
