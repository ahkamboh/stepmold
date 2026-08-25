"""Stage 5 — write the pack to disk, then prove it works.

The four stages before this one produce data structures. This one produces the artifact:
a directory of text files that `jig.pack.load_pack` accepts and `jig eval` scores.

    write_pack(directory, task, plan, prompts, script) -> str
    verify_pack(directory)                             -> jig.eval.Report
    compile_report(report)                             -> str

Two disciplines run through the whole module, and both exist because a compiler that is
merely *usually* right is worse than no compiler:

**Round-trip everything.** jig reads a YAML subset (`jig/yamlish.py`), not YAML, and that
subset resolves `no` to `False` and `007` to `7`. An emitter that guesses wrong produces a
pack that loads cleanly and means something else. So every document is parsed back with
jig's own reader *before* it reaches the disk and compared against the structure it was
built from; then the finished directory is handed to `load_pack` and compared against the
plan. A quoting bug fails the compile, loudly, instead of shipping.

**Never edit the contract.** `evalset.jsonl` is the gold set verbatim. A compiler that
adjusts its own test until it passes has measured nothing — the number it reports would be
a statement about the compiler's willingness to lower the bar, not about the pack. The
cases are validated here (a case naming an ending that does not exist is a real error) but
never rewritten, and `verify_pack` is the gate: a pack that does not score full marks
against its own gold cases is not a pack the compiler may claim to have built.

Standard library only, and nothing here is imported by the runtime.
"""

import json
import os
import re
import shutil
import textwrap

from ..cli import resolve_model
from ..eval import evaluate
from ..pack import load_pack
from ..yamlish import YamlError, parse as parse_yaml
from .spec import BuildError

__all__ = ["write_pack", "verify_pack", "compile_report"]

# Node names become file names (`prompts/<node>.txt`), and they are also the identifiers
# a `when:` path and an `expr:` are written in. Anything outside this is a name that
# would work in one of those places and not the others.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The plain scalars that survive `jig/yamlish.py` unchanged: a leading letter and nothing
# in the body that the parser gives a meaning to. Everything else is double-quoted. The
# whitelist is deliberately narrow — over-quoting costs readability, under-quoting costs
# correctness, and only one of those is caught by a test.
_PLAIN = re.compile(r"^[A-Za-z][A-Za-z0-9_.\- ]*[A-Za-z0-9_.]$|^[A-Za-z]$")

# Words `jig/yamlish.py:_scalar` resolves to something other than a string. Kept here
# rather than imported because they are that parser's private resolution table; if it
# grows a word, the round-trip check below is what catches the divergence.
_RESOLVED = (
    "", "~", "null", "Null", "NULL",
    "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON",
    "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF",
)

_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r", "\0": "\\0"}

# The files this compiler owns. `overwrite=True` clears exactly these and no more, so a
# hand-written README or a note beside the pack survives a recompile.
_MANAGED = ("manifest.yaml", "graph.yaml", "evalset.jsonl", "fakes/script.json")
_MANAGED_DIRS = ("prompts", "grammars")

_SCRIPT_PATH = "fakes/script.json"


# --------------------------------------------------------------------------- writing


def write_pack(directory, task, plan, prompts, script, overwrite=False):
    """Emit a complete pack for `task`/`plan` into `directory`, and return its path.

    `prompts` maps node name -> emit template. A key of the form `"<node>.think"` is
    written as that node's two-stage think template (`prompts/<node>.think.txt`), which
    is the one artifact whose location jig will not let a node override.

    `script` is the offline model: a list of responses or an object keyed by prompt
    substring, written to `fakes/script.json` and named by the manifest, so the finished
    pack scores with no GPU and no network.

    Everything is checked before anything is written. A compile that fails leaves the
    target directory as it found it.
    """
    directory = os.path.normpath(directory)
    _check_target(directory, overwrite)
    _check_plan(task, plan, prompts, script)
    files = _render(task, plan, prompts, script)

    if overwrite:
        _clear(directory)
    for relative, text in files.items():
        full = os.path.join(directory, relative.replace("/", os.sep))
        parent = os.path.dirname(full)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        with open(full, "w") as handle:
            handle.write(text)

    _check_written(directory, task, plan, prompts)
    return directory


def _check_target(directory, overwrite):
    """Refuse to write over a pack somebody may have tuned by hand."""
    if not os.path.exists(directory):
        return
    if not os.path.isdir(directory):
        raise BuildError("%s exists and is not a directory" % directory)
    if os.listdir(directory) and not overwrite:
        raise BuildError(
            "%s is not empty. A pack's prompts are the part an author edits by hand, so "
            "the compiler will not overwrite one unless you pass overwrite=True."
            % directory
        )


def _clear(directory):
    for relative in _MANAGED:
        full = os.path.join(directory, relative.replace("/", os.sep))
        if os.path.isfile(full):
            os.remove(full)
    for name in _MANAGED_DIRS:
        # A prompt left behind for a node this compile removed is dead text that still
        # reads as part of the pack in a diff. Only these two directories are cleared.
        full = os.path.join(directory, name)
        if os.path.isdir(full):
            shutil.rmtree(full)


# -------------------------------------------------------------------------- checking


def _check_plan(task, plan, prompts, script):
    """Everything that can be known before a byte is written."""
    if not task.name or not _IDENTIFIER.match(task.name):
        raise BuildError(
            "pack name %r is not a plain identifier; it names the pack in every report "
            "and checkpoint" % (task.name,)
        )
    if not plan.nodes:
        raise BuildError("the plan has no generate nodes, so the pack would do nothing")
    if not plan.endings:
        raise BuildError(
            "the plan has no endings; a graph with no `end` node can only run out of "
            "steps"
        )
    if not task.cases:
        raise BuildError(
            "no gold cases: the evalset is the pack's contract and `jig eval` refuses "
            "an empty one, so a pack compiled without cases could never be verified"
        )
    if not script:
        raise BuildError(
            "the offline script is empty; FakeModel needs at least one response"
        )

    names = _check_names(plan)
    _check_writes(task, plan)
    _check_prompts(plan, prompts)
    _check_edges(plan, names)
    _check_cases(task, plan)


def _check_names(plan):
    names = []
    for name in [node.name for node in plan.nodes] + list(plan.endings):
        if not _IDENTIFIER.match(name or ""):
            raise BuildError(
                "node name %r is not a plain identifier; node names become file names "
                "(prompts/<node>.txt)" % (name,)
            )
        if name in names:
            raise BuildError("node %r is declared twice in the plan" % name)
        names.append(name)
    if plan.entry not in names:
        raise BuildError(
            "entry %r is not a node in the plan (nodes: %s)"
            % (plan.entry, ", ".join(names))
        )
    return names


def _check_writes(task, plan):
    """Every field written by exactly one node, and by nobody the run inputs own.

    Both halves matter at run time and neither is caught by `load_pack`: a field written
    twice is ambiguous provenance, a field written by nobody never reaches the output,
    and a field sharing a name with a run input is a `StateCollision` on the first case.
    """
    declared = [spec.name for spec in task.fields]
    written = plan.written_fields

    unknown = [name for name in written if name not in declared]
    if unknown:
        raise BuildError(
            "node(s) write field(s) the examples never showed: %s"
            % ", ".join(sorted(set(unknown)))
        )
    missing = [name for name in declared if name not in written]
    if missing:
        raise BuildError(
            "no node writes field(s): %s — they would never appear in the output"
            % ", ".join(missing)
        )
    twice = sorted({name for name in written if written.count(name) > 1})
    if twice:
        raise BuildError(
            "field(s) written by more than one node: %s — a field must have exactly one "
            "author or its provenance is a guess" % ", ".join(twice)
        )
    known = set(declared) | set(task.inputs)
    for node in plan.nodes:
        unwritable = [name for name in node.reads if name not in known]
        if unwritable:
            raise BuildError(
                "node %r reads %s, which is neither a run input nor a field any node "
                "writes" % (node.name, ", ".join(sorted(unwritable)))
            )

    collisions = [name for name in declared if name in task.inputs]
    if collisions:
        raise BuildError(
            "field(s) %s share a name with a run input; a generate node may not "
            "overwrite an input, so give the field a different name in its grammar"
            % ", ".join(collisions)
        )


def _check_prompts(plan, prompts):
    wanted = {node.name for node in plan.nodes}
    for node in plan.nodes:
        text = prompts.get(node.name)
        if not text or not text.strip():
            raise BuildError("no prompt for generate node %r" % node.name)
    for key in prompts:
        base = key[: -len(".think")] if key.endswith(".think") else key
        if base not in wanted:
            raise BuildError(
                "prompt %r does not belong to any generate node in the plan" % key
            )
        if key.endswith(".think") and not plan.node_named(base).two_stage:
            raise BuildError(
                "a think template was given for %r, which the plan does not mark "
                "two_stage; jig would never read it" % base
            )


def _check_edges(plan, names):
    endings = set(plan.endings)
    for index, edge in enumerate(plan.edges, start=1):
        if not isinstance(edge, dict):
            raise BuildError("edge %d in the plan is not a mapping" % index)
        unknown = set(edge) - {"from", "to", "when", "description"}
        if unknown:
            raise BuildError(
                "edge %d has unknown key(s): %s" % (index, ", ".join(sorted(unknown)))
            )
        for role in ("from", "to"):
            if edge.get(role) not in names:
                raise BuildError(
                    "edge %d has %s %r, which is not a node in the plan"
                    % (index, role, edge.get(role))
                )
        if edge["from"] in endings:
            raise BuildError(
                "edge %d leaves ending %r; an end node terminates the run"
                % (index, edge["from"])
            )
        when = edge.get("when")
        if when is not None and not isinstance(when, dict):
            raise BuildError("edge %d: 'when' must be a mapping" % index)

    if plan.edges:
        sources = {edge["from"] for edge in plan.edges}
        stranded = [node.name for node in plan.nodes if node.name not in sources]
        if stranded:
            # load_pack refuses this too, but only once the files are on disk. Saying it
            # here keeps a failed compile from leaving a broken pack behind.
            raise BuildError(
                "node(s) %s have no outgoing edge and are not endings"
                % ", ".join(stranded)
            )


def _check_cases(task, plan):
    """Validate the gold cases. Validate — never rewrite: they are the contract."""
    endings = set(plan.endings)
    for number, case in enumerate(task.cases, start=1):
        if not isinstance(case, dict):
            raise BuildError("gold case %d is not an object" % number)
        for key in ("input", "expect"):
            if not isinstance(case.get(key), dict):
                raise BuildError("gold case %d has a missing or non-object %r" % (number, key))
        end = case.get("end")
        if end is not None and end not in endings:
            raise BuildError(
                "gold case %d expects ending %r, which the plan does not declare "
                "(endings: %s)" % (number, end, ", ".join(plan.endings))
            )


# -------------------------------------------------------------------------- emitting


def _render(task, plan, prompts, script):
    """Build every file as text. Nothing touches the disk until all of it exists."""
    files = {}
    files["manifest.yaml"] = _yaml_document(_manifest(task, plan), "manifest.yaml")
    files["graph.yaml"] = _yaml_document(_graph(task, plan), "graph.yaml", _graph_header(plan))
    for node in plan.nodes:
        files["prompts/%s.txt" % node.name] = _text(prompts[node.name])
        think = prompts.get("%s.think" % node.name)
        if think:
            files["prompts/%s.think.txt" % node.name] = _text(think)
        files["grammars/%s.json" % node.name] = _grammar_json(_grammar(task, node))
    files["evalset.jsonl"] = _evalset(task)
    files[_SCRIPT_PATH] = _json(script, sort_keys=True)
    return files


def _manifest(task, plan):
    manifest = {
        "name": task.name,
        "version": 1,
        "entry": plan.entry,
        "model": "fake:" + _SCRIPT_PATH,
    }
    description = (task.description or "").strip()
    note = textwrap.fill(
        "Compiled by `jig build`. The model above is the scripted stand-in this pack "
        "ships with, so the evalset scores offline with no GPU and no network. Point "
        "--model at a real backend to run real inputs.",
        width=84,
    )
    manifest["description"] = description + "\n\n" + note if description else note
    return manifest


def _graph_header(plan):
    return "# %s -> %s\n#\n%s\n\n" % (
        " -> ".join(node.name for node in plan.nodes),
        " | ".join(plan.endings),
        "# Each node has one narrow job and one grammar, so no node has to plan and no\n"
        "# node needs a large model to be reliable.",
    )


def _graph(task, plan):
    nodes = {}
    for node in plan.nodes:
        spec = {"type": "generate", "max_tokens": _max_tokens(task, node.writes)}
        if node.two_stage:
            # Bare `true`. `two_stage` is the one node key jig does not shape-check —
            # `bool("false")` is True — so a quoted value here would silently double
            # every call this node makes (docs/pack-format.md).
            spec["two_stage"] = True
        nodes[node.name] = spec

    projection = [spec.name for spec in task.fields]
    for ending in plan.endings:
        # Every ending projects every field. `_project` skips what state does not hold,
        # so a branch that never ran a node simply projects fewer keys — while a
        # hand-trimmed projection per branch is a second place for the field list to
        # drift out of step with the grammars.
        nodes[ending] = {"type": "end", "output": list(projection)}

    graph = {"max_steps": len(nodes) + 4, "nodes": nodes}
    graph["edges"] = [_edge(edge) for edge in (plan.edges or _default_edges(plan))]
    return graph


def _edge(edge):
    out = {"from": edge["from"], "to": edge["to"]}
    if edge.get("when"):
        out["when"] = dict(edge["when"])
    if edge.get("description"):
        out["description"] = edge["description"]
    return out


def _default_edges(plan):
    """A plan that declares no edges is a straight line into its first ending.

    Induction may legitimately produce one — a four-step extraction has nothing to branch
    on — and refusing it here would force every caller to spell out the obvious chain.
    Anything with a branch declares its edges.
    """
    names = [node.name for node in plan.nodes]
    targets = names[1:] + [plan.endings[0]]
    return [{"from": source, "to": target} for source, target in zip(names, targets)]


def _max_tokens(task, writes):
    """An emit budget measured off the gold answers this node has to reproduce.

    jig has no tokenizer and must not grow one, so this counts bytes. A token is at least
    one byte in every tokenizer, which makes the byte length of the longest gold answer
    already an upper bound on its token count; the extra half plus a fixed floor covers
    the whitespace a model adds around the same JSON. It is a deliberate over-estimate,
    not a measurement — `max_tokens` is a ceiling, and a ceiling set too low turns a
    correct answer into a rejected one.
    """
    longest = 0
    for case in task.cases:
        expect = case.get("expect") or {}
        answer = {name: expect[name] for name in writes if name in expect}
        longest = max(longest, len(json.dumps(answer)))
    budget = longest + longest // 2 + 32
    return max(32, -(-budget // 32) * 32)


def _grammar(task, node):
    """The node's contract: closed, with every field it writes required.

    `optional` on a field means "null in at least one gold case", and that is expressed as
    a nullable *type*, not as an absent key. A field that may be missing cannot be
    asserted by an evalset case expecting null — the case would fail as "missing from
    output" — so the closed schema keeps the key and widens the value.
    """
    properties = {}
    for name in node.writes:
        spec = task.field_named(name)
        schema = dict(spec.schema)
        if spec.optional:
            declared = schema.get("type")
            declared = [declared] if isinstance(declared, str) else list(declared or [])
            if "null" not in declared:
                declared.append("null")
            schema["type"] = declared
            if "enum" in schema and None not in schema["enum"]:
                schema["enum"] = list(schema["enum"]) + [None]
        properties[name] = schema
    return {
        "type": "object",
        "properties": properties,
        "required": list(node.writes),
        "additionalProperties": False,
    }


def _grammar_json(grammar):
    """A grammar, laid out the way the example packs are: one line per property.

    `json.dumps(indent=2)` explodes a two-name type union over four lines, and a node's
    contract is the artifact a reviewer most needs to take in at a glance.
    """
    properties = ",\n".join(
        "    %s: %s" % (json.dumps(name), json.dumps(schema, sort_keys=False))
        for name, schema in grammar["properties"].items()
    )
    text = (
        '{\n  "type": "object",\n  "properties": {\n%s\n  },\n'
        '  "required": %s,\n  "additionalProperties": false\n}\n'
        % (properties, json.dumps(grammar["required"]))
    )
    try:
        written = json.loads(text)
    except ValueError as exc:
        raise BuildError("emitted grammar is not valid JSON: %s" % exc)
    if written != grammar:
        raise BuildError("emitted grammar does not read back as it was written")
    return text


def _evalset(task):
    """The gold cases, verbatim.

    The compiler must never edit its own contract. A compiler that adjusts the test until
    the pack passes is worthless — the score it reports would say only that it was willing
    to move the target. Cases are copied out exactly as they came in; if the pack cannot
    meet them, `verify_pack` says so and the compile has failed.
    """
    return "".join(json.dumps(case) + "\n" for case in task.cases)


def _text(value):
    return value if value.endswith("\n") else value + "\n"


def _json(value, sort_keys=False):
    return json.dumps(value, indent=2, sort_keys=sort_keys) + "\n"


# ------------------------------------------------------------------------ yaml emit


def _yaml_document(data, filename, header=""):
    """Render `data`, then read it back with jig's own parser before returning it.

    This is the check that matters. jig reads a YAML subset that resolves `no`, `on` and
    `007` to values that are not strings, so a quoting mistake produces a file that loads
    fine and means something else. Comparing the parse against the structure it came from
    turns that class of bug into a failed compile.
    """
    text = header + _dump(data, 0)
    try:
        parsed = parse_yaml(text, filename=filename)
    except YamlError as exc:
        raise BuildError(
            "emitted %s is not readable by jig's own YAML parser: %s" % (filename, exc)
        )
    if parsed != data:
        raise BuildError(
            "emitted %s does not read back as it was written — a value changed meaning "
            "on the way through the YAML subset.\n  wrote: %r\n  read:  %r"
            % (filename, data, parsed)
        )
    return text


def _dump(mapping, indent):
    lines = []
    _dump_mapping(mapping, indent, lines)
    return "\n".join(lines) + "\n"


def _dump_mapping(mapping, indent, lines):
    pad = " " * indent
    for position, (key, value) in enumerate(mapping.items()):
        block = _is_block(key, value)
        if block and position:
            lines.append("")  # one blank line before each nested block, for the diff
        if block:
            lines.append("%s%s:" % (pad, _scalar(key)))
            if isinstance(value, dict):
                _dump_mapping(value, indent + 2, lines)
            else:
                # jig's parser wants a nested sequence indented past its key, so this
                # never emits the zero-indent `edges:` / `- from:` form real YAML allows.
                _dump_sequence(value, indent + 2, lines)
        elif isinstance(value, str) and _block_safe(value):
            lines.append("%s%s: |-" % (pad, _scalar(key)))
            lines.extend(
                ("%s  %s" % (pad, line)) if line else "" for line in value.split("\n")
            )
        else:
            lines.append("%s%s: %s" % (pad, _scalar(key), _inline(value)))


def _dump_sequence(items, indent, lines):
    pad = " " * indent
    for position, item in enumerate(items):
        if isinstance(item, dict) and item:
            if position:
                lines.append("")
            first = len(lines)
            _dump_mapping(item, indent + 2, lines)
            lines[first] = pad + "-" + lines[first][indent + 1:]
        else:
            lines.append("%s- %s" % (pad, _inline(item)))


# Mappings get an indented block, which is what every example pack looks like. `when:`
# is the exception: an edge reads better on two lines than on four, and its mapping is
# one short state path by construction.
_FLOW_KEYS = ("when",)


def _is_block(key, value):
    """A value that gets its own indented block rather than sitting on the key's line."""
    if isinstance(value, dict):
        return bool(value) and not (key in _FLOW_KEYS and _fits_flow(value))
    if isinstance(value, list):
        return bool(value) and not _fits_flow(value)
    return False


def _fits_flow(collection):
    items = collection.values() if isinstance(collection, dict) else collection
    if any(isinstance(item, (dict, list)) for item in items):
        return False
    return len(_inline(collection)) <= 72


def _inline(value):
    if isinstance(value, dict):
        return "{%s}" % ", ".join(
            "%s: %s" % (_scalar(key), _inline(item)) for key, item in value.items()
        )
    if isinstance(value, list):
        return "[%s]" % ", ".join(_inline(item) for item in value)
    return _scalar(value)


def _scalar(value):
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if not isinstance(value, str):
        raise BuildError("cannot write %r into YAML" % (value,))
    if value in _RESOLVED or not _PLAIN.match(value):
        return _quote(value)
    return value


def _quote(value):
    out = ['"']
    for char in value:
        if char in _ESCAPES:
            out.append(_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append("\\u%04x" % ord(char))
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _block_safe(text):
    """Whether `text` survives a `|-` literal block scalar unchanged.

    Worth the check only for readability: a multi-line description written as one
    backslash-escaped line is legal and unreadable. The first content line sets the
    block's indent, so a leading space on it would be eaten; `-` chomping drops trailing
    newlines; and a control character would be re-flowed into an extra line.
    """
    if "\n" not in text:
        return False
    if text != text.rstrip("\n") or text.startswith((" ", "\t", "\n")):
        return False
    return not any(ord(char) < 0x20 and char != "\n" for char in text)


# --------------------------------------------------------------------- verification


def _check_written(directory, task, plan, prompts):
    """Load the finished directory as jig would, and hold it against the plan.

    The in-memory round trip proves each document reads back as written; this proves the
    *pack* is the one that was planned — the node set, the entry, the edges, the prompt
    bytes and the case count. Anything that disagrees is an emitter bug, and finding it
    here costs a compile instead of a production run.
    """
    try:
        pack = load_pack(directory)
    except Exception as exc:
        raise BuildError(
            "the emitted pack does not load: %s: %s" % (type(exc).__name__, exc)
        )

    planned = {node.name for node in plan.nodes} | set(plan.endings)
    if set(pack.nodes) != planned:
        raise BuildError(
            "emitted graph has nodes %s, planned %s"
            % (", ".join(sorted(pack.nodes)), ", ".join(sorted(planned)))
        )
    if pack.entry != plan.entry:
        raise BuildError("emitted entry %r, planned %r" % (pack.entry, plan.entry))
    if len(pack.evalset) != len(task.cases):
        raise BuildError(
            "emitted %d evalset cases from %d gold cases"
            % (len(pack.evalset), len(task.cases))
        )
    for node in plan.nodes:
        loaded = pack.nodes[node.name]
        if loaded.prompt != _text(prompts[node.name]):
            raise BuildError("prompt for %r did not survive the write" % node.name)
        if bool(loaded.two_stage) != bool(node.two_stage):
            raise BuildError("node %r lost its two_stage flag" % node.name)
    for ending in plan.endings:
        if pack.nodes[ending].type != "end":
            raise BuildError("ending %r was not written as an end node" % ending)


def verify_pack(directory):
    """Load the pack at `directory` and score it against its own gold cases.

    This is the gate. `jig eval` is what a buyer runs, so the compiler runs exactly that:
    the same loader, the same scripted model named by the manifest, the same
    `jig.eval.evaluate`. A report that is not full marks means the compile failed, however
    plausible the prompts look.
    """
    pack = load_pack(directory)
    if not pack.evalset:
        raise BuildError(
            "pack %r has no evalset cases, so there is nothing to verify against"
            % pack.name
        )
    return evaluate(pack, resolve_model(None, pack))


def compile_report(report):
    """A short human summary of a verification: the score, and who is to blame.

    On failure it leans on the per-node attribution `jig.eval` already produces, because
    "10/12, and both failures are the priority node" is the sentence that says what to
    recompile — which is the signal a whole-pack pass/fail throws away.
    """
    lines = ["%s: %d/%d cases passed" % (report.pack, report.passed, report.total)]
    if report.passed_all:
        return lines[0]
    if report.by_node:
        lines.append(
            "  blamed on: %s"
            % ", ".join("%s=%d" % item for item in sorted(report.by_node.items()))
        )
    shown = [case for case in report.cases if not case.passed]
    for case in shown[:3]:
        lines.append("  FAIL %s [%s]: %s" % (case.name, case.node or "<unknown>", _why(case)))
    if len(shown) > 3:
        lines.append("  ... and %d more" % (len(shown) - 3))
    return "\n".join(lines)


def _why(case):
    if case.error:
        return case.error
    if case.mismatches:
        first = case.mismatches[0]
        return "%s: expected %r, got %s" % (
            first.field,
            first.expected,
            first.note or repr(first.actual),
        )
    return "failed"
