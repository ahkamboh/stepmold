"""Stage 4 — the scripted offline model, derived from the gold answers.

Every pack ships a stand-in model so `jig eval` scores in CI with no GPU and no network
(`manifest.yaml: model: fake:fakes/script.json`). Writing that file by hand is the most
expensive part of authoring a pack and the least interesting: a node's `writes` says
which fields it produces and a gold case says what those fields must be, so the answer a
node owes a case is

    {f: case["expect"][f] for f in node.writes}

and the only real work is deciding *which* cases reach the node at all. That is the bug
this stage exists to prevent. A branching graph means the number of answers a node needs
is the number of cases that REACH it, and nothing else in jig cross-checks the two: a
script that is one answer long at node three fails at case eleven with `ModelExhausted`,
naming neither the case nor the reason. So the walk here follows the plan's edges per
case rather than assuming every case visits every node — `route` is that walk, and it is
public because the linter and the assembler both want the same answer.

**Keying.** `FakeModel`'s keyed mode matches a key that is a substring of the prompt,
longest key first, so the key has to be something a *rendered* prompt is guaranteed to
contain. This stage never sees the prompts (they come from the scribe, in parallel), so
it cannot read a key off them; it publishes one instead. `node_key` is the contract:
every node's prompt must contain the phrase "the <node> step", which is the header line
the example packs already use ("You are the classify step of ..."). `check_script`
verifies it against the real prompts once both halves exist, which is the honest place
for a cross-stage assumption to be caught.

**Values are queues, never bare strings.** A string answers every matching prompt, so a
node whose script is a string can never run out — and a node that can never run out is a
node whose branching is never checked. A list of exactly the answers the reaching cases
expect turns a routing mistake into a loud failure at eval time, which is the whole
point. The queue is in evalset order because `jig eval` builds one `FakeModel` for the
whole run (`cli.resolve_model`) and walks the cases in file order.

**The think stage is the exception.** Without a `prompts/<node>.think.txt` the think
prompt is the emit prompt plus a fixed suffix (`codegen.DEFAULT_THINK_SUFFIX`), so any
key that matches the emit call matches the think call too, and a queue would have its
first answer eaten by a call whose output is thrown away. The fix is one entry keyed on
the suffix itself: it is longer than any node key, so it wins the longest-match, and its
value is a plain string so it answers every think call in the run. That also survives the
retry ladder, which re-thinks whenever it rejects a two-stage answer
(`verify.run_node`) — a queue there would run out on the first rejection.
"""

import json
from dataclasses import dataclass
from typing import List, Optional

from ..codegen import DEFAULT_THINK_SUFFIX
from .spec import BuildError

__all__ = [
    "MARKER",
    "Route",
    "THINK_ANSWER",
    "THINK_KEY",
    "check_script",
    "node_key",
    "route",
    "script_for",
]

# The phrase a node's prompt must contain for its scripted answers to be found. Keep it
# longer than a bare node name: "priority" appears in a prompt that merely *reads*
# `{priority}`, and would then swallow another node's answers.
MARKER = "the %s step"

# Distinct from every node key by construction (see `_check_think_key`), so the think
# call resolves here and the emit call resolves to the node.
THINK_KEY = DEFAULT_THINK_SUFFIX.strip()
THINK_ANSWER = (
    "Weighing the case against the rules above before committing to an answer."
)

# A case that has not reached an ending by here is going round a loop the gold answers
# cannot break, since the answers this stage feeds a node are the same every time round.
_MAX_STEPS = 100

_ZERO = {"string": "", "integer": 0, "number": 0.0, "boolean": False}


def node_key(name):
    """The script key for `name` — and the phrase its prompt is required to contain."""
    return MARKER % name


@dataclass(frozen=True)
class Route:
    """Which nodes one gold case walks through, and where it ends.

    `nodes` may name a node twice: a graph with a loop in it visits one node per turn of
    the loop, and each visit costs a generation, so each visit costs an answer.
    """

    nodes: List[str]
    ending: Optional[str] = None


def script_for(task, plan):
    """Build the keyed `FakeModel` script that replays `task`'s gold answers.

    The result is the object `fakes/script.json` holds: `{key: [answer, ...]}`, one queue
    per node in evalset order, plus the single think entry when the plan has a two-stage
    node in it.
    """
    if not task.cases:
        raise BuildError(
            "cannot script an offline model from an empty examples file: the scripted "
            "answers are the gold answers, so a pack with no cases has nothing to say"
        )
    _check_expectations(task, plan)
    _check_think_key(plan)

    queues = {}
    for case in task.cases:
        for name in route(task, plan, case).nodes:
            node = plan.node_named(name)
            queues.setdefault(name, []).append(_answer_text(task, node, case))

    # Node order, not queue-discovery order, so two runs of the compiler over the same
    # plan produce byte-identical files and a rebuild shows up as an empty diff.
    script = {}
    for node in plan.nodes:
        answers = queues.get(node.name)
        if answers:
            script[node_key(node.name)] = answers
        # A node no case reaches gets no entry rather than an empty queue: an empty queue
        # is a promise of answers that would fail as `ModelExhausted` mid-run, and the
        # honest report of an unexercised node is `check_script`'s, not a silent stub's.
    if any(node.two_stage for node in plan.nodes):
        script[THINK_KEY] = THINK_ANSWER
    if not script:
        # `FakeModel` refuses an empty script, and it is right to: a pack whose offline
        # model answers nothing cannot be evaluated at all.
        raise BuildError(
            "no node in the plan is reached by any gold case, so there is nothing to "
            "script; check the plan's entry (%r) and its edges" % plan.entry
        )
    return script


def route(task, plan, case):
    """Walk `plan` the way the runtime would, answering each node from `case`.

    The walker picks the first outgoing edge whose `when` matches state
    (`graph._next`), and state here is the case's inputs plus the gold answers committed
    so far — which is exactly what state holds at that moment in a run the case passes.
    So the nodes this returns are the nodes that case will actually generate at.
    """
    state = dict(case.get("input") or {})
    endings = set(plan.endings)
    visited = []
    name = plan.entry
    for _ in range(_MAX_STEPS):
        if name in endings:
            return Route(nodes=visited, ending=name)
        node = _node_or_error(plan, name, case)
        visited.append(name)
        state.update(_answer(task, node, case))
        name = _successor(plan, node, state, case)
    raise BuildError(
        "case %r still had not reached an ending after %d nodes; the plan's edges send "
        "it round a loop that the gold answers never break"
        % (_case_name(case), _MAX_STEPS)
    )


def check_script(script, task, plan, prompts=None):
    """Lint a script against the plan and the gold cases; return the problems found.

    Human-readable strings, one per problem, empty when there is nothing to say. This is
    the check nobody had: every failure it names shows up otherwise as `ModelExhausted`
    at some unrelated case, or — worse — as a green evalset that is scoring the wrong
    node's answers.

    `prompts` is the scribe's `{node: template}`, when the caller has it. It is what
    turns "no prompt can match this entry" from an assumption into a check; the templates
    are unrendered, but a key lives in the fixed half of a template, not in `{state}`.

    Not everything reported is a defect. A field no gold case pins is scripted with a
    placeholder and is reported here because *nothing tests it* — true of two fields in
    the shipped support_triage pack, and worth saying out loud rather than treating a
    clean list as proof. Callers that want a hard gate should read the list, not just
    its length.
    """
    problems = []
    if not isinstance(script, dict):
        return [
            "the script is a %s, not a keyed object; only a keyed script can answer a "
            "branching graph, where a node is not called once per case"
            % type(script).__name__
        ]

    expected, walk_failed = _expected_queues(task, plan, problems)
    two_stage = [node.name for node in plan.nodes if node.two_stage]

    keys = {node_key(node.name): node.name for node in plan.nodes}
    for key in script:
        if key in keys or key == THINK_KEY:
            continue
        problems.append(
            "script key %r matches no node in the plan: no prompt this pack ships can "
            "contain it, so its answers are never used" % key
        )

    problems.extend(_check_think_entry(script, two_stage, keys))
    if prompts:
        problems.extend(_check_prompts(script, plan, prompts, keys))
    problems.extend(_check_written_twice(plan))
    problems.extend(_check_input_collisions(task, plan))
    if not walk_failed:
        problems.extend(_check_placeholders(task, plan))
        problems.extend(_check_endings(task, plan))
        for node in plan.nodes:
            problems.extend(
                _check_node(script, task, node, expected.get(node.name, []))
            )
    return problems


# --------------------------------------------------------------- the answers


def _answer(task, node, case):
    """What `node` must say for `case`, as an object."""
    expect = case.get("expect") or {}
    answer = {}
    for name in node.writes:
        answer[name] = expect[name] if name in expect else _placeholder(task, name)
    return answer


def _answer_text(task, node, case):
    # `ensure_ascii=False` so a pack whose gold answers hold a customer's name or a €
    # sign stays readable in the diff; `json.dump` re-escapes on the way to disk if the
    # assembler asks it to.
    return json.dumps(_answer(task, node, case), ensure_ascii=False)


def _placeholder(task, name):
    """A grammar-legal value for a field this case does not pin.

    A node's grammar requires every field the node writes, so an answer that simply
    omits one is rejected by `verify` and the case fails on a field nobody was testing.
    The value is chosen to be the least assertive thing the schema allows: a member of
    the enum first (an enum'd field may not accept null), then null for a field the
    examples showed as optional, then a value that was actually observed.
    """
    spec = task.field_named(name)   # raises BuildError naming the field
    if spec.enum:
        return spec.enum[0]
    if spec.optional:
        return None
    if spec.examples:
        return spec.examples[0]
    return _ZERO.get(spec.type, "")


# --------------------------------------------------------------- traversal


def _node_or_error(plan, name, case):
    try:
        return plan.node_named(name)
    except BuildError:
        raise BuildError(
            "case %r reaches %r, which is neither a node nor an ending in the plan"
            % (_case_name(case), name)
        )


def _successor(plan, node, state, case):
    """The node an edge from `node` leads to, given `state`."""
    edges = [edge for edge in plan.edges if _edge_end(edge, "from") == node.name]
    if not edges:
        if plan.edges:
            raise BuildError(
                "the plan has edges but none leaves node %r, so case %r has nowhere to "
                "go after it" % (node.name, _case_name(case))
            )
        return _linear_successor(plan, node)
    for edge in edges:
        if _matches(edge.get("when"), state):
            return _edge_end(edge, "to")
    raise BuildError(
        "no edge out of %r matches case %r; the case would dead-end there at run time"
        % (node.name, _case_name(case))
    )


def _linear_successor(plan, node):
    """Where a node leads in a plan that declares no edges at all: straight on.

    An edgeless plan is a straight line by the only reading available — node order, then
    the first ending. Saying so here means a linear task can be scripted from a plan that
    has not bothered to spell its chain out.
    """
    names = [item.name for item in plan.nodes]
    following = names.index(node.name) + 1
    if following < len(names):
        return names[following]
    if not plan.endings:
        raise BuildError(
            "the plan declares no edges and no endings, so there is no way to know what "
            "follows node %r" % node.name
        )
    return plan.endings[0]


def _edge_end(edge, side):
    if side not in edge:
        raise BuildError(
            "edge %r has no %r; an edge in a plan is the same shape as an edge in "
            "graph.yaml: {'from': ..., 'to': ..., 'when': ...}" % (edge, side)
        )
    return edge[side]


def _matches(when, state):
    # The same rule as `graph._matches`: every clause must hold, and a dotted key reads
    # into a nested value.
    if not when:
        return True
    for path, expected in when.items():
        if _lookup(path, state) != expected:
            return False
    return True


_MISSING = object()


def _lookup(path, state):
    current = state
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _case_name(case):
    return case.get("name") or "<unnamed>"


# --------------------------------------------------------------- checks


def _check_expectations(task, plan):
    """Every field a case expects must be written by some node, or be a run input."""
    written = set(plan.written_fields)
    inputs = set(task.inputs)
    for case in task.cases:
        for name in case.get("expect") or {}:
            if name in written or name in inputs or name in (case.get("input") or {}):
                continue
            raise BuildError(
                "case %r expects field %r, which no node in the plan writes and no "
                "input supplies — nothing in the compiled pack can ever produce it"
                % (_case_name(case), name)
            )


def _check_think_key(plan):
    """The think entry only wins the longest-match if it is longer than every node key.

    Both of these are theoretical against sane node names, and both would show up as a
    node quietly returning its neighbour's answer rather than as an error, which is the
    kind of bug that costs an afternoon.
    """
    if not any(node.two_stage for node in plan.nodes):
        return
    for node in plan.nodes:
        key = node_key(node.name)
        if len(key) >= len(THINK_KEY):
            raise BuildError(
                "node %r's script key is as long as the think key, so its think call "
                "would eat an answer meant for its emit call; shorten the node name"
                % node.name
            )
        if key in THINK_ANSWER:
            raise BuildError(
                "node %r's script key appears in the scripted think notes, which are "
                "quoted back into the emit prompt; rename the node" % node.name
            )


def _expected_queues(task, plan, problems):
    """The answers each node is owed. A walk that fails is reported, not raised."""
    queues = {}
    try:
        for case in task.cases:
            for name in route(task, plan, case).nodes:
                queues.setdefault(name, []).append(
                    (_case_name(case), _answer(task, plan.node_named(name), case))
                )
    except BuildError as exc:
        problems.append(
            "the plan cannot be walked, so no queue length can be checked: %s" % exc
        )
        return queues, True
    return queues, False


def _check_node(script, task, node, owed):
    problems = []
    key = node_key(node.name)
    entry = script.get(key, _MISSING)

    if entry is _MISSING:
        if owed:
            problems.append(
                "node %r has no entry in the script, but %d gold case(s) reach it — "
                "every one of them dies with ModelExhausted" % (node.name, len(owed))
            )
        else:
            problems.append(
                "node %r is reached by no gold case, so nothing scripts it and nothing "
                "tests it" % node.name
            )
        return problems

    if not owed:
        problems.append(
            "node %r is reached by no gold case, but the script holds an answer for it"
            % node.name
        )
        return problems

    if isinstance(entry, str):
        # Legal, and occasionally right — but it can never run out, so it hides exactly
        # the miscount this stage exists to catch.
        distinct = {json.dumps(answer, sort_keys=True) for _, answer in owed}
        problems.append(
            "node %r is scripted with a single string, which answers every call: the "
            "%d case(s) that reach it are not being checked against it%s"
            % (node.name, len(owed),
               "" if len(distinct) == 1
               else ", and they do not all expect the same answer")
        )
        answers = [entry]
    elif isinstance(entry, list):
        answers = entry
        if len(answers) != len(owed):
            problems.append(
                "node %r is reached by %d gold case(s) but its script holds %d "
                "answer(s)%s" % (node.name, len(owed), len(answers),
                                 _first_unreached(owed, len(answers)))
            )
    else:
        return problems + [
            "node %r's script entry is a %s; FakeModel takes a string or a list of "
            "strings" % (node.name, type(entry).__name__)
        ]

    for index, text in enumerate(answers):
        problems.extend(_check_answer(task, node, index, text, owed))
    return problems


def _check_answer(task, node, index, text, owed):
    problems = []
    where = "node %r answer #%d" % (node.name, index + 1)
    if not isinstance(text, str):
        return ["%s is a %s; FakeModel returns text" % (where, type(text).__name__)]
    try:
        value = json.loads(text)
    except ValueError as exc:
        return ["%s is not valid JSON (%s)" % (where, exc)]
    if not isinstance(value, dict):
        return ["%s is not a JSON object, so no node grammar can accept it" % where]

    missing = [name for name in node.writes if name not in value]
    if missing:
        problems.append(
            "%s omits %s, which the node's grammar requires"
            % (where, ", ".join(repr(name) for name in missing))
        )
    extra = [name for name in value if name not in node.writes]
    if extra:
        problems.append(
            "%s carries %s, which the node does not write (its grammar sets "
            "additionalProperties: false)" % (where, ", ".join(repr(n) for n in extra))
        )
    if index < len(owed):
        case_name, expected = owed[index]
        for name in node.writes:
            if name in value and name in expected and value[name] != expected[name]:
                problems.append(
                    "%s answers %s=%r, but case %r expects %r"
                    % (where, name, value[name], case_name, expected[name])
                )
    problems.extend(_check_enums(task, node, value, where))
    return problems


def _check_enums(task, node, value, where):
    """An answer outside its field's enum is the 11-of-12 trap.

    When the examples showed a closed set in every case but one, `analyze` declares the
    enum and `induce` puts it in the node's grammar — and this stage faithfully scripts
    the twelfth case's odd value, because the gold answer is never edited. What comes out
    is a pack that fails its own evalset with a grammar rejection at a node nobody
    suspects. Saying it here costs one dict lookup.
    """
    problems = []
    enums = {spec.name: spec.enum for spec in task.fields if spec.enum}
    for name in node.writes:
        if name not in value or name not in enums:
            continue
        if value[name] not in enums[name]:
            problems.append(
                "%s answers %s=%r, which is outside the field's enum %r: the node's "
                "grammar rejects it and the case fails at a node that looks innocent"
                % (where, name, value[name], list(enums[name]))
            )
    return problems


def _first_unreached(owed, held):
    if held >= len(owed):
        return ""
    return " — case %r is the first one with nothing left to answer it" % owed[held][0]


def _check_think_entry(script, two_stage, keys):
    problems = []
    entry = script.get(THINK_KEY, _MISSING)
    if entry is _MISSING:
        if two_stage:
            problems.append(
                "no think entry: %s two-stage, and without a think.txt its think prompt "
                "is the emit prompt plus a suffix, so its first emit answer is consumed "
                "by the think call" % _nodes_are(two_stage)
            )
        return problems
    if not two_stage:
        problems.append(
            "the script holds a think entry, but no node in the plan is two-stage"
        )
    if isinstance(entry, list):
        problems.append(
            "the think entry is a queue; the retry ladder re-thinks every time it "
            "rejects a two-stage answer, so a queue there runs out mid-run"
        )
        entry = entry[0] if entry else ""
    if isinstance(entry, str):
        for key in keys:
            if key in entry:
                problems.append(
                    "the scripted think notes contain %r, and the notes are quoted back "
                    "into the emit prompt, where they would win the match" % key
                )
    return problems


def _check_prompts(script, plan, prompts, keys):
    """Does each node's own prompt actually resolve to its own entry?"""
    problems = []
    for node in plan.nodes:
        template = prompts.get(node.name)
        if template is None:
            problems.append(
                "node %r has no prompt, so nothing can match its entry" % node.name
            )
            continue
        matches = [key for key in script if key in template]
        if not matches:
            problems.append(
                "node %r's prompt contains none of the script's keys; it must contain "
                "%r" % (node.name, node_key(node.name))
            )
            continue
        winner = max(matches, key=len)
        if winner != node_key(node.name):
            problems.append(
                "node %r's prompt matches %r before its own key %r, so it would answer "
                "with %s"
                % (node.name, winner, node_key(node.name),
                   "the think notes" if winner == THINK_KEY
                   else "node %r's answers" % keys.get(winner, winner))
            )
    return problems


def _check_written_twice(plan):
    seen = {}
    problems = []
    for node in plan.nodes:
        for name in node.writes:
            if name in seen:
                problems.append(
                    "field %r is written by both %r and %r, so the later answer "
                    "overwrites the earlier one" % (name, seen[name], node.name)
                )
            else:
                seen[name] = node.name
    return problems


def _check_input_collisions(task, plan):
    """A node that writes a run input takes down every case, not one.

    Merge-mode commit refuses to overwrite a key the caller supplied (`graph.commit`
    raises `StateCollision`), so this is not a subtle scoring problem — it is the whole
    evalset failing at the first node with a message about state rather than about the
    plan.
    """
    inputs = set(task.inputs)
    for case in task.cases:
        inputs.update(case.get("input") or {})
    problems = []
    for node in plan.nodes:
        for name in node.writes:
            if name in inputs:
                problems.append(
                    "node %r writes %r, which is a run input: committing it raises "
                    "StateCollision and every case dies at that node"
                    % (node.name, name)
                )
    return problems


def _check_placeholders(task, plan):
    """Fields the script has to invent, reported once per field rather than per case."""
    problems = []
    pinned = set()
    for case in task.cases:
        pinned.update(case.get("expect") or {})
    for node in plan.nodes:
        for name in node.writes:
            if name in pinned:
                continue
            try:
                value = _placeholder(task, name)
            except BuildError as exc:
                problems.append("node %r writes %r: %s" % (node.name, name, exc))
                continue
            problems.append(
                "node %r writes %r, which no gold case pins: the script answers %r "
                "every time, and nothing tests it" % (node.name, name, value)
            )
    return problems


def _check_endings(task, plan):
    """A case that says where it ends is asserting the plan's routing; check it here."""
    problems = []
    for case in task.cases:
        declared = case.get("end")
        if not declared:
            continue
        reached = route(task, plan, case).ending
        if reached != declared:
            problems.append(
                "case %r says it ends at %r, but the plan's edges take it to %r; its "
                "answers are queued at the wrong nodes"
                % (_case_name(case), declared, reached)
            )
    return problems


def _nodes_are(names):
    if len(names) == 1:
        return "node %r is" % names[0]
    return "nodes %s are" % ", ".join(repr(name) for name in names)
