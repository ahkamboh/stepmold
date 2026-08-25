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
contain. When the prompts do not exist yet (the scribe runs in parallel with this stage)
there is nothing to read a key off, so this stage publishes one instead: `node_key` is
that contract, and every generated prompt must contain the phrase "the <node> step".

When the prompts *do* exist — regenerating the offline model for a pack somebody already
wrote by hand — guessing is the wrong move, because a hand-written prompt says "the
clear-post step" or "the routing step" and no published convention will ever cover both.
So `script_for` and `check_script` both accept the templates, and `keys_for` reads a key
off each one: the node's own marker when the prompt happens to carry it, and otherwise
the first literal (placeholder-free) line of the template, which is the one span of a
prompt that survives rendering unchanged. Either way the chosen key is checked to occur
in its own template and in no other, so the longest-match cannot land on a neighbour.

**Values are queues, never bare strings.** A string answers every matching prompt, so a
node whose script is a string can never run out — and a node that can never run out is a
node whose branching is never checked. A list of exactly the answers the reaching cases
expect turns a routing mistake into a loud failure at eval time, which is the whole
point. The queue is in evalset order because `jig eval` builds one `FakeModel` for the
whole run (`cli.resolve_model`) and walks the cases in file order.

**The think stage is the exception.** A two-stage node makes two calls on one visit, so
it needs a key of its own for the first one. With a `prompts/<node>.think.txt` that key
comes off the think template like any other. Without one the think prompt is the emit
prompt plus a fixed suffix (`codegen.DEFAULT_THINK_SUFFIX`), so any key that matches the
emit call matches the think call too, and a queue would have its first answer eaten by a
call whose output is thrown away. The fix is one entry keyed on the suffix itself: it is
longer than any node key, so it wins the longest-match, and its value is a plain string
so it answers every think call in the run. That also survives the retry ladder, which
re-thinks whenever it rejects a two-stage answer (`verify.run_node`) — a queue there
would run out on the first rejection.

**The linter replays the run.** `check_script` does not ask whether the script's keys
look right; it walks every gold case through the plan, builds the prompt each call would
actually be given — rendered, with the think notes quoted back in, exactly as
`codegen.build_prompt` does — and asks `FakeModel`'s own question of it: which key wins.
That is the only version of the check that is true of a real pack. `examples/incident_triage`
keys its script on the alert id, so its keys appear in no template and only in a
*rendered* prompt; `examples/content_moderation` names a node `clear_post` and its prompt
"the clear-post step". A linter that assumed either would call a shipping pack broken,
which is worse than no linter at all, because the first thing anyone does with one is
point it at a pack they know works.
"""

import json
from dataclasses import dataclass
from typing import List, Optional

from ..codegen import DEFAULT_THINK_SUFFIX, SCRATCHPAD, SCRATCHPAD_BLOCK
from ..errors import MissingVariable
from ..render import render
from .spec import BuildError

__all__ = [
    "MARKER",
    "Route",
    "THINK_ANSWER",
    "THINK_KEY",
    "check_script",
    "keys_for",
    "node_key",
    "route",
    "script_for",
]

# The phrase a node's prompt must contain for its scripted answers to be found. Keep it
# longer than a bare node name: "priority" appears in a prompt that merely *reads*
# `{priority}`, and would then swallow another node's answers.
MARKER = "the %s step"

# Distinct from every node key by construction (see `_check_think_keys`), so the think
# call resolves here and the emit call resolves to the node.
THINK_KEY = DEFAULT_THINK_SUFFIX.strip()
THINK_ANSWER = (
    "Weighing the case against the rules above before committing to an answer."
)

# A case that has not reached an ending by here is going round a loop the gold answers
# cannot break, since the answers this stage feeds a node are the same every time round.
_MAX_STEPS = 100

# The least assertive value of each type that `FieldSpec.schema` actually accepts. `array`
# and `object` are here because `analyze` induces both — `examples/meeting_actions` puts a
# list of action items straight into `expect` — and a missing entry does not fail loudly:
# `.get(type, "")` would hand an array field the empty *string*, which every grammar in
# the pack rejects at the node that writes it. Values are built per call rather than
# shared, so no answer can hold a mutable object another answer also holds.
_ZERO = {
    "string": lambda: "",
    "integer": lambda: 0,
    "number": lambda: 0.0,
    "boolean": lambda: False,
    "array": lambda: [],
    "object": lambda: {},
}


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


@dataclass(frozen=True)
class _Call:
    """One model call the evalset makes, and the script key `FakeModel` would pick.

    `owed` is the answer object the call has to produce, or None for a think call, whose
    output is notes that never reach state. `pins` is the subset of it the gold case
    actually asserts: the rest is placeholder, and holding a hand-written script to a
    value this stage invented would report every free-text field in the repository as a
    disagreement. `index` is the call's position in its key's queue, which is what pairs
    an answer with the case that consumes it.
    """

    case: str
    node: str
    stage: str                  # "think" | "emit"
    key: Optional[str]
    index: int
    owed: Optional[dict] = None
    pins: Optional[dict] = None


def script_for(task, plan, prompts=None, think_prompts=None):
    """Build the keyed `FakeModel` script that replays `task`'s gold answers.

    The result is the object `fakes/script.json` holds: `{key: [answer, ...]}`, one queue
    per node in evalset order, plus a think entry per two-stage node.

    `prompts` and `think_prompts` are `{node: template}` when the caller already has the
    pack's prompts — regenerating the offline model for a pack whose prompts are
    hand-written. Given them, the keys are read off those templates and are guaranteed to
    match; without them the keys are the published `node_key` markers and it is the
    prompt writer's job to carry them.
    """
    if not task.cases:
        raise BuildError(
            "cannot script an offline model from an empty examples file: the scripted "
            "answers are the gold answers, so a pack with no cases has nothing to say"
        )
    _check_expectations(task, plan)
    keys = keys_for(plan, prompts, think_prompts)
    _check_think_keys(plan, keys)

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
        if not answers:
            # A node no case reaches gets no entry rather than an empty queue: an empty
            # queue is a promise of answers that would fail as `ModelExhausted` mid-run,
            # and the honest report of an unexercised node is `check_script`'s, not a
            # silent stub's.
            continue
        emit_key, think_key = keys[node.name]
        if think_key is not None:
            # A plain string, not a queue: the retry ladder re-thinks every time it
            # rejects a two-stage answer, so a queue here runs out at the first rejection.
            script[think_key] = THINK_ANSWER
        script[emit_key] = answers
    if not script:
        # `FakeModel` refuses an empty script, and it is right to: a pack whose offline
        # model answers nothing cannot be evaluated at all.
        raise BuildError(
            "no node in the plan is reached by any gold case, so there is nothing to "
            "script; check the plan's entry (%r) and its edges" % plan.entry
        )
    return script


def keys_for(plan, prompts=None, think_prompts=None):
    """Pick each node's script keys: `{node: (emit_key, think_key or None)}`.

    `think_key` is None for a single-stage node, `THINK_KEY` for a two-stage node whose
    think prompt is the emit prompt plus the default suffix (there is nothing else in it
    to key on), and a key read off `prompts/<node>.think.txt` when the pack ships one.

    Every key returned is checked to occur in its own template and in no other, because
    the failure mode of getting this wrong is not an error: it is one node quietly
    answering with another node's queue, and an evalset that scores the wrong thing.
    """
    templates = {}
    if prompts:
        for node in plan.nodes:
            template = prompts.get(node.name)
            if template is not None:
                templates[("emit", node.name)] = template
        for node in plan.nodes:
            template = (think_prompts or {}).get(node.name)
            if node.two_stage and template is not None:
                templates[("think", node.name)] = template

    keys = {}
    for node in plan.nodes:
        emit = _key_from(("emit", node.name), templates, node_key(node.name))
        think = None
        if node.two_stage:
            if ("think", node.name) in templates:
                think = _key_from(("think", node.name), templates, node_key(node.name))
            else:
                think = THINK_KEY
        keys[node.name] = (emit, think)
    return keys


def _key_from(label, templates, preferred):
    """A key for one template: its node marker when that works, else its first literal line."""
    template = templates.get(label)
    if template is None:
        # No template to read: the published marker is the contract the scribe owes, and
        # `check_script` is where the two halves are finally compared.
        return preferred
    for candidate in (preferred, _literal_line(template)):
        if candidate and _only_in(candidate, label, templates):
            return candidate
    raise BuildError(
        "cannot find a script key for the %s prompt of node %r: it neither contains %r "
        "nor opens with a line that is free of {placeholders} and unique to it. A key "
        "must survive rendering and must appear in no other prompt in the pack"
        % (label[0], label[1], preferred)
    )


def _literal_line(template):
    """The first line of a template that rendering leaves alone.

    A `{name}` is replaced with run state, so any line holding one is a different string
    in every case and cannot be a key. The first line without one is the header the pack
    author wrote to say which step this is, which is exactly the span a key wants.
    """
    for line in template.splitlines():
        line = line.strip()
        if line and "{" not in line and "}" not in line:
            return line
    return None


def _only_in(candidate, label, templates):
    if candidate not in templates[label]:
        return False
    return not any(
        candidate in text for other, text in templates.items() if other != label
    )


def route(task, plan, case):
    """Walk `plan` the way the runtime would, answering each node from `case`.

    The walker picks the first outgoing edge whose `when` matches state
    (`graph._next`), and state here is the case's inputs plus the gold answers committed
    so far — which is exactly what state holds at that moment in a run the case passes.
    So the nodes this returns are the nodes that case will actually generate at.
    """
    visits, ending = _walk(task, plan, case)
    return Route(nodes=[node.name for node, _, _ in visits], ending=ending)


def _walk(task, plan, case):
    """`route`, but keeping the state and the answer each visit was made with.

    The linter needs both to rebuild the prompt a call was actually given, and rebuilding
    that prompt is the only way to ask which script key answers it.
    """
    state = dict(case.get("input") or {})
    endings = set(plan.endings)
    visits = []
    name = plan.entry
    for _ in range(_MAX_STEPS):
        if name in endings:
            return visits, name
        node = _node_or_error(plan, name, case)
        answer = _answer(task, node, case)
        visits.append((node, dict(state), answer))
        state.update(answer)
        name = _successor(plan, node, state, case)
    raise BuildError(
        "case %r still had not reached an ending after %d nodes; the plan's edges send "
        "it round a loop that the gold answers never break"
        % (_case_name(case), _MAX_STEPS)
    )


def check_script(script, task, plan, prompts=None, think_prompts=None):
    """Lint a script against the plan, the prompts and the gold cases; return the problems.

    Human-readable strings, one per problem, empty when there is nothing to say. This is
    the check nobody had: every failure it names shows up otherwise as `ModelExhausted`
    at some unrelated case, or — worse — as a green evalset that is scoring the wrong
    node's answers.

    `prompts` and `think_prompts` are the pack's `{node: template}` mappings. Both are
    optional and both are worth having: without them the linter has to assume the node
    markers this stage publishes, and a hand-written pack is under no obligation to use
    them. `think_prompts` in particular is the difference between passing a pack and
    proving it: a two-stage node makes two calls per visit, and a linter that reasons
    about one of them will bless a pack that dies with `ModelExhausted` on the other.

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

    try:
        calls, runs = _simulate(script, task, plan, prompts, think_prompts)
        walked = True
    except BuildError as exc:
        calls, runs, walked = [], [], False
        problems.append(
            "the plan cannot be walked, so no queue length can be checked: %s" % exc
        )

    if prompts:
        problems.extend(
            "node %r has no prompt, so nothing can match its entry" % node.name
            for node in plan.nodes
            if prompts.get(node.name) is None
        )
    if walked:
        problems.extend(_check_resolution(script, plan, calls))
        problems.extend(_check_queues(script, task, plan, calls))
    problems.extend(_check_think_entry(script, plan, calls, walked))
    problems.extend(_check_input_collisions(task, plan))
    problems.extend(_check_rescues(task))
    if walked:
        problems.extend(_check_written_twice(plan, runs))
        problems.extend(_check_placeholders(task, plan))
        problems.extend(_check_endings(task, runs))
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
    The value has to be the least assertive thing the grammar *accepts*, and the grammar
    is `FieldSpec.schema`: `{"type": "string"}`, with an `enum` when one was induced and
    never a null. So the order is the enum first (an enum'd field admits nothing else),
    then a value the examples actually showed, then the zero of the field's type.

    Null is deliberately not in that list, and this is the correction of a real bug: the
    `optional` flag says the examples showed the field absent or null somewhere, but
    `FieldSpec.schema` emits no `"null"` for it, so scripting null answered most of the
    optional fields in the example packs with a value their own grammar rejects. A field
    that is genuinely nullable and has no non-null example left to borrow therefore gets
    the typed zero — `""`, `0`, `[]` — which is a lie about the data but a legal one, and
    `_check_placeholders` reports every field this happens to so the author can see it.
    """
    spec = task.field_named(name)   # raises BuildError naming the field
    if spec.enum:
        return spec.enum[0]
    for value in spec.examples:
        # A real observation beats an invented zero: it is the right shape for an array
        # or object field, and it reads as the pack's own data in the diff.
        if value is not None:
            return value
    return _ZERO.get(spec.type, _ZERO["string"])()


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


# --------------------------------------------------------------- the replay


def _simulate(script, task, plan, prompts, think_prompts):
    """Replay the whole evalset against `script` exactly as `jig eval` would.

    Returns the calls in run order, each carrying the key `FakeModel` would resolve it
    to, and one `(case, nodes, ending)` triple per case. Nothing here trusts a naming
    convention: the prompt is built the way `codegen` builds it — rendered against the
    state the case holds at that moment, with the scripted think notes quoted back in —
    and the key is chosen the way `FakeModel` chooses it.
    """
    calls = []
    runs = []
    used = {}
    for case in task.cases:
        name = _case_name(case)
        visits, ending = _walk(task, plan, case)
        runs.append((name, [node.name for node, _, _ in visits], ending))
        pinned = case.get("expect") or {}
        for node, state, answer in visits:
            pins = {f: pinned[f] for f in node.writes if f in pinned}
            notes = None
            if node.two_stage:
                prompt = _think_prompt(node, state, prompts, think_prompts)
                call = _record(script, calls, used, name, node, "think", prompt, None)
                notes = _scripted(script, call)
                # A think call whose key is missing returns nothing at run time, so the
                # emit prompt it feeds carries no notes rather than the string "None".
                notes = "" if notes is None else notes
            prompt = _emit_prompt(node, state, prompts, notes)
            _record(script, calls, used, name, node, "emit", prompt, answer, pins)
    return calls, runs


def _record(script, calls, used, case, node, stage, prompt, owed, pins=None):
    key = _winner(script, prompt)
    index = used.get(key, 0)
    if key is not None:
        used[key] = index + 1
    call = _Call(case=case, node=node.name, stage=stage, key=key, index=index,
                 owed=owed, pins=pins)
    calls.append(call)
    return call


def _winner(script, prompt):
    """The key `FakeModel._keyed` would pick for `prompt`, or None if it would give up."""
    matches = [key for key in script if isinstance(key, str) and key in prompt]
    if not matches:
        return None
    return max(matches, key=len)


def _scripted(script, call):
    """The text the script hands `call`, or None when its queue has nothing left."""
    if call.key is None:
        return None
    entry = script.get(call.key)
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list) and call.index < len(entry):
        text = entry[call.index]
        return text if isinstance(text, str) else None
    return None


def _emit_prompt(node, state, prompts, notes):
    """The emit prompt for one visit, assembled the way `codegen.build_prompt` does."""
    template = (prompts or {}).get(node.name)
    if template is None:
        # No prompt to read, so stand in the one thing the prompt is contractually
        # required to contain. It is the weakest honest assumption available.
        template = node_key(node.name)
    placeholder = "{%s}" % SCRATCHPAD
    if notes is not None and placeholder in template:
        return _render(template, dict(state, **{SCRATCHPAD: notes}))
    prompt = _render(template, state)
    if notes is not None:
        prompt += SCRATCHPAD_BLOCK % notes
    return prompt


def _think_prompt(node, state, prompts, think_prompts):
    """The think prompt for one visit, assembled the way `codegen.think` does."""
    template = (think_prompts or {}).get(node.name)
    if template is None:
        template = (prompts or {}).get(node.name) or node_key(node.name)
        template += DEFAULT_THINK_SUFFIX
    scope = dict(state)
    scope.setdefault(SCRATCHPAD, "")
    return _render(template, scope)


def _render(template, state):
    # A template that reads a variable this stage cannot supply is the prompt writer's
    # problem, not the script's, and reporting it here would be this linter complaining
    # about someone else's file. The key lives in the fixed half of a template anyway, so
    # the unrendered text answers the only question asked of it.
    try:
        return render(template, state)
    except MissingVariable:
        return template


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


def _check_think_keys(plan, keys):
    """The shared think entry only wins the longest-match if it beats every node key.

    Only nodes that fall back on `THINK_KEY` are at risk: a node with its own think
    template has a key of its own, already proved unique by `keys_for`. Both of these are
    theoretical against sane node names, and both would show up as a node quietly
    returning its neighbour's answer rather than as an error, which is the kind of bug
    that costs an afternoon.
    """
    if not any(think == THINK_KEY for _, think in keys.values()):
        return
    for node in plan.nodes:
        key = keys[node.name][0]
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


def _check_resolution(script, plan, calls):
    """Which key each call lands on: none, its own, or somebody else's."""
    problems = []
    by_stage = {}
    for call in calls:
        by_stage.setdefault((call.node, call.stage), []).append(call)

    # A call no key matches. Reported per node and stage, not per case: thirteen copies
    # of one sentence is not thirteen times the information. The first case it happens to
    # is named, because a pack keyed per case (examples/incident_triage) can lose one
    # alert's key and keep the rest.
    for node in plan.nodes:
        for stage in ("think", "emit"):
            starved = [c for c in by_stage.get((node.name, stage), []) if c.key is None]
            if not starved:
                continue
            problems.append(
                "node %r's %s prompt contains none of the script's keys, so its call at "
                "case %r dies with ModelExhausted; it must contain %r"
                % (node.name, stage, starved[0].case, node_key(node.name))
            )

    # A two-stage node whose two calls land on one key: the think call eats the emit
    # answer, which is the whole reason a think entry exists.
    for node in plan.nodes:
        think = {c.key for c in by_stage.get((node.name, "think"), []) if c.key}
        emit = {c.key for c in by_stage.get((node.name, "emit"), []) if c.key}
        for key in sorted(think & emit):
            problems.append(
                "no think entry for node %r: it is two-stage, and its think prompt "
                "resolves to the same key %r as its emit prompt, so its first emit "
                "answer is consumed by the think call" % (node.name, key)
            )

    problems.extend(_check_shared_keys(plan, calls))
    problems.extend(_check_unused_keys(script, plan, calls))
    for node in plan.nodes:
        if any(call.node == node.name for call in calls):
            continue
        if node_key(node.name) in script:
            problems.append(
                "node %r is reached by no gold case, but the script holds an answer "
                "for it" % node.name
            )
        else:
            problems.append(
                "node %r is reached by no gold case, so nothing scripts it and nothing "
                "tests it" % node.name
            )
    return problems


def _check_shared_keys(plan, calls):
    """One key serving two nodes is one node answering with the other's queue."""
    order = [node.name for node in plan.nodes]
    owners = {}
    for call in calls:
        if call.key is not None:
            owners.setdefault(call.key, []).append(call.node)

    problems = []
    for key, names in owners.items():
        distinct = sorted(set(names), key=order.index)
        if len(distinct) < 2:
            continue
        # The node whose own marker the key is owns it; failing that, the first one the
        # plan declares. Everybody else is being answered by somebody else's queue.
        owner = next((n for n in distinct if node_key(n) == key), distinct[0])
        for name in distinct:
            if name == owner:
                continue
            problems.append(
                "node %r's prompt matches %r before its own key %r, so it would answer "
                "with node %r's answers" % (name, key, node_key(name), owner)
            )
    return problems


def _check_unused_keys(script, plan, calls):
    """A key no call lands on is dead weight, and usually a typo in a node name."""
    reached = {call.key for call in calls}
    unreached = {node_key(node.name) for node in plan.nodes
                 if not any(call.node == node.name for call in calls)}
    return [
        "script key %r matches no node in the plan: no prompt this pack ships can "
        "contain it, so its answers are never used" % key
        for key in script
        if key not in reached and key not in unreached
    ]


def _check_queues(script, task, plan, calls):
    """Queue length and answer content, per key, in the order the run consumes them."""
    problems = []
    by_key = {}
    for call in calls:
        if call.key is not None:
            by_key.setdefault(call.key, []).append(call)

    for key, served in by_key.items():
        entry = script[key]           # `_winner` only ever names a key the script holds
        emit = [call for call in served if call.stage == "emit"]
        if isinstance(entry, str):
            problems.extend(_check_string_entry(task, plan, key, entry, served, emit))
            continue
        if not isinstance(entry, list):
            problems.append(
                "the script entry for %r is a %s; FakeModel takes a string or a list "
                "of strings" % (key, type(entry).__name__)
            )
            continue
        if len(entry) != len(served):
            problems.append(_length_problem(key, served, len(entry)))
        for index, text in enumerate(entry):
            if index < len(served):
                call = served[index]
            else:
                # An answer past the end of the queue is never consumed, so it is checked
                # for shape but not against a case: pairing it with the last case's gold
                # answer would report a disagreement that no run can ever reach.
                call = _Call(served[-1].case, served[-1].node, "emit", key, index)
            if call.stage == "think":
                continue     # notes are free text; no grammar ever sees them
            problems.extend(_check_answer(task, plan, call, index, text))
    return problems


def _check_string_entry(task, plan, key, entry, served, emit):
    """A string answers every matching prompt, so it can never run out.

    Legal, and occasionally right — `examples/invoice_extract` scripts each of its
    one-answer flag nodes this way, and a node exactly one case reaches loses nothing by
    it. But the moment two calls share the string, the miscount this whole stage exists
    to catch is hidden, so that is where it is worth saying.
    """
    problems = []
    if len(emit) > 1:
        owed = [call.owed for call in emit if call.owed is not None]
        distinct = {json.dumps(answer, sort_keys=True) for answer in owed}
        problems.append(
            "node %r is scripted with a single string, which answers every call: the "
            "%d case(s) that reach it are not being checked against it%s"
            % (served[0].node, len(emit),
               "" if len(distinct) <= 1
               else ", and they do not all expect the same answer")
        )
    if emit:
        problems.extend(_check_answer(task, plan, emit[0], 0, entry))
    return problems


def _length_problem(key, served, held):
    tail = ""
    if held < len(served):
        tail = " — case %r is the first one with nothing left to answer it" % (
            served[held].case
        )
    if all(call.stage == "think" for call in served):
        return "node %r makes %d think call(s) on key %r but its script holds %d " \
               "answer(s)%s" % (served[0].node, len(served), key, held, tail)
    return "node %r is reached by %d gold case(s) but its script holds %d answer(s)%s" % (
        served[0].node, len(served), held, tail
    )


def _check_answer(task, plan, call, index, text):
    problems = []
    node = plan.node_named(call.node)
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
    for name in node.writes:
        pins = call.pins or {}
        if name in value and name in pins and value[name] != pins[name]:
            problems.append(
                "%s answers %s=%r, but case %r expects %r"
                % (where, name, value[name], call.case, pins[name])
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


def _check_think_entry(script, plan, calls, walked):
    """The shared think entry, which is the one entry with a fixed key and fixed shape."""
    problems = []
    two_stage = [node.name for node in plan.nodes if node.two_stage]
    entry = script.get(THINK_KEY, _MISSING)
    if entry is _MISSING:
        if two_stage and not walked:
            # With a walk, `_check_resolution` says something sharper: which key the
            # think call actually landed on. Without one this is all that can be said.
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
        # Only the shared entry has to be a string. A per-node think key read off a
        # `think.txt` answers one node's think calls and can carry an exact queue, which
        # is what four of the packs in examples/ do.
        problems.append(
            "the think entry is a queue; the retry ladder re-thinks every time it "
            "rejects a two-stage answer, so a queue there runs out mid-run"
        )
        entry = entry[0] if entry else ""
    if isinstance(entry, str):
        for key in script:
            if key != THINK_KEY and key in entry:
                problems.append(
                    "the scripted think notes contain %r, and the notes are quoted back "
                    "into the emit prompt, where they would win the match" % key
                )
    return problems


def _check_written_twice(plan, runs):
    """Two nodes writing one field, on a path a gold case actually takes.

    Route-aware on purpose. Two nodes on *different* branches may own the same field and
    routinely do — `examples/invoice_extract` has five nodes that write `review_reason`,
    one per terminal reason, and exactly one of them runs. Only nodes that a single case
    visits in turn can overwrite each other.
    """
    problems = []
    seen = set()
    for _, names, _ in runs:
        written = {}
        for name in names:
            for field in plan.node_named(name).writes:
                earlier = written.get(field)
                if earlier is not None and earlier != name:
                    if (field, earlier, name) in seen:
                        continue
                    seen.add((field, earlier, name))
                    problems.append(
                        "field %r is written by both %r and %r, so the later answer "
                        "overwrites the earlier one" % (field, earlier, name)
                    )
                else:
                    written[field] = name
    return problems


def _check_rescues(task):
    """A case that declares a rescue cannot be scripted from this plan.

    `rescued: true` says the case is *meant* to burn a node's retry ladder and take its
    `on_fail` edge, so that node needs one answer per rung plus one for the rescue path —
    and a `GraphPlan` records neither `on_fail` nor which node is supposed to fail. One
    answer per visit is what this stage can honestly produce, so the case is named here
    rather than scripted wrongly and left to fail at eval time.
    """
    return [
        "case %r declares rescued: true, which needs a node to fail its whole ladder; "
        "the plan says nothing about on_fail, so its answers are scripted as if it "
        "passed first time" % _case_name(case)
        for case in task.cases
        if case.get("rescued")
    ]


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


def _check_endings(task, runs):
    """A case that says where it ends is asserting the plan's routing; check it here."""
    problems = []
    for case, (name, _, reached) in zip(task.cases, runs):
        declared = case.get("end")
        if declared and reached != declared:
            problems.append(
                "case %r says it ends at %r, but the plan's edges take it to %r; its "
                "answers are queued at the wrong nodes" % (name, declared, reached)
            )
    return problems


def _nodes_are(names):
    if len(names) == 1:
        return "node %r is" % names[0]
    return "nodes %s are" % ", ".join(repr(name) for name in names)
