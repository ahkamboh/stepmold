"""Stage 1 of `jig build` — the schema, induced from the gold examples.

No model, no network, no randomness: everything here is arithmetic over the `expect`
objects the author already wrote. `analyze` is the first stage precisely *because* it
needs nothing, and everything downstream (the planner, the prompt writer, the offline
script) is allowed to assume the answer it produces.

What it derives, per key seen in any `expect`:

| property   | how it is decided                                                  |
| ---------- | ------------------------------------------------------------------ |
| `type`     | the one JSON type that covers every non-null observation           |
| `optional` | absent from some `expect`, or null in some                          |
| `enum`     | the closed-set rule below                                           |
| `examples` | the first few distinct non-null values, verbatim, for the prompt    |

Booleans are checked before integers, as `jig/grammar.py` does, because Python says a
bool is an int and JSON says it is not. An `int` in one case and a `float` in another is
`number` — the only widening the compiler performs, because it is the only one JSON's
type list makes lossless.

`array` and `object` are inferred too, though `FieldSpec` names only the four scalar
types: `examples/meeting_actions` puts a list of action items straight into `expect`, and
a compiler that refused it could not compile a pack this repo ships. Their element
schemas are left to a later stage; this one only reports the outer type.


The enum rule
-------------

An enum is the highest-value thing this stage infers, because it is the one constraint
that makes a wrong answer *unrepresentable* under a constrained decoder rather than
merely unlikely. It is also the one that is expensive to get wrong: a value missing from
the enum can never be produced, no matter how obviously correct it is, so an over-eager
guess silently caps the finished pack's ceiling below 100%.

A field gets an enum only when all of these hold:

1. its type is exactly `string`. A boolean is already a closed set of two, and an
   integer that happens to take four values in the gold set (`action_count` is 0, 2, 3
   or 4 in `meeting_actions`) is a count, not a category — enumerating it would forbid
   the fifth action item.
2. at least `MIN_ENUM_OBSERVATIONS` (4) non-null observations. Three cases cannot
   distinguish a closed set from a coincidence.
3. between 2 and `MAX_ENUM_VALUES` (12) distinct values. One value is a habit of this
   gold set, not a vocabulary — `meeting_actions.lead` is `"Maya"` once and null eleven
   times, and an enum of `["Maya"]` would be a person's name welded into a grammar.
   Beyond twelve, the "enum" has stopped being a decoding constraint and become a
   memorised list of whatever the author happened to write down.
4. every value is label-shaped: at most `MAX_LABEL_LENGTH` (40) characters, one line,
   no leading or trailing space. This is what keeps a free-text field out: a `reason`
   that is `"none"` in half the cases and a distinct sentence in the rest passes the
   arithmetic and fails here.
5. **the majority of observations land on a value that recurs.** Summing the counts of
   every value seen `MIN_TIMES_SEEN` (2) or more times must reach at least half the
   observations.

Rule 5 is the load-bearing one, and it is what decides a field that is enum-like in 11
of 12 cases with a unique twelfth: 11 of 12 observations recur, so it is a vocabulary
with a rare member, and the rare member goes *into* the enum rather than being dropped
from it. The strictest reading — *every* value must recur — throws real enums away:
`content_moderation.policy_category` takes eight values in thirteen cases with five of
them seen once, and it is unmistakably a vocabulary. The loosest reading — *any* value
recurs — admits free text the moment two cases share a `"none"`. A majority separates
them, and the length guard catches what slips through.

The enum is every distinct non-null value observed, sorted, singletons included: a value
in the gold set that the grammar cannot express is a case the pack can never pass.

Measured against the six packs in `examples/` — 54 fields, of which the packs
enum-constrain 32 (`tests/test_build_analyze.py` runs this):

    24  induced identically to the shipped grammar
     6  induced as a strict subset of it
     2  not induced at all (`incident_triage.defect`, `lead_qualify.industry`)
     0  invented, widened, or inferred for a field the packs leave open

The last line is the one worth defending. Every open field in those packs
(`supplier_name`, `invoice_number`, `due_date`, `order_id`, `review_note`, `lead`,
`fallback_owner`) scores exactly zero recurrence — all of their values are distinct,
which is what "open" means — so the majority test has room to spare on both sides.

Two shapes of shyness remain, and they differ in how they fail. Inferring *nothing*
(`defect`: three cases, three values, one each) costs accuracy a later stage or the
author can add back. Inferring a *subset* — `invoice_extract` ships seven currencies and
its evalset shows four — is the real residual risk: the compiled pack still passes the
gold set, which is the contract it is measured against, but a run that should answer
`"JPY"` cannot. That is a widening a stage with a model and the task description can
make later; it is not one this stage can make from arithmetic, and guessing at it here
would be inventing values, which is the one thing the numbers above say never happens.
"""

import json
from collections import OrderedDict

from .spec import BuildError, FieldSpec, TaskSpec

__all__ = ["analyze"]

# See "The enum rule" above. These are the whole policy; nothing else tunes it.
MIN_ENUM_OBSERVATIONS = 4
MAX_ENUM_VALUES = 12
MIN_TIMES_SEEN = 2
MAX_LABEL_LENGTH = 40

MAX_EXAMPLES = 4

# Checked in order, and `bool` must precede `int`: `isinstance(True, int)` is True in
# Python and false in JSON, and a boolean typed as an integer would let a grammar accept
# `1` where the pack means `true`.
_JSON_TYPES = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (list, "array"),
    (dict, "object"),
)

# The one pair of types a single JSON type name covers. Everything else that disagrees is
# an authoring mistake the compiler must not paper over.
_WIDENS_TO_NUMBER = frozenset(["integer", "number"])


def analyze(description, cases, name):
    """Induce a `TaskSpec` from a task description and the gold cases.

    `cases` are evalset lines — `{"input": {...}, "expect": {...}}`, optionally carrying
    "name", "end" and "rescued", which this stage reads for diagnostics and otherwise
    leaves alone. The cases are carried into the spec verbatim: they are the contract the
    compiled pack is measured against, and the compiler never edits them.
    """
    if not cases:
        raise BuildError(
            "no gold examples: `jig build` induces the schema from the examples, so it "
            "needs at least one case of the form "
            '{"input": {...}, "expect": {...}}'
        )

    inputs = _input_keys(cases)
    fields = [
        _field_spec(key, _observations(key, cases))
        for key in _expect_keys(cases)
    ]
    return TaskSpec(
        name=name,
        description=description,
        inputs=inputs,
        fields=fields,
        cases=list(cases),
    )


# --------------------------------------------------------------------------- inputs


def _input_keys(cases):
    """The keys every case supplies, in the order the first case lists them.

    Insisting they match across cases is not pedantry: the prompt writer renders one
    template against every case, so a key that is present in nine cases and missing in
    the tenth is a `KeyError` at eval time in a pack that looked fine at build time.
    """
    first = _case_object(cases[0], 0, "input")
    expected = list(first)
    for index, case in enumerate(cases[1:], start=1):
        keys = list(_case_object(case, index, "input"))
        if set(keys) != set(expected):
            missing = sorted(set(expected) - set(keys))
            extra = sorted(set(keys) - set(expected))
            raise BuildError(
                "%s: 'input' keys differ from %s%s%s. Every case must supply the same "
                "input keys — add the missing ones, or drop them from every case."
                % (
                    _where(cases, index),
                    _where(cases, 0),
                    _clause(" (missing %s)", missing),
                    _clause(" (unexpected %s)", extra),
                )
            )
    return expected


def _expect_keys(cases):
    """Every key seen in any `expect`, in order of first appearance.

    Order of first appearance, rather than sorted, because it is the order the author
    wrote and the order a reader of the evalset already has in their head.
    """
    keys = OrderedDict()
    for index, case in enumerate(cases):
        for key in _case_object(case, index, "expect"):
            keys[key] = True
    if not keys:
        raise BuildError(
            "every 'expect' is empty: there is no field for the compiled pack to "
            "produce. Put the values the workflow must output in 'expect'."
        )
    return list(keys)


def _observations(key, cases):
    """Every value this key took, plus how many cases left it out."""
    seen = []
    absent = 0
    nulls = 0
    for index, case in enumerate(cases):
        expect = _case_object(case, index, "expect")
        if key not in expect:
            absent += 1
        elif expect[key] is None:
            nulls += 1
        else:
            seen.append((index, expect[key]))
    return seen, absent, nulls


# ---------------------------------------------------------------------------- fields


def _field_spec(key, observations):
    seen, absent, nulls = observations
    if not seen:
        raise BuildError(
            "field %r is null or absent in every example, so the compiler cannot tell "
            "what type it should be. Give at least one case a real value for %r, or "
            "drop the field." % (key, key)
        )
    values = [value for _, value in seen]
    type_name = _one_type(key, seen)
    return FieldSpec(
        name=key,
        type=type_name,
        enum=_enum(type_name, values),
        optional=bool(absent or nulls),
        examples=_examples(values),
    )


def _one_type(key, seen):
    """The single JSON type covering every observation, or a BuildError naming both."""
    chosen, chosen_at = _type_of(seen[0][1]), seen[0][0]
    for index, value in seen[1:]:
        name = _type_of(value)
        if name == chosen:
            continue
        if {name, chosen} <= _WIDENS_TO_NUMBER:
            chosen = "number"  # an int here and a float there is one JSON `number`
            continue
        raise BuildError(
            "field %r is %s in case %d and %s in case %d — no single JSON type covers "
            "both. Make %r the same type in every case, or split it into two fields."
            % (key, chosen, chosen_at + 1, name, index + 1, key)
        )
    return chosen


def _type_of(value):
    for python_type, name in _JSON_TYPES:
        if isinstance(value, python_type):
            return name
    raise BuildError(
        "a value of type %s appeared in 'expect'; the examples must be JSON, and JSON "
        "has no %s." % (type(value).__name__, type(value).__name__)
    )


def _enum(type_name, values):
    """The closed set of values this field takes, or None. See the module docstring."""
    if type_name != "string":
        return None
    if len(values) < MIN_ENUM_OBSERVATIONS:
        return None

    counts = OrderedDict()
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not 2 <= len(counts) <= MAX_ENUM_VALUES:
        return None
    if not all(_label_shaped(value) for value in counts):
        return None

    # The majority test: how much of the gold set landed on a value the field has taken
    # before. Free text scores zero here however many cases there are, because "free"
    # means no value ever comes back.
    recurring = sum(n for n in counts.values() if n >= MIN_TIMES_SEEN)
    if recurring * 2 < len(values):
        return None
    return sorted(counts)


def _label_shaped(value):
    """Is this a label a decoder could be pinned to, or is it prose?"""
    return (
        len(value) <= MAX_LABEL_LENGTH
        and "\n" not in value
        and value == value.strip()
    )


def _examples(values):
    """A few distinct values, in the order the author wrote them.

    For the prompt writer to quote, so keep them verbatim — a paraphrased example is
    a prompt that teaches the model a format the grammar will then reject.
    """
    chosen = []
    seen = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        chosen.append(value)
        if len(chosen) == MAX_EXAMPLES:
            break
    return chosen


# ----------------------------------------------------------------------- diagnostics


def _case_object(case, index, key):
    """`case[key]`, insisting it is there and is an object."""
    if not isinstance(case, dict):
        raise BuildError(
            "case %d is a %s; every case must be an object of the form "
            '{"input": {...}, "expect": {...}}' % (index + 1, type(case).__name__)
        )
    if key not in case:
        raise BuildError(
            "case %d has no %r. Every case needs both 'input' and 'expect'."
            % (index + 1, key)
        )
    value = case[key]
    if not isinstance(value, dict):
        raise BuildError(
            "case %d: %r is a %s; it must be an object mapping names to values."
            % (index + 1, key, type(value).__name__)
        )
    return value


def _where(cases, index):
    """How to refer to a case in an error — by the author's name for it, if it has one."""
    case = cases[index]
    label = case.get("name") if isinstance(case, dict) else None
    if isinstance(label, str) and label.strip():
        return "case %d (%r)" % (index + 1, _clip(label))
    return "case %d" % (index + 1)


def _clause(template, names):
    return template % ", ".join(repr(name) for name in names) if names else ""


def _clip(text, limit=40):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."
