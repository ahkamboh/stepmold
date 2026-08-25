"""Stage 1 of `stepmold build` — the schema, induced from the gold examples.

No model, no network, no randomness: everything here is arithmetic over the cases the
author already wrote — the `expect` objects, plus (for rule 6 below) the question of
whether an `input` already spelled a value out. `analyze` is the first stage precisely
*because* it needs nothing, and everything downstream (the planner, the prompt writer,
the offline script) is allowed to assume the answer it produces.

What it derives, per key seen in any `expect`:

| property   | how it is decided                                                  |
| ---------- | ------------------------------------------------------------------ |
| `type`     | the one JSON type that covers every non-null observation           |
| `optional` | absent from some `expect`, or null in some                          |
| `enum`     | the closed-set rule below                                           |
| `examples` | the first few distinct non-null values, verbatim, for the prompt    |

Booleans are checked before integers, as `stepmold/grammar.py` does, because Python says a
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
6. **more than one value has to come back.** At least `MIN_RECURRING_VALUES` (2) distinct
   values must be seen `MIN_TIMES_SEEN` or more times — or, if only one is, the field
   must pass the transcription test below.

Rule 5 is the load-bearing one, and it is what decides a field that is enum-like in 11
of 12 cases with a unique twelfth: 11 of 12 observations recur, so it is a vocabulary
with a rare member, and the rare member goes *into* the enum rather than being dropped
from it. The strictest reading — *every* value must recur — throws real enums away:
`content_moderation.policy_category` takes eight values in thirteen cases with five of
them seen once, and it is unmistakably a vocabulary. The loosest reading — *any* value
recurs — admits free text the moment two cases share a `"none"`. A majority separates
them, and the length guard catches what slips through.

Rule 6 closes the hole rule 3 only half covers. Rule 3 refuses a field with a single
distinct value; rule 5 is satisfied by a single *dominant* value, and those are the same
failure by two routes. An `approver` that is `"maya"` seven times and five other
first names once each scores 7 of 12 recurring — a majority carried entirely by one
habit — and an enum of those six names makes a seventh approver unrepresentable under a
constrained decoder, which no retry and no better model recovers from.

Counting alone cannot finish that job, and it is worth being exact about why, because the
obvious strictening throws real enums away. `invoice_extract.currency` is `"USD"` nine
times and `"EUR"`, `"GBP"`, `"PKR"` once each; `invoice_extract.review_reason` is `"none"`
six times and six distinct reasons once each. Both are shipped enums; both have exactly
one recurring value; and on every count this stage can take — observations, distinct
values, singletons, the dominant value's share — the `approver` above (7 plus five
singletons) scores at least as enum-like as `review_reason` (6 plus six singletons). No
threshold over those numbers admits the one and refuses the other.

What separates them is where the value came from. A name or a ledger code is in the
`expect` because the input already said it: the model reads it off the page, and the page
can always say a name the gold set never showed. A category is in the `expect` because the
model chose it from a vocabulary the input never spells out. So when exactly one value
recurs, the tie is broken on transcription: if `TRANSCRIBED_QUARTERS` (3) quarters or more
of the observations appear verbatim, case-insensitively, in that case's own `input`, the
field is being copied rather than classified, and its singletons are a sample of an open
population instead of the rare members of a closed one. Over the six packs the gap is not
close: no field they enum-constrain scores above 5 of 12 (`meeting_kind`, whose labels do
sometimes appear in the transcript), while every open field that is not prose is copied
in at least eleven observations of twelve — `supplier_name`, `due_date`, `order_id`,
`lead` and `fallback_owner` are copied in all of them. The
test runs only where the counts are ambiguous, so a genuine vocabulary that happens to
echo its input keeps its enum on rule 6's first clause — `category` in the expense pack
below is `"travel"`, `"software"` or `"meals"`, four times each and each word printed on
the receipt.

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

The seventh pack is the one that earned rule 6: twelve expense approvals, written after
the fact to test generalisation rather than to be compiled, whose `approver` and
`cost_center` rules 1-5 happily welded into grammars of six first names and six ledger
codes. It is carried in `tests/test_build_analyze.py` so the shape stays covered.

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
import re
from collections import OrderedDict

from .spec import BuildError, FieldSpec, TaskSpec

__all__ = ["analyze"]

# See "The enum rule" above. These are the whole policy; nothing else tunes it.
MIN_ENUM_OBSERVATIONS = 4
MAX_ENUM_VALUES = 12
MIN_TIMES_SEEN = 2
MIN_RECURRING_VALUES = 2
TRANSCRIBED_QUARTERS = 3      # in quarters, so the comparison stays integer arithmetic
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
            "no gold examples: `stepmold build` induces the schema from the examples, so it "
            "needs at least one case of the form "
            '{"input": {...}, "expect": {...}}'
        )

    inputs = _input_keys(cases)
    haystacks = _haystacks(cases)
    fields = [
        _field_spec(key, _observations(key, cases), haystacks)
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


def _haystacks(cases):
    """Each case's `input`, flattened to one lowercased string to search in.

    Rule 6 of the enum rule asks whether a value was copied out of the input rather than
    chosen, and a substring test over the whole input object answers that without caring
    which key carried it. Lowercased because `"maya"` in the `expect` and `"Maya Chen"`
    in the input are the same transcription.

    `input` is not the compiler's contract to police — only `expect` is, and `_type_of`
    does that — so a value JSON cannot express is stringified here, not raised on.
    """
    return [
        json.dumps(_case_object(case, index, "input"), default=repr,
                   skipkeys=True, ensure_ascii=False).lower()
        for index, case in enumerate(cases)
    ]


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


def _field_spec(key, observations, haystacks):
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
        enum=_enum(type_name, seen, haystacks),
        optional=bool(absent or nulls),
        examples=_examples(values),
    )


def _one_type(key, seen):
    """The single JSON type covering every observation, or a BuildError naming both.

    `chosen` is the running type and may have widened to `number`; `witness` is what the
    case the message names actually held. The author is looking for a line in a file, and
    telling them case 1 is a `number` when the line says `1` sends them to the wrong line.
    """
    witness, witness_at = _type_of(seen[0][1]), seen[0][0]
    chosen = witness
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
            % (key, witness, witness_at + 1, name, index + 1, key)
        )
    return chosen


def _type_of(value):
    """The JSON type name of `value`, refusing anything JSON cannot express.

    Nested values are checked, not just the outer one: `_examples` hands whole values to
    `json.dumps` for the prompt writer, so a set buried in a list would otherwise reach
    the author as a bare `TypeError` from inside the standard library, with no field name
    and no case number anywhere in it.
    """
    for python_type, name in _JSON_TYPES:
        if isinstance(value, python_type):
            if name in ("array", "object"):
                _check_inside(value)
            return name
    raise BuildError(
        "a value of type %s appeared in 'expect'; the examples must be JSON, and JSON "
        "has no %s." % (type(value).__name__, type(value).__name__)
    )


def _check_inside(container):
    """Walk a list or dict, refusing anything JSON cannot express.

    Null is allowed here and has no name in `_JSON_TYPES`: a null *field* is an absent
    observation and never reaches `_type_of`, but a null inside a list of action items is
    an ordinary JSON value.
    """
    if isinstance(container, dict):
        for key in container:
            if not isinstance(key, str):
                raise BuildError(
                    "an object key of type %s appeared in 'expect'; the examples must be "
                    "JSON, and every JSON object key is a string." % type(key).__name__
                )
    for item in (container.values() if isinstance(container, dict) else container):
        if item is not None:
            _type_of(item)


def _mentioned(value, haystack):
    """Whether `haystack` names `value` as a word, rather than merely containing it.

    A bare substring test counts a short label as copied when it never appeared: `ok` sits
    inside `broken`, so a status of ok/warn/fail/skip reads as transcribed and a real enum
    is lost. Word boundaries are what separate "the input said this" from "these letters
    happen to occur".
    """
    needle = value.lower().strip()
    if not needle:
        return False
    return re.search(r"(?<!\w)%s(?!\w)" % re.escape(needle), haystack) is not None


def _enum(type_name, seen, haystacks):
    """The closed set of values this field takes, or None. See the module docstring."""
    if type_name != "string":
        return None
    values = [value for _, value in seen]
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
    repeated = [value for value, n in counts.items() if n >= MIN_TIMES_SEEN]
    recurring = sum(counts[value] for value in repeated)
    if recurring * 2 < len(values):
        return None

    # Rule 6. A majority carried by one dominant value is a habit with a scatter around
    # it, which is the same failure as rule 3's single value by another route: an
    # `approver` seen as "maya" seven times and five other names once each clears the
    # majority test outright. Two values coming back is what a vocabulary being *used*
    # looks like, so that ends it; with only one, the counts cannot tell that field from
    # `invoice_extract.currency`, and the tie goes to where the values came from.
    if len(repeated) < MIN_RECURRING_VALUES:
        transcribed = sum(
            1 for index, value in seen if _mentioned(value, haystacks[index])
        )
        if transcribed * 4 >= len(values) * TRANSCRIBED_QUARTERS:  # in quarters
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
