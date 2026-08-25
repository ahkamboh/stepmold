"""`jig build` stage 1 — schema induction, and the enum rule it turns on.

The last two classes are the ones that matter. `TestAgainstTheShippedPacks` runs `analyze`
over the evalsets of all six packs in `examples/` and holds the result against the
grammars those packs actually ship; `TestTheExpenseApprovalPack` runs it over a seventh
pack the stage was never developed against, whose `approver` and `cost_center` columns are
the shape rule 6 exists to refuse. Everything above them probes a boundary of the enum
rule with the smallest gold set that can express it.
"""

import copy
import glob
import json
import os
import unittest

from jig.build.analyze import (
    MAX_ENUM_VALUES,
    MAX_LABEL_LENGTH,
    MIN_ENUM_OBSERVATIONS,
    MIN_RECURRING_VALUES,
    MIN_TIMES_SEEN,
    TRANSCRIBED_QUARTERS,
    analyze,
)
from jig.build.spec import BuildError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")


def case(expect, **input_keys):
    return {"input": input_keys or {"text": "x"}, "expect": expect}


def cases_for(field, values):
    """One case per value, with a constant input — the shape most tests here need."""
    return [case({field: value}, text="line %d" % index)
            for index, value in enumerate(values)]


def quoting_cases(field, values, quoted):
    """One case per value; the first `quoted` of them print their value in the input.

    Rule 6's transcription test asks whether the input already spelled the answer out, so
    a test of it needs to control that per case rather than per field.
    """
    return [
        {"input": {"text": "case %d says %s" % (index, value if index < quoted else "-")},
         "expect": {field: value}}
        for index, value in enumerate(values)
    ]


def field_of(values, name="f", description="d", pack="p"):
    spec = analyze(description, cases_for(name, values), pack)
    return spec.field_named(name)


# ------------------------------------------------------------------------ the types


class TestTypes(unittest.TestCase):
    def test_a_bool_is_not_an_integer(self):
        # Python says isinstance(True, int); JSON does not. jig/grammar.py checks bool
        # first for the same reason, and a disagreement here would emit a grammar that
        # accepts 1 where the pack means true.
        self.assertEqual(field_of([True, False, True, False]).type, "boolean")
        self.assertEqual(field_of([1, 0, 1, 0]).type, "integer")

    def test_an_int_here_and_a_float_there_is_a_number(self):
        self.assertEqual(field_of([1, 2.5, 3]).type, "number")
        self.assertEqual(field_of([2.5, 1]).type, "number")

    def test_all_ints_stay_integer(self):
        self.assertEqual(field_of([1, 2, 3]).type, "integer")

    def test_strings_arrays_and_objects(self):
        self.assertEqual(field_of(["a", "b"]).type, "string")
        self.assertEqual(field_of([[1], []]).type, "array")
        self.assertEqual(field_of([{"a": 1}, {}]).type, "object")

    def test_a_bool_mixed_with_an_int_is_not_widened(self):
        # `number` covers int and float. Nothing covers bool and int, so this must fail
        # loudly rather than pick one.
        with self.assertRaises(BuildError) as caught:
            field_of([True, 1])
        self.assertIn("'f'", str(caught.exception))

    def test_the_conflict_names_the_type_the_case_it_cites_really_held(self):
        # [1, 2.5, "s"] widens to `number` at case 2 before the string in case 3 breaks
        # it. The author is looking for a line in a file, and case 1 holds `1` — calling
        # it a number sends them to the wrong line.
        with self.assertRaises(BuildError) as caught:
            field_of([1, 2.5, "s"])
        message = str(caught.exception)
        self.assertIn("integer in case 1", message)
        self.assertIn("string in case 3", message)

    def test_a_string_in_one_case_and_an_object_in_another_names_both_cases(self):
        with self.assertRaises(BuildError) as caught:
            field_of(["a", "b", {"a": 1}])
        message = str(caught.exception)
        self.assertIn("'f'", message)
        self.assertIn("string", message)
        self.assertIn("object", message)
        self.assertIn("case 3", message)      # the offender, 1-based like the file
        self.assertIn("case 1", message)

    def test_a_value_that_is_not_json_is_refused(self):
        with self.assertRaises(BuildError) as caught:
            analyze("d", [case({"f": {1, 2}})], "p")
        self.assertIn("set", str(caught.exception))

    def test_a_non_json_value_nested_in_a_list_or_dict_is_refused_too(self):
        # `expect` is checked all the way down, not just at the top level: `examples`
        # json.dumps whole values for the prompt writer, so a set buried in a list would
        # otherwise reach the author as a bare TypeError from the standard library with
        # neither the field nor the case in it.
        for value in ([{1, 2}], {"items": [{"tag": {1, 2}}]}, [[{1, 2}]]):
            with self.assertRaises(BuildError) as caught:
                analyze("d", [case({"f": value})], "p")
            self.assertIn("set", str(caught.exception))

    def test_an_object_key_that_is_not_a_string_is_refused(self):
        with self.assertRaises(BuildError) as caught:
            analyze("d", [case({"f": {(1, 2): "x"}})], "p")
        message = str(caught.exception)
        self.assertIn("tuple", message)
        self.assertIn("key", message)

    def test_a_null_inside_a_list_or_object_is_ordinary_json(self):
        # meeting_actions puts action items straight into `expect`, and an item whose
        # owner is unknown is null. Only a null *field* is an absent observation.
        spec = field_of([[{"owner": None}], [], [None]])
        self.assertEqual(spec.type, "array")


# --------------------------------------------------------------------- the enum rule


class TestEnumIsInferred(unittest.TestCase):
    def test_a_small_closed_set_of_repeated_labels(self):
        spec = field_of(["billing", "technical", "billing", "technical"])
        self.assertEqual(spec.enum, ["billing", "technical"])

    def test_the_enum_is_sorted_and_holds_every_observed_value(self):
        # Sorted for a reviewable diff; complete because a gold value the grammar cannot
        # express is a case the compiled pack can never pass.
        spec = field_of(["p2", "p0", "p1", "p0", "p2", "p1"])
        self.assertEqual(spec.enum, ["p0", "p1", "p2"])

    def test_eleven_of_twelve_share_four_values_and_the_twelfth_is_unique(self):
        # The case the rule exists to decide. Eleven observations recur, one does not:
        # 11/12 is a majority, so the field is a vocabulary with a rare member — and the
        # rare member is *in* the enum, not dropped from it.
        values = ["a", "a", "a", "b", "b", "b", "c", "c", "d", "d", "d", "rare"]
        self.assertEqual(field_of(values).enum, ["a", "b", "c", "d", "rare"])

    def test_a_field_that_is_none_half_the_time_and_a_short_label_otherwise(self):
        # invoice_extract.review_reason: "none" six times, six distinct reasons once
        # each. Exactly half the mass recurs, which is enough.
        values = ["none"] * 6 + [
            "past_due", "totals_mismatch", "tax_out_of_range",
            "missing_identifier", "unusual_document", "unexpected_currency",
        ]
        self.assertEqual(len(field_of(values).enum), 7)

    def test_the_majority_test_is_at_exactly_one_half(self):
        two_of_four = ["a", "a", "b", "c"]                    # 2/4 recurs -> enum
        two_of_five = ["a", "a", "b", "c", "d"]               # 2/5 recurs -> not
        self.assertEqual(field_of(two_of_four).enum, ["a", "b", "c"])
        self.assertIsNone(field_of(two_of_five).enum)

    def test_twelve_distinct_values_is_the_ceiling(self):
        at_the_line = ["v%02d" % n for n in range(MAX_ENUM_VALUES)] * 2
        over_it = ["v%02d" % n for n in range(MAX_ENUM_VALUES + 1)] * 2
        self.assertEqual(len(field_of(at_the_line).enum), MAX_ENUM_VALUES)
        self.assertIsNone(field_of(over_it).enum)

    def test_four_observations_is_the_floor(self):
        self.assertIsNone(field_of(["a", "a", "b"]).enum)
        self.assertEqual(field_of(["a", "a", "b", "b"]).enum, ["a", "b"])

    def test_nulls_do_not_count_toward_the_four(self):
        # Three real observations and two nulls is still three observations.
        self.assertIsNone(field_of(["a", "a", "b", None, None]).enum)


class TestEnumIsWithheld(unittest.TestCase):
    def test_free_text_never_repeats_and_never_enums(self):
        suppliers = ["ACME SUPPLIES INC.", "Nordwind GmbH", "Thistle Print Ltd",
                     "Blue Harbor Freight", "Kestrel Systems", "Ridgeline Tools"]
        self.assertIsNone(field_of(suppliers).enum)

    def test_one_repeated_value_is_a_habit_not_a_vocabulary(self):
        # meeting_actions.lead: "Maya" and otherwise null. An enum of ["Maya"] would
        # weld one gold set's person into the grammar of every future run.
        spec = field_of(["Maya", "Maya", "Maya", None, None, None, None, None])
        self.assertIsNone(spec.enum)
        self.assertTrue(spec.optional)

    def test_prose_that_happens_to_repeat_is_still_prose(self):
        # Half the mass recurs, so only the length guard stands between this field and
        # a grammar pinned to two sentences.
        sentence = "The reviewer could not reach a conclusion from the thread alone."
        values = [sentence, sentence, "b", "c"]
        self.assertGreater(len(sentence), MAX_LABEL_LENGTH)
        self.assertIsNone(field_of(values).enum)

    def test_the_label_length_boundary(self):
        at_the_line = "x" * MAX_LABEL_LENGTH
        over_it = "x" * (MAX_LABEL_LENGTH + 1)
        self.assertEqual(field_of([at_the_line, at_the_line, "b", "b"]).enum,
                         sorted([at_the_line, "b"]))
        self.assertIsNone(field_of([over_it, over_it, "b", "b"]).enum)

    def test_a_multiline_or_padded_value_is_not_label_shaped(self):
        self.assertIsNone(field_of(["a\nb", "a\nb", "c", "c"]).enum)
        self.assertIsNone(field_of([" a", " a", "c", "c"]).enum)

    def test_integers_are_counted_not_enumerated(self):
        # meeting_actions.action_count is 0, 2, 3 or 4 across twelve cases. Enumerating
        # it would make a fifth action item unrepresentable.
        spec = field_of([2, 2, 2, 2, 0, 0, 3, 3, 4, 2, 2, 1])
        self.assertEqual(spec.type, "integer")
        self.assertIsNone(spec.enum)

    def test_booleans_are_already_closed(self):
        self.assertIsNone(field_of([True, False, True, False]).enum)

    def test_arrays_and_objects_are_never_enumerated(self):
        self.assertIsNone(field_of([[], [], ["a"], ["a"]]).enum)
        self.assertIsNone(field_of([{}, {}, {"a": 1}, {"a": 1}]).enum)


# ------------------------------------------------ rule 6: one habit is not a vocabulary


class TestOneRecurringValueIsNotEnough(unittest.TestCase):
    """The hole rule 3 only half covered, and the transcription test that closes it.

    Rule 5 is satisfied by a single dominant value: seven of twelve on `"maya"` is a
    majority even though every other observation is a name seen once. Under constrained
    decoding an enum of those six names makes a seventh approver *unrepresentable*, which
    is not recoverable at run time and not visible until it bites.
    """

    def test_one_dominant_value_and_a_scatter_of_names_is_refused(self):
        names = ["maya"] * 7 + ["tomas", "priya", "jonas", "nadia", "dev"]
        cases = quoting_cases("f", names, quoted=12)
        self.assertIsNone(analyze("d", cases, "p").field_named("f").enum)

    def test_the_same_counts_are_still_an_enum_when_nothing_was_copied(self):
        # invoice_extract.currency: "USD" nine times, three others once each, and the
        # invoice text says "$" as often as it says the code. The counts alone cannot
        # tell this from the approver above, so the input has to break the tie.
        codes = ["USD"] * 9 + ["EUR", "GBP", "PKR"]
        cases = quoting_cases("f", codes, quoted=0)
        self.assertEqual(analyze("d", cases, "p").field_named("f").enum,
                         ["EUR", "GBP", "PKR", "USD"])

    def test_two_values_coming_back_ends_it_however_the_input_reads(self):
        # A vocabulary being *used* needs no tiebreak: `category` on an expense form is
        # printed on the receipt and is still a closed set of three.
        values = ["travel"] * 4 + ["software"] * 4 + ["meals"] * 4
        cases = quoting_cases("f", values, quoted=12)
        self.assertEqual(analyze("d", cases, "p").field_named("f").enum,
                         ["meals", "software", "travel"])

    def test_the_transcription_boundary_in_both_directions(self):
        # One recurring value either side of three quarters. Eight of twelve copied is
        # under the line and stays an enum; nine is on it and is refused.
        values = ["dom"] * 7 + ["s1", "s2", "s3", "s4", "s5"]
        under = quoting_cases("f", values, quoted=8)
        on_the_line = quoting_cases("f", values, quoted=9)
        self.assertEqual(len(analyze("d", under, "p").field_named("f").enum), 6)
        self.assertIsNone(analyze("d", on_the_line, "p").field_named("f").enum)

    def test_the_recurrence_boundary_is_two_distinct_values(self):
        # Everything copied, so only the count of *recurring* values decides. One
        # recurring value is refused; two is a vocabulary.
        one = ["a", "a", "b", "c", "d", "e"]
        two = ["a", "a", "b", "b", "c", "d"]
        self.assertIsNone(
            analyze("d", quoting_cases("f", one, quoted=6), "p").field_named("f").enum)
        self.assertEqual(
            analyze("d", quoting_cases("f", two, quoted=6), "p").field_named("f").enum,
            ["a", "b", "c", "d"])

    def test_a_value_is_transcribed_however_the_input_capitalises_it(self):
        # "maya" in the expect and "Maya Chen" in the input are one transcription.
        cases = [{"input": {"text": "Manager: %s Chen" % value.title()},
                  "expect": {"f": value}}
                 for value in ["maya"] * 7 + ["tomas", "priya", "jonas", "nadia", "dev"]]
        self.assertIsNone(analyze("d", cases, "p").field_named("f").enum)

    def test_the_transcription_test_reads_every_input_key(self):
        # The value can be copied out of any part of the input, so the whole object is
        # searched rather than one key the stage would have to guess.
        values = ["maya"] * 7 + ["tomas", "priya", "jonas", "nadia", "dev"]
        cases = [{"input": {"ticket": "no name here", "meta": {"manager": value}},
                  "expect": {"f": value}}
                 for value in values]
        self.assertIsNone(analyze("d", cases, "p").field_named("f").enum)

    def test_an_input_that_is_not_json_does_not_break_the_test(self):
        # Only 'expect' is the compiler's contract. An input JSON cannot express is
        # stringified for the search, not raised on — analyze must still produce a spec.
        values = ["a", "a", "b", "c"]
        cases = [{"input": {"text": "x", "odd": {1, 2}}, "expect": {"f": value}}
                 for value in values]
        self.assertEqual(analyze("d", cases, "p").field_named("f").enum, ["a", "b", "c"])


# ---------------------------------------------------------------- optional, examples


class TestOptional(unittest.TestCase):
    def test_null_in_one_case_makes_it_optional(self):
        self.assertTrue(field_of(["A-1", "B-2", None, "C-3"]).optional)

    def test_absent_from_one_expect_makes_it_optional(self):
        cases = [case({"a": "x", "b": "y"}), case({"a": "x"})]
        spec = analyze("d", cases, "p")
        self.assertFalse(spec.field_named("a").optional)
        self.assertTrue(spec.field_named("b").optional)

    def test_present_and_non_null_everywhere_is_required(self):
        self.assertFalse(field_of(["a", "b"]).optional)

    def test_a_field_that_is_null_in_every_case_cannot_be_typed(self):
        with self.assertRaises(BuildError) as caught:
            field_of([None, None, None])
        message = str(caught.exception)
        self.assertIn("'f'", message)
        self.assertIn("null", message)

    def test_nulls_do_not_decide_the_type(self):
        spec = field_of([None, 49.99, None, 15.0])
        self.assertEqual(spec.type, "number")
        self.assertTrue(spec.optional)


class TestExamples(unittest.TestCase):
    def test_a_few_distinct_values_in_the_order_the_author_wrote_them(self):
        spec = field_of(["b", "a", "b", "c", "d", "e", "f"])
        self.assertEqual(spec.examples, ["b", "a", "c", "d"])

    def test_examples_are_verbatim_and_skip_nulls(self):
        spec = field_of([None, "A-1001", None, "B-2002"])
        self.assertEqual(spec.examples, ["A-1001", "B-2002"])

    def test_unhashable_values_are_still_de_duplicated(self):
        spec = field_of([{"a": 1}, {"a": 1}, {"b": 2}])
        self.assertEqual(spec.examples, [{"a": 1}, {"b": 2}])


# ------------------------------------------------------------------- what it refuses


class TestRefusals(unittest.TestCase):
    def test_no_cases_at_all(self):
        with self.assertRaises(BuildError) as caught:
            analyze("d", [], "p")
        self.assertIn("expect", str(caught.exception))

    def test_an_expect_that_is_not_an_object(self):
        cases = [case({"a": "x"}), {"input": {"text": "y"}, "expect": ["a"]}]
        with self.assertRaises(BuildError) as caught:
            analyze("d", cases, "p")
        message = str(caught.exception)
        self.assertIn("case 2", message)
        self.assertIn("'expect'", message)

    def test_a_case_missing_expect_entirely(self):
        with self.assertRaises(BuildError) as caught:
            analyze("d", [{"input": {"text": "y"}}], "p")
        self.assertIn("'expect'", str(caught.exception))

    def test_a_case_that_is_not_an_object(self):
        with self.assertRaises(BuildError) as caught:
            analyze("d", ["just a string"], "p")
        self.assertIn("case 1", str(caught.exception))

    def test_input_keys_that_differ_between_cases(self):
        cases = [
            {"name": "first", "input": {"a": 1, "b": 2}, "expect": {"f": "x"}},
            {"name": "second", "input": {"a": 1}, "expect": {"f": "y"}},
        ]
        with self.assertRaises(BuildError) as caught:
            analyze("d", cases, "p")
        message = str(caught.exception)
        self.assertIn("case 2", message)
        self.assertIn("'second'", message)     # named, so the author can find the line
        self.assertIn("'b'", message)

    def test_an_extra_input_key_is_also_a_difference(self):
        cases = [
            {"input": {"a": 1}, "expect": {"f": "x"}},
            {"input": {"a": 1, "z": 9}, "expect": {"f": "y"}},
        ]
        with self.assertRaises(BuildError) as caught:
            analyze("d", cases, "p")
        self.assertIn("'z'", str(caught.exception))

    def test_every_expect_empty(self):
        with self.assertRaises(BuildError) as caught:
            analyze("d", [case({}), case({})], "p")
        self.assertIn("empty", str(caught.exception))


# ------------------------------------------------------------------------ the shape


class TestTheSpecItCarries(unittest.TestCase):
    def test_it_carries_the_name_description_and_cases_verbatim(self):
        cases = [case({"f": "a"}), case({"f": "b"})]
        original = copy.deepcopy(cases)
        spec = analyze("triage a ticket", cases, "support_triage")
        self.assertEqual(spec.name, "support_triage")
        self.assertEqual(spec.description, "triage a ticket")
        self.assertEqual(spec.cases, original)
        self.assertEqual(cases, original)      # the gold set is never edited

    def test_input_and_field_order_follow_the_author(self):
        cases = [
            {"input": {"ticket": "t", "locale": "en"}, "expect": {"queue": "q"}},
            {"input": {"locale": "en", "ticket": "t"}, "expect": {"category": "c"}},
        ]
        spec = analyze("d", cases, "p")
        self.assertEqual(spec.inputs, ["ticket", "locale"])
        self.assertEqual([f.name for f in spec.fields], ["queue", "category"])

    def test_the_field_spec_renders_the_grammar_fragment(self):
        spec = field_of(["billing", "billing", "other", "other"])
        self.assertEqual(spec.schema,
                         {"type": "string", "enum": ["billing", "other"]})

    def test_it_is_deterministic(self):
        cases = cases_for("f", ["a", "b", "a", "c", None])
        self.assertEqual(analyze("d", cases, "p"), analyze("d", cases, "p"))

    def test_the_stage_never_imports_the_runtime(self):
        # The compiler does not ship. Nothing here may pull in the walker, the pack
        # loader or a backend, or the separation stops being real.
        import jig.build.analyze as module
        source = open(module.__file__).read()
        for banned in ("jig.graph", "jig.pack", "jig.eval", "jig.backends",
                       "from ..", "import jig"):
            self.assertNotIn(banned, source)


# ----------------------------------------------------------- against the real packs


def shipped_view(pack):
    """What a pack's own grammars say about each field it writes.

    A field can be written by several nodes, each narrowing it differently: in
    `content_moderation`, `decide` may emit any of three decisions and `force_review`
    exactly one. The task-level truth is the union — the set of values the pack as a
    whole can produce. A field is enum-constrained at the task level only if *every*
    node that writes it constrains it, because one unconstrained writer is enough to
    let any string through (`invoice_extract.needs_review` is a bare boolean in
    `review` and `enum: [true]` in the flag nodes).
    """
    view = {}
    for path in sorted(glob.glob(os.path.join(pack, "grammars", "*.json"))):
        with open(path) as handle:
            schema = json.load(handle)
        for name, sub in (schema.get("properties") or {}).items():
            declared = sub.get("type")
            declared = [declared] if isinstance(declared, str) else list(declared or [])
            entry = view.setdefault(name, {"types": set(), "null": False, "enums": []})
            entry["null"] = entry["null"] or "null" in declared
            entry["types"].update(set(declared) - {"null"})
            entry["enums"].append(sub.get("enum"))
    for entry in view.values():
        parts = entry.pop("enums")
        entry["enum"] = (
            sorted({json.dumps(v, sort_keys=True) for part in parts for v in part})
            if all(part is not None for part in parts) else None
        )
    return view


def load_cases(pack):
    with open(os.path.join(pack, "evalset.jsonl")) as handle:
        return [json.loads(line) for line in handle if line.strip()]


# The two fields across all six packs where the rule is shy: the pack constrains them
# and the gold set does not carry enough evidence to say so. `defect` appears in three
# of thirteen cases with three distinct values; `industry` takes seven values in ten
# cases and only "other" ever recurs, which is 4/10 — just under the majority line.
# Listed rather than tolerated: a new name appearing here is a regression to explain.
KNOWN_SHY = {("incident_triage", "defect"), ("lead_qualify", "industry")}


def compiles_to_generate_nodes(pack):
    """Whether `jig build` could have produced this pack — i.e. it has no tool nodes.

    `build/assemble.py` emits `type: generate` and `type: end` and nothing else, so a
    pack with a `type: tool` node is outside what stage 1 is measured against. It is also
    outside what this sweep can check: `analyze` induces a field from every `expect` key,
    and a tool node's fields are declared by the *host's* registry (`Tool.writes`), not by
    a grammar file in the pack — so `shipped_view` cannot see them and would report a
    field no grammar declares. `examples/refund_desk` is the pack this excludes.
    """
    with open(os.path.join(pack, "graph.yaml")) as handle:
        return "type: tool" not in handle.read()


class TestAgainstTheShippedPacks(unittest.TestCase):
    """Induce a schema from each pack's evalset; hold it against that pack's grammars."""

    def setUp(self):
        self.packs = sorted(
            path for path in glob.glob(os.path.join(EXAMPLES, "*"))
            if os.path.isfile(os.path.join(path, "evalset.jsonl"))
            and compiles_to_generate_nodes(path)
        )
        self.assertEqual(len(self.packs), 6)

    def test_every_induced_type_is_the_type_the_pack_declares(self):
        checked = 0
        for pack in self.packs:
            view = shipped_view(pack)
            spec = analyze("", load_cases(pack), os.path.basename(pack))
            for field in spec.fields:
                self.assertIn(field.name, view,
                              "%s: no grammar declares %r" % (pack, field.name))
                self.assertIn(field.type, view[field.name]["types"],
                              "%s.%s: induced %s" % (pack, field.name, field.type))
                checked += 1
        self.assertEqual(checked, 54)   # every expect key of every pack

    def test_a_nullable_field_is_always_induced_as_optional(self):
        for pack in self.packs:
            view = shipped_view(pack)
            for field in analyze("", load_cases(pack), os.path.basename(pack)).fields:
                if view[field.name]["null"]:
                    self.assertTrue(field.optional,
                                    "%s.%s" % (pack, field.name))

    def test_no_enum_is_ever_invented_or_widened(self):
        """The failure that cannot be recovered downstream: never once, on six packs.

        An induced enum must be a subset of what the pack can actually emit. A value the
        pack ships and the compiler omits costs accuracy; a value the compiler invents
        makes an output the pack needs unrepresentable, and no retry reaches that.
        """
        for pack in self.packs:
            view = shipped_view(pack)
            for field in analyze("", load_cases(pack), os.path.basename(pack)).fields:
                if field.enum is None:
                    continue
                theirs = view[field.name]["enum"]
                self.assertIsNotNone(
                    theirs, "%s.%s: enum invented for an open field" % (pack, field.name))
                mine = {json.dumps(v, sort_keys=True) for v in field.enum}
                self.assertLessEqual(
                    mine, set(theirs),
                    "%s.%s: induced values the pack cannot emit" % (pack, field.name))

    def test_the_only_enums_it_misses_are_the_two_documented_ones(self):
        missed = set()
        exact = 0
        for pack in self.packs:
            name = os.path.basename(pack)
            view = shipped_view(pack)
            for field in analyze("", load_cases(pack), name).fields:
                theirs = view[field.name]["enum"]
                if theirs is None:
                    continue
                if field.enum is None:
                    missed.add((name, field.name))
                elif {json.dumps(v, sort_keys=True) for v in field.enum} == set(theirs):
                    exact += 1
        self.assertLessEqual(missed, KNOWN_SHY)
        # 24 of the 32 enum-constrained fields come back byte-identical; the rest are
        # strict subsets, because those packs' gold sets never exercise every value.
        self.assertGreaterEqual(exact, 24)

    def test_support_triage_comes_back_exactly_as_it_ships(self):
        """The pack the whole repo is measured on, field by field."""
        spec = analyze("Triage a support ticket.",
                       load_cases(os.path.join(EXAMPLES, "support_triage")),
                       "support_triage")
        self.assertEqual(spec.inputs, ["ticket"])
        self.assertEqual(
            [(f.name, f.type, f.enum, f.optional) for f in spec.fields],
            [
                ("category", "string",
                 ["account", "billing", "other", "technical"], False),
                ("order_id", "string", None, True),
                ("amount_usd", "number", None, True),
                ("sentiment", "string", ["angry", "calm", "frustrated"], False),
                ("priority", "string", ["p0", "p1", "p2", "p3"], False),
                ("queue", "string",
                 ["billing-ops", "eng-support", "general", "identity"], False),
                ("escalate", "boolean", None, False),
            ],
        )

    def test_the_thresholds_are_the_ones_the_docstring_states(self):
        self.assertEqual((MIN_ENUM_OBSERVATIONS, MAX_ENUM_VALUES, MAX_LABEL_LENGTH),
                         (4, 12, 40))
        self.assertEqual((MIN_TIMES_SEEN, MIN_RECURRING_VALUES, TRANSCRIBED_QUARTERS),
                         (2, 2, 3))


# ------------------------------------------------------------- and against a seventh


# Twelve expense claims, written to test generalisation rather than to be compiled: no
# pack in `examples/` has a short-token field the author left open, so until this one
# existed the length guard of rule 4 was doing all the work and the majority rule of
# rule 5 had never actually been exercised. Every claim prints the manager who signs it
# and the cost center it lands on, because that is the only way a reader — or a
# model — could know either. `maya` signs seven of the twelve and five other managers
# sign one each; `CC-1041` carries the same shape.
EXPENSE_ROWS = [
    ("Rui Alves", "Maya Chen", "CC-1041", "Platform", "travel",
     "Flight, Lisbon to Berlin", 612.40, "attached", "approved"),
    ("Rui Alves", "Maya Chen", "CC-1041", "Platform", "software",
     "Annual IDE licence, one seat", 289.00, "attached", "approved"),
    ("Ana Ruiz", "Tomas Beck", "CC-2037", "Field Sales", "meals",
     "Client dinner, four covers", 214.80, "missing", "needs_receipt"),
    ("Rui Alves", "Maya Chen", "CC-1041", "Platform", "travel",
     "Business class upgrade, no waiver on file", 1840.00, "attached", "rejected"),
    ("Sofia Marek", "Priya Nair", "CC-3312", "Data", "software",
     "Warehouse seat, one month", 450.00, "attached", "approved"),
    ("Rui Alves", "Maya Chen", "CC-1041", "Platform", "meals",
     "Team lunch, six covers", 132.10, "missing", "needs_receipt"),
    ("Lars Vogt", "Jonas Weber", "CC-4408", "Support", "travel",
     "Train, Munich to Vienna", 96.50, "attached", "approved"),
    ("Rui Alves", "Maya Chen", "CC-1041", "Platform", "software",
     "Monitoring add-on, one seat", 78.00, "attached", "approved"),
    ("Hana Ito", "Nadia Salem", "CC-5150", "Design", "meals",
     "Workshop catering", 305.25, "attached", "approved"),
    ("Rui Alves", "Maya Chen", "CC-1041", "Platform", "travel",
     "Hotel, seven nights, over the nightly cap", 2310.00, "attached", "rejected"),
    ("Omar Haddad", "Dev Shastri", "CC-6021", "Security", "software",
     "Penetration testing toolkit", 999.00, "attached", "approved"),
    ("Rui Alves", "Maya Chen", "CC-1041", "Platform", "meals",
     "Offsite breakfast", 64.00, "missing", "needs_receipt"),
]


def expense_cases():
    """The seventh pack's evalset, in the shape `analyze` is given."""
    cases = []
    for index, row in enumerate(EXPENSE_ROWS):
        submitter, manager, code, team, category, line, amount, receipt, decision = row
        cases.append({
            "name": "%s, %s" % (category, line.split(",")[0].lower()),
            "input": {"claim_text": "\n".join([
                "Expense EXP-41%02d" % (index + 1),
                "Submitted by: %s" % submitter,
                "Manager: %s" % manager,
                "Cost center: %s (%s)" % (code, team),
                "Category: %s" % category.title(),
                "%s ... %.2f USD" % (line, amount),
                "Receipt: %s" % receipt,
            ])},
            "expect": {
                "approver": manager.split()[0].lower(),
                "cost_center": code,
                "category": category,
                "amount_usd": amount,
                "decision": decision,
            },
        })
    return cases


class TestTheExpenseApprovalPack(unittest.TestCase):
    """A pack the stage was never developed against, and the two fields it welded shut.

    An invented enum is the one failure downstream cannot repair: under constrained
    decoding a value outside the enum is unrepresentable, not merely unlikely, so a pack
    compiled with `approver` pinned to these six names can never name a seventh —
    however plainly the claim in front of it says so, however good the model, however
    many retries. It is also invisible until it bites, which is why the shape is nailed
    down here rather than left to the next corpus.
    """

    def setUp(self):
        self.cases = expense_cases()
        self.spec = analyze("Approve or reject an expense claim.",
                            self.cases, "expense_approval")

    def enum_of(self, name):
        return self.spec.field_named(name).enum

    def counts_of(self, name):
        counts = {}
        for case in self.cases:
            value = case["expect"][name]
            counts[value] = counts.get(value, 0) + 1
        return counts

    def test_the_fixture_is_the_shape_the_rule_has_to_decide(self):
        # Guard the gold set itself: if a later edit softens these counts the two tests
        # below would start passing for the wrong reason.
        for name in ("approver", "cost_center"):
            counts = self.counts_of(name)
            self.assertEqual(len(counts), 6)
            self.assertEqual(sorted(counts.values()), [1, 1, 1, 1, 1, 7])
        self.assertEqual(len(self.cases), 12)

    def test_rules_one_to_five_would_have_welded_both_shut(self):
        # Why this pack is here at all. Both fields are strings, both have twelve
        # observations and six short one-line values, and in both the majority lands on
        # a value that recurs — seven of twelve, all of it on one habit. Every test the
        # rule made before rule 6 passes.
        for name in ("approver", "cost_center"):
            counts = self.counts_of(name)
            observations = sum(counts.values())
            recurring = sum(n for n in counts.values() if n >= MIN_TIMES_SEEN)
            self.assertGreaterEqual(observations, MIN_ENUM_OBSERVATIONS)
            self.assertLessEqual(len(counts), MAX_ENUM_VALUES)
            self.assertTrue(all(len(v) <= MAX_LABEL_LENGTH for v in counts))
            self.assertGreaterEqual(recurring * 2, observations)
            self.assertLess(len([n for n in counts.values() if n >= MIN_TIMES_SEEN]),
                            MIN_RECURRING_VALUES)

    def test_a_roster_of_managers_is_not_a_vocabulary(self):
        self.assertIsNone(self.enum_of("approver"))

    def test_a_column_of_ledger_codes_is_not_a_vocabulary(self):
        self.assertIsNone(self.enum_of("cost_center"))

    def test_the_pack_keeps_the_enums_it_really_has(self):
        # The refusal has to be surgical. `category` is printed on every claim and would
        # fail the transcription test outright, but three of its values come back, so
        # rule 6 never asks. `decision` is nowhere in the input and is a vocabulary by
        # the counts alone.
        self.assertEqual(self.enum_of("category"), ["meals", "software", "travel"])
        self.assertEqual(self.enum_of("decision"),
                         ["approved", "needs_receipt", "rejected"])

    def test_the_rest_of_the_spec_is_unaffected(self):
        self.assertEqual([(f.name, f.type) for f in self.spec.fields],
                         [("approver", "string"), ("cost_center", "string"),
                          ("category", "string"), ("amount_usd", "number"),
                          ("decision", "string")])
        self.assertEqual(self.spec.inputs, ["claim_text"])
        self.assertFalse(any(f.optional for f in self.spec.fields))


if __name__ == "__main__":
    unittest.main()
