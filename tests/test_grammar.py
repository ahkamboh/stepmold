"""T3 — the stdlib JSON Schema subset that every node output is checked against."""

import unittest

from jig.grammar import (
    SchemaError,
    ValidationError,
    schema_to_grammar,
    validate_against,
)

TICKET = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["billing", "technical", "other"]},
        "priority": {"type": "integer"},
        "confident": {"type": "boolean"},
        "score": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "note": {"type": ["string", "null"]},
    },
    "required": ["category", "priority"],
    "additionalProperties": False,
}


class TestSchemaToGrammar(unittest.TestCase):
    def test_wraps_the_schema_in_a_backend_neutral_struct(self):
        grammar = schema_to_grammar(TICKET)
        self.assertEqual(grammar["kind"], "json_schema")
        self.assertEqual(grammar["schema"], TICKET)

    def test_does_not_alias_the_caller_s_schema(self):
        schema = {"type": "object", "properties": {}}
        grammar = schema_to_grammar(schema)
        grammar["schema"]["properties"]["injected"] = {"type": "string"}
        self.assertEqual(schema["properties"], {})

    def test_rejects_a_schema_it_could_not_enforce(self):
        with self.assertRaises(SchemaError) as caught:
            schema_to_grammar({"type": "banana"})
        self.assertIn("banana", str(caught.exception))

    def test_rejects_a_non_mapping_schema(self):
        with self.assertRaises(SchemaError):
            schema_to_grammar(["not", "a", "schema"])

    def test_rejects_unknown_keywords_so_typos_are_loud(self):
        with self.assertRaises(SchemaError) as caught:
            schema_to_grammar({"type": "object", "requried": ["a"]})
        self.assertIn("requried", str(caught.exception))


class TestValidObjects(unittest.TestCase):
    def test_a_complete_object_passes(self):
        self.assertIsNone(
            validate_against(
                TICKET,
                {
                    "category": "billing",
                    "priority": 2,
                    "confident": True,
                    "score": 0.5,
                    "tags": ["a", "b"],
                    "note": None,
                },
            )
        )

    def test_optional_properties_may_be_absent(self):
        self.assertIsNone(
            validate_against(TICKET, {"category": "other", "priority": 1})
        )

    def test_an_integer_satisfies_number(self):
        self.assertIsNone(validate_against({"type": "number"}, 3))

    def test_a_bool_does_not_satisfy_integer(self):
        with self.assertRaises(ValidationError):
            validate_against({"type": "integer"}, True)

    def test_an_empty_schema_accepts_anything(self):
        self.assertIsNone(validate_against({}, {"whatever": [1, 2]}))


class TestViolations(unittest.TestCase):
    def _fails(self, schema, obj):
        with self.assertRaises(ValidationError) as caught:
            validate_against(schema, obj)
        return caught.exception

    def test_missing_required_field_names_the_field(self):
        error = self._fails(TICKET, {"category": "billing"})
        self.assertIn("priority", str(error))
        self.assertIn("required", str(error))

    def test_wrong_type_names_the_field_and_both_types(self):
        error = self._fails(TICKET, {"category": "billing", "priority": "high"})
        self.assertIn("priority", str(error))
        self.assertIn("integer", str(error))
        self.assertIn("str", str(error))

    def test_value_outside_enum_names_the_field_and_the_choices(self):
        error = self._fails(TICKET, {"category": "refund", "priority": 1})
        self.assertIn("category", str(error))
        self.assertIn("refund", str(error))
        self.assertIn("billing", str(error))

    def test_additional_property_names_the_offender(self):
        error = self._fails(
            TICKET, {"category": "billing", "priority": 1, "sentiment": "angry"}
        )
        self.assertIn("sentiment", str(error))

    def test_additional_properties_allowed_when_not_forbidden(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.assertIsNone(validate_against(schema, {"a": "x", "b": 2}))

    def test_top_level_type_mismatch(self):
        error = self._fails(TICKET, ["not", "an", "object"])
        self.assertIn("object", str(error))

    def test_array_item_violation_reports_the_index(self):
        error = self._fails(
            TICKET, {"category": "billing", "priority": 1, "tags": ["ok", 7]}
        )
        self.assertIn("tags[1]", str(error))

    def test_nested_object_violation_reports_the_path(self):
        schema = {
            "type": "object",
            "properties": {
                "ticket": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                }
            },
            "required": ["ticket"],
        }
        error = self._fails(schema, {"ticket": {}})
        self.assertIn("ticket.id", str(error))

    def test_union_type_is_enforced(self):
        error = self._fails(TICKET, {"category": "billing", "priority": 1, "note": 3})
        self.assertIn("note", str(error))

    def test_the_error_exposes_the_path_for_programmatic_use(self):
        error = self._fails(TICKET, {"category": "billing", "priority": "high"})
        self.assertEqual(error.path, "priority")

    def test_validation_error_is_a_value_error(self):
        self.assertTrue(issubclass(ValidationError, ValueError))
