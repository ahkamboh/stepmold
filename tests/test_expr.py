"""T4 — the tiny expression language `assert` nodes are written in."""

import unittest

from jig.errors import ExprError
from jig.expr import evaluate, is_true

STATE = {
    "category": "billing",
    "priority": 2,
    "tags": ["urgent", "paid"],
    "classification": {"category": "billing", "confidence": 0.9},
    "flag": True,
}


class TestEvaluate(unittest.TestCase):
    def test_reads_a_name_from_state(self):
        self.assertEqual(evaluate("category", STATE), "billing")

    def test_dotted_access_into_a_mapping(self):
        self.assertEqual(evaluate("classification.category", STATE), "billing")

    def test_comparisons(self):
        self.assertTrue(evaluate('category == "billing"', STATE))
        self.assertTrue(evaluate("priority > 1", STATE))
        self.assertTrue(evaluate("priority >= 2 and priority <= 3", STATE))
        self.assertFalse(evaluate('category != "billing"', STATE))

    def test_membership(self):
        self.assertTrue(evaluate('"urgent" in tags', STATE))
        self.assertTrue(evaluate('category in ["billing", "technical"]', STATE))
        self.assertTrue(evaluate('"refund" not in tags', STATE))

    def test_boolean_operators(self):
        self.assertTrue(evaluate("flag and priority == 2", STATE))
        self.assertTrue(evaluate("not (priority == 5)", STATE))
        self.assertTrue(evaluate("priority == 5 or flag", STATE))

    def test_arithmetic(self):
        self.assertEqual(evaluate("priority + 1", STATE), 3)
        self.assertEqual(evaluate("priority * 2 - 1", STATE), 3)

    def test_indexing(self):
        self.assertEqual(evaluate("tags[0]", STATE), "urgent")
        self.assertEqual(evaluate('classification["confidence"]', STATE), 0.9)

    def test_whitelisted_helpers(self):
        self.assertEqual(evaluate("len(tags)", STATE), 2)
        self.assertEqual(evaluate("lower(category)", STATE), "billing")
        self.assertTrue(evaluate('startswith(category, "bill")', STATE))
        self.assertEqual(evaluate("max(priority, 5)", STATE), 5)

    def test_literals(self):
        self.assertEqual(evaluate("true", {}), True)
        self.assertEqual(evaluate("false", {}), False)
        self.assertEqual(evaluate("null", {}), None)
        self.assertEqual(evaluate("True", {}), True)

    def test_is_true_coerces(self):
        self.assertTrue(is_true("tags", STATE))
        self.assertFalse(is_true("[]", STATE))
        self.assertTrue(is_true('category == "billing"', STATE))


class TestRefusals(unittest.TestCase):
    def _fails(self, expression, state=None):
        with self.assertRaises(ExprError) as caught:
            evaluate(expression, STATE if state is None else state)
        return str(caught.exception)

    def test_unknown_name_says_what_is_available(self):
        message = self._fails("nonexistent == 1")
        self.assertIn("nonexistent", message)
        self.assertIn("category", message)

    def test_missing_dotted_leaf(self):
        self.assertIn("classification.missing", self._fails("classification.missing"))

    def test_syntax_error_is_reported_not_raised_raw(self):
        self.assertIn("could not parse", self._fails("category =="))

    def test_function_calls_are_whitelisted(self):
        message = self._fails('__import__("os")')
        self.assertIn("__import__", message)

    def test_attribute_access_on_a_non_mapping_is_refused(self):
        self.assertIn("category.upper", self._fails("category.upper"))

    def test_assignment_and_statements_are_refused(self):
        self.assertIn("could not parse", self._fails("x = 1"))

    def test_lambda_is_refused(self):
        self.assertIn("Lambda", self._fails("lambda: 1"))

    def test_comprehensions_are_refused(self):
        self.assertIn("Comprehension", self._fails("[t for t in tags]"))

    def test_dunder_names_are_refused_even_if_present_in_state(self):
        self.assertIn("__class__", self._fails("__class__", {"__class__": 1}))

    def test_an_empty_expression_is_refused(self):
        self.assertIn("empty", self._fails("   "))
