"""T4 — `{var}` prompt rendering against run state."""

import unittest

from jig.errors import MissingVariable
from jig.render import render


class TestRender(unittest.TestCase):
    def test_substitutes_a_plain_variable(self):
        self.assertEqual(render("Ticket: {ticket}", {"ticket": "help"}), "Ticket: help")

    def test_substitutes_several(self):
        self.assertEqual(
            render("{a}/{b}", {"a": "x", "b": "y", "unused": 1}), "x/y"
        )

    def test_dotted_paths_reach_into_nested_state(self):
        state = {"classification": {"category": "billing"}}
        self.assertEqual(render("{classification.category}", state), "billing")

    def test_non_strings_render_as_json(self):
        self.assertEqual(render("{n}", {"n": 3}), "3")
        self.assertEqual(render("{f}", {"f": 1.5}), "1.5")
        self.assertEqual(render("{b}", {"b": True}), "true")
        self.assertEqual(render("{z}", {"z": None}), "null")
        self.assertEqual(render("{d}", {"d": {"k": 1}}), '{"k": 1}')
        self.assertEqual(render("{l}", {"l": ["a", "b"]}), '["a", "b"]')

    def test_doubled_braces_are_literal_so_prompts_can_show_json(self):
        rendered = render('Reply like {{"category": "billing"}} for {t}', {"t": "x"})
        self.assertEqual(rendered, 'Reply like {"category": "billing"} for x')

    def test_a_template_with_no_variables_is_unchanged(self):
        self.assertEqual(render("no vars here", {}), "no vars here")

    def test_missing_variable_names_itself_and_lists_what_exists(self):
        with self.assertRaises(MissingVariable) as caught:
            render("{ticket}", {"other": 1})
        message = str(caught.exception)
        self.assertIn("ticket", message)
        self.assertIn("other", message)

    def test_missing_dotted_leaf_names_the_full_path(self):
        with self.assertRaises(MissingVariable) as caught:
            render("{a.b}", {"a": {"c": 1}})
        self.assertIn("a.b", str(caught.exception))

    def test_dotted_path_through_a_non_mapping_is_an_error(self):
        with self.assertRaises(MissingVariable):
            render("{a.b}", {"a": "string"})

    def test_unclosed_brace_is_left_alone(self):
        self.assertEqual(render("100% of {a", {"a": 1}), "100% of {a")
