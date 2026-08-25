"""T4 — `{var}` prompt rendering against run state."""

import unittest

from stepmold.errors import MissingVariable
from stepmold.render import render


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


class TestSubstitutionIsSinglePass(unittest.TestCase):
    """A substituted value is text, never template.

    Run state holds every value the workflow has seen, so a second pass would let a
    ticket reading `{card_number}` print another state key into the prompt. These fail
    under `str.format`, under a `while "{" in text` loop, and under any two-pass render.
    """

    def test_a_value_that_looks_like_a_placeholder_stays_literal(self):
        state = {"ticket": "{card}", "card": "4111-1111-1111-1111"}
        self.assertEqual(render("T: {ticket}", state), "T: {card}")

    def test_a_value_cannot_use_the_templates_own_escape(self):
        state = {"ticket": "{{card}}", "card": "4111-1111-1111-1111"}
        self.assertEqual(render("T: {ticket}", state), "T: {{card}}")

    def test_a_json_rendered_value_does_not_re_expand_its_own_braces(self):
        """`as_text` writes an object as JSON — a value made mostly of braces.

        No hostile ticket is needed for this one: any object-valued state key carries
        braces into the prompt, and a second pass would resolve whatever is inside them.
        """
        state = {"payload": {"note": "{card}"}, "card": "4111-1111-1111-1111"}
        rendered = render("P: {payload}", state)
        self.assertEqual(rendered, 'P: {"note": "{card}"}')
        self.assertNotIn("4111", rendered)

    def test_a_self_referential_value_does_not_recurse(self):
        self.assertEqual(render("T: {ticket}", {"ticket": "{ticket}"}), "T: {ticket}")


class TestTheMissingVariableMessage(unittest.TestCase):
    """The message is built from caller-supplied key names, so it is escaped and clipped."""

    def test_the_key_list_is_repr_d_so_an_invisible_key_is_visible(self):
        with self.assertRaises(MissingVariable) as caught:
            render("{ticket}", {"tick\u200bet": 1})
        self.assertIn("\\u200b", str(caught.exception))

    def test_a_giant_key_does_not_become_a_giant_message(self):
        with self.assertRaises(MissingVariable) as caught:
            render("{ticket}", {"z" * 100000: 1})
        self.assertLess(len(str(caught.exception)), 300)

    def test_an_empty_state_still_says_so(self):
        with self.assertRaises(MissingVariable) as caught:
            render("{ticket}", {})
        self.assertIn("nothing", str(caught.exception))
