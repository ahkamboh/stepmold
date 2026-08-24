"""Covers the stdlib YAML subset the pack format is written in (see NIGHT-LOG, T2)."""

import unittest

from jig.yamlish import YamlError, parse


class TestScalars(unittest.TestCase):
    def test_strings_numbers_bools_and_null(self):
        doc = parse(
            "name: support_triage\n"
            "version: 3\n"
            "ratio: 0.25\n"
            "enabled: true\n"
            "disabled: false\n"
            "missing: null\n"
            "tilde: ~\n"
            "blank:\n"
        )
        self.assertEqual(doc["name"], "support_triage")
        self.assertEqual(doc["version"], 3)
        self.assertEqual(doc["ratio"], 0.25)
        self.assertIs(doc["enabled"], True)
        self.assertIs(doc["disabled"], False)
        self.assertIsNone(doc["missing"])
        self.assertIsNone(doc["tilde"])
        self.assertIsNone(doc["blank"])

    def test_quoted_strings_keep_their_text(self):
        doc = parse("a: 'true'\nb: \"3\"\nc: 'it''s fine'\nd: \"line\\nbreak\"\n")
        self.assertEqual(doc["a"], "true")
        self.assertEqual(doc["b"], "3")
        self.assertEqual(doc["c"], "it's fine")
        self.assertEqual(doc["d"], "line\nbreak")

    def test_a_colon_inside_a_value_is_kept(self):
        doc = parse("model: fake:fakes/script.json\n")
        self.assertEqual(doc["model"], "fake:fakes/script.json")

    def test_negative_numbers(self):
        doc = parse("low: -2\nhigh: +5\nsci: 1e3\n")
        self.assertEqual(doc["low"], -2)
        self.assertEqual(doc["high"], 5)
        self.assertEqual(doc["sci"], 1000.0)


class TestStructure(unittest.TestCase):
    def test_nested_mappings(self):
        doc = parse(
            "nodes:\n"
            "  classify:\n"
            "    type: generate\n"
            "    two_stage: true\n"
            "  done:\n"
            "    type: end\n"
        )
        self.assertEqual(
            doc,
            {
                "nodes": {
                    "classify": {"type": "generate", "two_stage": True},
                    "done": {"type": "end"},
                }
            },
        )

    def test_sequence_of_scalars(self):
        self.assertEqual(parse("keys:\n  - a\n  - b\n"), {"keys": ["a", "b"]})

    def test_sequence_of_mappings(self):
        doc = parse(
            "edges:\n"
            "  - from: classify\n"
            "    to: extract\n"
            "  - from: extract\n"
            "    to: done\n"
        )
        self.assertEqual(
            doc["edges"],
            [
                {"from": "classify", "to": "extract"},
                {"from": "extract", "to": "done"},
            ],
        )

    def test_nested_sequence_inside_a_sequence_item(self):
        doc = parse("edges:\n  - to: a\n    tags:\n      - x\n      - y\n")
        self.assertEqual(doc["edges"], [{"to": "a", "tags": ["x", "y"]}])

    def test_top_level_sequence(self):
        self.assertEqual(parse("- one\n- two\n"), ["one", "two"])

    def test_comments_and_blank_lines_are_ignored(self):
        doc = parse("# leading\n\nname: jig   # trailing\n\n# trailing block\n")
        self.assertEqual(doc, {"name": "jig"})

    def test_hash_inside_a_quoted_string_is_not_a_comment(self):
        self.assertEqual(parse("t: 'a # b'\n"), {"t": "a # b"})

    def test_document_markers_are_skipped(self):
        self.assertEqual(parse("---\nname: jig\n...\n"), {"name": "jig"})

    def test_empty_document(self):
        self.assertIsNone(parse(""))
        self.assertIsNone(parse("# only a comment\n"))


class TestBlockScalars(unittest.TestCase):
    def test_literal_keeps_line_breaks(self):
        doc = parse("text: |\n  line one\n  line two\n")
        self.assertEqual(doc["text"], "line one\nline two\n")

    def test_literal_strip_drops_the_trailing_newline(self):
        doc = parse("text: |-\n  line one\n  line two\n")
        self.assertEqual(doc["text"], "line one\nline two")

    def test_folded_joins_lines_with_spaces(self):
        doc = parse("text: >-\n  a folded\n  description\n")
        self.assertEqual(doc["text"], "a folded description")

    def test_folded_keeps_a_blank_line_as_a_break(self):
        doc = parse("text: >-\n  first para\n\n  second para\n")
        self.assertEqual(doc["text"], "first para\nsecond para")

    def test_a_block_scalar_does_not_swallow_the_next_key(self):
        doc = parse("text: |-\n  body\nafter: 1\n")
        self.assertEqual(doc, {"text": "body", "after": 1})

    def test_a_block_scalar_keeps_hashes_and_colons_verbatim(self):
        doc = parse("text: |-\n  # not a comment\n  key: not a mapping\n")
        self.assertEqual(doc["text"], "# not a comment\nkey: not a mapping")

    def test_a_block_scalar_nested_in_a_mapping(self):
        doc = parse("node:\n  prompt: |-\n    hello\n    there\n  type: generate\n")
        self.assertEqual(doc["node"], {"prompt": "hello\nthere", "type": "generate"})

    def test_a_block_scalar_as_a_sequence_item(self):
        doc = parse("items:\n  - |-\n    one\n    two\n  - plain\n")
        self.assertEqual(doc["items"], ["one\ntwo", "plain"])

    def test_indentation_inside_a_literal_block_is_preserved(self):
        doc = parse("text: |-\n  outer\n    inner\n")
        self.assertEqual(doc["text"], "outer\n  inner")


class TestFlowCollections(unittest.TestCase):
    def test_flow_sequence(self):
        self.assertEqual(parse("k: [a, b, 3]\n"), {"k": ["a", "b", 3]})

    def test_flow_mapping(self):
        self.assertEqual(parse("when: {category: billing}\n"),
                         {"when": {"category": "billing"}})

    def test_empty_flow_collections(self):
        self.assertEqual(parse("a: []\nb: {}\n"), {"a": [], "b": {}})

    def test_nested_flow(self):
        self.assertEqual(parse("k: {a: [1, 2], b: {c: d}}\n"),
                         {"k": {"a": [1, 2], "b": {"c": "d"}}})

    def test_quoted_values_in_flow(self):
        self.assertEqual(parse("k: ['a, b', 'c']\n"), {"k": ["a, b", "c"]})


class TestErrors(unittest.TestCase):
    def _error(self, text):
        with self.assertRaises(YamlError) as caught:
            parse(text, filename="graph.yaml")
        return str(caught.exception)

    def test_missing_colon_reports_the_line(self):
        message = self._error("name: jig\njust a bare line\n")
        self.assertIn("graph.yaml:2", message)
        self.assertIn("key: value", message)

    def test_tabs_are_rejected(self):
        self.assertIn("tab", self._error("a:\n\tb: 1\n"))

    def test_bad_indentation_is_rejected(self):
        self.assertIn("indent", self._error("a: 1\n  b: 2\n"))

    def test_leading_indentation_is_rejected(self):
        self.assertIn("indent", self._error("  a: 1\n"))

    def test_duplicate_keys_are_rejected(self):
        self.assertIn("duplicate", self._error("a: 1\na: 2\n"))

    def test_keep_chomping_is_rejected_loudly(self):
        self.assertIn("keep chomping", self._error("a: |+\n  text\n\n"))

    def test_anchors_are_rejected_loudly(self):
        self.assertIn("anchor", self._error("a: &base\n"))

    def test_unterminated_flow_collection(self):
        self.assertIn("flow", self._error("a: [1, 2\n"))

    def test_unterminated_quote(self):
        self.assertIn("unterminated", self._error("a: 'oops\n"))


class TestBlockScalarRefusals(unittest.TestCase):
    """The gaps in `refuse rather than mis-parse` that block scalars used to have."""

    def _error(self, text):
        with self.assertRaises(YamlError) as caught:
            parse(text, filename="graph.yaml")
        return str(caught.exception)

    def test_a_block_scalar_under_a_sequence_key_keeps_its_key(self):
        doc = parse("edges:\n  - when: |-\n      hi\n      there\n    to: done\n")
        self.assertEqual(doc["edges"], [{"when": "hi\nthere", "to": "done"}])

    def test_a_block_scalar_under_a_sequence_key_stops_at_the_sibling_key(self):
        doc = parse("edges:\n  - when: |-\n      hi\n    to: done\n  - to: next\n")
        self.assertEqual(doc["edges"], [{"when": "hi", "to": "done"}, {"to": "next"}])

    def test_a_folded_scalar_under_a_sequence_key_keeps_its_key(self):
        doc = parse("edges:\n  - note: >-\n      a folded\n      note\n    to: done\n")
        self.assertEqual(doc["edges"], [{"note": "a folded note", "to": "done"}])

    def test_an_under_indented_block_line_is_refused_not_truncated(self):
        message = self._error("text: |-\n    aaaa\n  bbbb\n")
        self.assertIn("graph.yaml:3", message)
        self.assertIn("block scalar", message)

    def test_an_under_indented_short_block_line_is_refused(self):
        message = self._error("text: |-\n    aaaa\n  bb\n")
        self.assertIn("graph.yaml:3", message)
        self.assertIn("block scalar", message)

    def test_an_empty_block_scalar_is_the_empty_string(self):
        self.assertEqual(parse("text: |\nafter: 1\n"), {"text": "", "after": 1})
        self.assertEqual(parse("text: >-\nafter: 1\n"), {"text": "", "after": 1})


class TestFoldedIndentation(unittest.TestCase):
    def test_more_indented_lines_are_kept_literally(self):
        self.assertEqual(parse("text: >-\n  a\n    indented\n  b\n")["text"],
                         "a\n  indented\nb")

    def test_a_blank_line_before_a_more_indented_line_adds_a_break(self):
        self.assertEqual(parse("text: >-\n  a\n\n    indented\n  b\n")["text"],
                         "a\n\n  indented\nb")

    def test_two_blank_lines_become_two_breaks(self):
        self.assertEqual(parse("text: >-\n  a\n\n\n  b\n")["text"], "a\n\nb")


class TestCommentsInPlainScalars(unittest.TestCase):
    def test_an_apostrophe_in_a_plain_scalar_does_not_hide_a_comment(self):
        doc = parse("model: fake:fakes/o'brien.json  # local copy\n")
        self.assertEqual(doc["model"], "fake:fakes/o'brien.json")

    def test_a_double_quote_inside_a_plain_scalar_does_not_hide_a_comment(self):
        doc = parse('note: 6" pipe  # imperial\n')
        self.assertEqual(doc["note"], '6" pipe')

    def test_a_quote_after_a_colon_does_not_open_a_string(self):
        # Real YAML reads this as the plain scalar `b:'c` plus a comment.
        self.assertEqual(parse("a: b:'c # d\n"), {"a": "b:'c"})

    def test_a_quoted_value_still_hides_a_hash(self):
        self.assertEqual(parse("t: 'a # b'  # real comment\n"), {"t": "a # b"})
        self.assertEqual(parse('k: ["a # b"]  # real\n'), {"k": ["a # b"]})


class TestDocumentMarkers(unittest.TestCase):
    def _error(self, text):
        with self.assertRaises(YamlError) as caught:
            parse(text, filename="graph.yaml")
        return str(caught.exception)

    def test_a_second_document_is_refused(self):
        message = self._error("a: 1\n---\nb: 2\n")
        self.assertIn("graph.yaml:2", message)
        self.assertIn("multiple documents", message)

    def test_content_after_the_end_marker_is_refused(self):
        message = self._error("a: 1\n...\nb: 2\n")
        self.assertIn("graph.yaml:3", message)
        self.assertIn("multiple documents", message)

    def test_a_repeated_start_marker_is_refused(self):
        self.assertIn("multiple documents", self._error("---\na: 1\n---\nb: 2\n"))


class TestFlowRefusals(unittest.TestCase):
    def _error(self, text):
        with self.assertRaises(YamlError) as caught:
            parse(text, filename="graph.yaml")
        return str(caught.exception)

    def test_a_trailing_comma_does_not_invent_an_entry(self):
        self.assertEqual(parse("k: [a, b,]\n"), {"k": ["a", "b"]})
        self.assertEqual(parse("k: {a: 1,}\n"), {"k": {"a": 1}})

    def test_an_empty_flow_entry_is_refused(self):
        self.assertIn("empty entry", self._error("k: [a,,b]\n"))
        self.assertIn("empty entry", self._error("k: [,a]\n"))

    def test_a_flow_mapping_needs_a_space_after_the_colon(self):
        message = self._error("k: {key:value}\n")
        self.assertIn("graph.yaml:1", message)
        self.assertIn("space after", message)

    def test_a_quoted_flow_key_may_hug_the_colon(self):
        self.assertEqual(parse('k: {"a":1}\n'), {"k": {"a": 1}})

    def test_an_empty_flow_value_is_still_null(self):
        self.assertEqual(parse("k: {a:}\n"), {"k": {"a": None}})


class TestControlCharacters(unittest.TestCase):
    def test_c0_controls_are_refused(self):
        for char in ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e"):
            with self.assertRaises(YamlError) as caught:
                parse("a: x%sb: y\n" % char, filename="graph.yaml")
            message = str(caught.exception)
            self.assertIn("graph.yaml:1", message)
            self.assertIn("control character", message)

    def test_the_line_number_of_a_late_control_character(self):
        with self.assertRaises(YamlError) as caught:
            parse("a: 1\nb: 2\nc: \x0c\n", filename="graph.yaml")
        self.assertIn("graph.yaml:3", str(caught.exception))
