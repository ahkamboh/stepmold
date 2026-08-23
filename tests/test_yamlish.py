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

    def test_block_scalars_are_rejected_loudly(self):
        self.assertIn("block scalar", self._error("a: |\n  text\n"))

    def test_anchors_are_rejected_loudly(self):
        self.assertIn("anchor", self._error("a: &base\n"))

    def test_unterminated_flow_collection(self):
        self.assertIn("flow", self._error("a: [1, 2\n"))

    def test_unterminated_quote(self):
        self.assertIn("unterminated", self._error("a: 'oops\n"))
