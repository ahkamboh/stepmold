import unittest

from jig.model import FakeModel, Model, ModelExhausted


class TestOrderedMode(unittest.TestCase):
    def test_returns_responses_in_order(self):
        model = FakeModel(["first", "second"])
        self.assertEqual(model.generate("a"), "first")
        self.assertEqual(model.generate("b"), "second")

    def test_ignores_the_prompt(self):
        model = FakeModel(["only"])
        self.assertEqual(model.generate("anything at all"), "only")

    def test_records_every_call(self):
        model = FakeModel(["x", "y"])
        model.generate("p1", grammar={"type": "object"}, max_tokens=32)
        model.generate("p2")
        self.assertEqual([c.prompt for c in model.calls], ["p1", "p2"])
        self.assertEqual(model.calls[0].grammar, {"type": "object"})
        self.assertEqual(model.calls[0].max_tokens, 32)
        self.assertIsNone(model.calls[1].grammar)

    def test_call_count(self):
        model = FakeModel(["x", "y"])
        self.assertEqual(model.call_count, 0)
        model.generate("p")
        self.assertEqual(model.call_count, 1)


class TestKeyedMode(unittest.TestCase):
    def test_matches_a_substring_of_the_prompt(self):
        model = FakeModel({"classify": "billing", "extract": "{}"})
        self.assertEqual(model.generate("please classify this ticket"), "billing")
        self.assertEqual(model.generate("now extract the fields"), "{}")

    def test_a_key_may_be_reused(self):
        model = FakeModel({"classify": "billing"})
        self.assertEqual(model.generate("classify one"), "billing")
        self.assertEqual(model.generate("classify two"), "billing")

    def test_longest_matching_key_wins(self):
        model = FakeModel({"emit": "short", "emit ticket": "long"})
        self.assertEqual(model.generate("please emit ticket now"), "long")

    def test_a_key_may_hold_a_sequence_consumed_in_order(self):
        model = FakeModel({"classify": ["billing", "refund"]})
        self.assertEqual(model.generate("classify a"), "billing")
        self.assertEqual(model.generate("classify b"), "refund")


class TestExhaustion(unittest.TestCase):
    def test_ordered_mode_raises_when_the_script_runs_out(self):
        model = FakeModel(["only"])
        model.generate("a")
        with self.assertRaises(ModelExhausted) as caught:
            model.generate("b")
        self.assertIn("2", str(caught.exception))

    def test_keyed_mode_raises_when_nothing_matches(self):
        model = FakeModel({"classify": "billing"})
        with self.assertRaises(ModelExhausted) as caught:
            model.generate("summarise the thing")
        self.assertIn("summarise the thing", str(caught.exception))

    def test_keyed_sequence_raises_when_it_runs_out(self):
        model = FakeModel({"classify": ["one"]})
        model.generate("classify a")
        with self.assertRaises(ModelExhausted) as caught:
            model.generate("classify b")
        self.assertIn("classify", str(caught.exception))

    def test_empty_script_is_rejected_up_front(self):
        with self.assertRaises(ValueError):
            FakeModel([])
        with self.assertRaises(ValueError):
            FakeModel({})

    def test_a_bad_script_type_is_rejected(self):
        with self.assertRaises(TypeError):
            FakeModel("just a string")


class TestProtocol(unittest.TestCase):
    def test_fake_model_satisfies_the_model_protocol(self):
        self.assertIsInstance(FakeModel(["x"]), Model)

    def test_an_object_without_generate_does_not(self):
        self.assertNotIsInstance(object(), Model)
