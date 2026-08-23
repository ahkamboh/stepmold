"""Guards the stdlib test harness itself (see NIGHT-LOG 2026-08-24, T0)."""

import unittest

import pytest


class TestHarness(unittest.TestCase):
    def test_runner_collects_and_runs(self):
        self.assertEqual(1 + 1, 2)

    def test_raises_helper(self):
        with pytest.raises(ValueError) as caught:
            raise ValueError("boom")
        self.assertIn("boom", str(caught.value))

    def test_raises_helper_rejects_silence(self):
        with self.assertRaises(AssertionError):
            with pytest.raises(ValueError):
                pass
