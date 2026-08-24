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
        """The helper must complain when the block raises nothing.

        How it complains differs by runner: this repo's stdlib shim raises
        AssertionError, while real pytest raises Failed, which derives from
        BaseException rather than Exception. Asserting AssertionError specifically made
        the suite red for anyone who had pytest installed — and CI could never see it,
        because CI runs the shim. Accept either signal.
        """
        try:
            with pytest.raises(ValueError):
                pass
        except BaseException as exc:  # noqa: BLE001 - catching both runners is the point
            self.assertNotIsInstance(
                exc, ValueError, "the helper let the awaited exception type through"
            )
        else:
            self.fail("pytest.raises did not complain when nothing was raised")
