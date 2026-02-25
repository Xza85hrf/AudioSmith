"""Tests for audiosmith.progress module."""

import pytest
from unittest.mock import MagicMock, patch
from audiosmith.progress import ProgressTracker


class TestProgressTracker:
    @pytest.fixture
    def tracker(self):
        return ProgressTracker(total_steps=3, use_rich=False)

    def test_step_increments_counter(self, tracker):
        assert tracker._current_step == 0
        with tracker.step("Step1"):
            pass
        assert tracker._current_step == 1

    def test_multiple_steps(self, tracker):
        for name in ["A", "B", "C"]:
            with tracker.step(name):
                pass
        assert tracker._current_step == 3

    def test_step_calls_callback(self):
        cb = MagicMock()
        tracker = ProgressTracker(total_steps=2, use_rich=False, callback=cb)
        with tracker.step("TestStep"):
            pass
        cb.assert_called_once_with("TestStep", 1, 2, "complete")

    def test_no_callback_no_error(self, tracker):
        with tracker.step("Safe"):
            pass

    def test_update_no_error(self, tracker):
        tracker.update(1, "processing")

    def test_complete_without_start(self, tracker):
        tracker.complete()

    def test_fallback_when_no_rich(self):
        tracker = ProgressTracker(total_steps=1, use_rich=False)
        assert tracker._rich_available is False
        with tracker.step("NoRich"):
            pass

    @patch('audiosmith.progress.sys')
    def test_is_tty_true(self, mock_sys):
        mock_sys.stdout.isatty.return_value = True
        assert ProgressTracker._is_tty() is True

    @patch('audiosmith.progress.sys')
    def test_is_tty_false(self, mock_sys):
        mock_sys.stdout.isatty.return_value = False
        assert ProgressTracker._is_tty() is False
