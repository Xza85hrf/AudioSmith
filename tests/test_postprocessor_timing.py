"""Tests for TTSPostProcessor timing instrumentation.

Tests the step_timings, _timed_step, and get_timing_report functionality
added to the post-processor.
"""

import time

import numpy as np
import pytest

from audiosmith.postprocessing.config import PostProcessConfig
from audiosmith.postprocessing.processor import TTSPostProcessor

SR = 24000  # Standard sample rate for tests


def _make_tone(freq: float = 440.0, duration: float = 0.5, sr: int = SR) -> np.ndarray:
    """Generate a sine tone for testing."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _make_speech_like(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Generate speech-like audio: tone + silence gaps."""
    tone = _make_tone(200, 0.15, sr)
    gap = np.zeros(int(0.05 * sr), dtype=np.float32)
    pattern = np.concatenate([tone, gap])
    repeats = max(1, int(duration * sr / len(pattern)))
    wav = np.tile(pattern, repeats)[: int(sr * duration)]
    return wav.astype(np.float32)


class TestStepTimingsInitialization:
    """Test that step_timings is properly initialized."""

    def test_step_timings_empty_after_init(self):
        """step_timings should be an empty list after __init__."""
        processor = TTSPostProcessor()
        assert processor.step_timings == []
        assert isinstance(processor.step_timings, list)

    def test_step_timings_empty_with_custom_config(self):
        """step_timings should be empty even with custom config."""
        config = PostProcessConfig(enable_warmth=True, enable_dynamics=True)
        processor = TTSPostProcessor(config)
        assert processor.step_timings == []


class TestTimedStep:
    """Test the _timed_step method."""

    def test_timed_step_records_name_and_duration(self):
        """_timed_step should record (name, duration) tuple."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        def dummy_fn():
            return "result"

        result = processor._timed_step("test_step", dummy_fn)

        assert len(processor.step_timings) == 1
        name, duration = processor.step_timings[0]
        assert name == "test_step"
        assert isinstance(duration, float)
        assert duration >= 0.0

    def test_timed_step_duration_is_positive(self):
        """Duration should be a positive number (or zero for instant)."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        def quick_fn():
            return 42

        processor._timed_step("quick", quick_fn)
        _, duration = processor.step_timings[0]
        assert duration >= 0.0

    def test_timed_step_returns_function_result(self):
        """_timed_step should return the function's return value."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        def return_value():
            return "expected_value"

        result = processor._timed_step("step", return_value)
        assert result == "expected_value"

    def test_timed_step_with_args(self):
        """_timed_step should pass arguments to the function."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        def add(a, b):
            return a + b

        result = processor._timed_step("addition", add, 3, 5)
        assert result == 8
        assert len(processor.step_timings) == 1

    def test_timed_step_with_kwargs(self):
        """_timed_step should pass keyword arguments to the function."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        def multiply(x, factor=1):
            return x * factor

        result = processor._timed_step("multiply", multiply, 10, factor=3)
        assert result == 30
        assert len(processor.step_timings) == 1

    def test_timed_step_with_args_and_kwargs(self):
        """_timed_step should handle both args and kwargs."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        def concat(prefix, value, suffix=""):
            return f"{prefix}{value}{suffix}"

        result = processor._timed_step("concat", concat, "Hello", " World", suffix="!")
        assert result == "Hello World!"
        assert len(processor.step_timings) == 1

    def test_timed_step_accumulates_multiple_calls(self):
        """Multiple _timed_step calls should accumulate in step_timings."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        processor._timed_step("step1", lambda: 1)
        processor._timed_step("step2", lambda: 2)
        processor._timed_step("step3", lambda: 3)

        assert len(processor.step_timings) == 3
        names = [name for name, _ in processor.step_timings]
        assert names == ["step1", "step2", "step3"]

    def test_timed_step_duration_reflects_execution_time(self):
        """Duration should roughly match actual execution time."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        def slow_fn():
            time.sleep(0.01)  # 10ms sleep
            return "done"

        processor._timed_step("slow", slow_fn)
        _, duration = processor.step_timings[0]

        # Duration should be at least ~10ms (with some tolerance for system variability)
        assert duration >= 0.008


class TestGetTimingReport:
    """Test the get_timing_report method."""

    def test_timing_report_returns_dict(self):
        """get_timing_report should return a dict."""
        processor = TTSPostProcessor()
        report = processor.get_timing_report()
        assert isinstance(report, dict)

    def test_timing_report_empty_when_no_steps(self):
        """get_timing_report should return empty dict when no steps recorded."""
        processor = TTSPostProcessor()
        processor.step_timings = []
        report = processor.get_timing_report()
        assert report == {}

    def test_timing_report_converts_list_to_dict(self):
        """get_timing_report should convert step_timings list to dict."""
        processor = TTSPostProcessor()
        processor.step_timings = [("step_a", 0.1), ("step_b", 0.2), ("step_c", 0.15)]
        report = processor.get_timing_report()

        assert report == {
            "step_a": 0.1,
            "step_b": 0.2,
            "step_c": 0.15,
        }

    def test_timing_report_from_timed_steps(self):
        """get_timing_report should include all timed steps."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        processor._timed_step("encode", lambda: None)
        processor._timed_step("process", lambda: None)
        processor._timed_step("decode", lambda: None)

        report = processor.get_timing_report()
        assert set(report.keys()) == {"encode", "process", "decode"}
        assert all(isinstance(v, float) for v in report.values())

    def test_timing_report_values_are_floats(self):
        """All values in timing report should be floats."""
        processor = TTSPostProcessor()
        processor.step_timings = [("a", 0.1), ("b", 0.05), ("c", 0.03)]
        report = processor.get_timing_report()

        for value in report.values():
            assert isinstance(value, float)


class TestStepTimingsReset:
    """Test that step_timings is reset at the start of each process() call."""

    def test_step_timings_reset_on_process(self):
        """step_timings should be reset to empty list at beginning of process()."""
        config = PostProcessConfig(
            enable_dynamics=True,
            enable_breath=False,
            enable_silence=False,
            enable_warmth=False,
        )
        processor = TTSPostProcessor(config)

        # Manually add some timings
        processor.step_timings = [("old_step", 0.5)]

        # Call process with minimal audio
        audio = _make_speech_like(duration=0.2)
        processor.process(audio, SR)

        # step_timings should be reset to empty list
        assert processor.step_timings == []

    def test_each_process_call_resets_timings(self):
        """Each process() call should reset step_timings at its start."""
        config = PostProcessConfig(
            enable_dynamics=True,
            enable_breath=False,
            enable_silence=False,
            enable_warmth=False,
        )
        processor = TTSPostProcessor(config)

        audio = _make_speech_like(duration=0.2)

        # Manually set up timings from "previous" call
        processor.step_timings = [("prev_step_1", 0.1), ("prev_step_2", 0.2)]
        prev_timings_count = len(processor.step_timings)

        # Call process
        processor.process(audio, SR)

        # step_timings should be reset (empty since process doesn't populate it yet)
        assert processor.step_timings == []


class TestProcessWithTiming:
    """Test timing instrumentation interaction with process() calls."""

    def test_process_maintains_step_timings_attribute(self):
        """process() should maintain the step_timings attribute correctly."""
        config = PostProcessConfig(
            enable_dynamics=True,
            enable_breath=False,
            enable_silence=False,
            enable_warmth=False,
        )
        processor = TTSPostProcessor(config)

        audio = _make_speech_like(duration=0.2)
        processor.process(audio, SR)

        # step_timings should exist and be a list
        assert isinstance(processor.step_timings, list)

    def test_process_with_empty_audio_early_return(self):
        """process() with empty audio should return early without resetting timings."""
        processor = TTSPostProcessor()
        # Set up some initial timings
        processor.step_timings = [("previous_step", 0.5)]

        audio = np.array([], dtype=np.float32)
        result = processor.process(audio, SR)

        # Empty audio should return as-is before reaching the reset line
        assert len(result) == 0
        # Timings are not reset for empty audio (early return at line 96)
        assert processor.step_timings == [("previous_step", 0.5)]

    def test_timing_report_empty_after_process_without_recorded_steps(self):
        """get_timing_report() should return empty dict when no steps recorded."""
        processor = TTSPostProcessor()
        audio = _make_speech_like(duration=0.2)
        processor.process(audio, SR)

        # After process, step_timings is reset but not populated by basic process
        report = processor.get_timing_report()
        assert isinstance(report, dict)
        assert report == {}

    def test_manual_timing_with_timed_step_before_process(self):
        """_timed_step should work independently of process()."""
        processor = TTSPostProcessor()
        processor.step_timings = []

        # Manually time some operations before/after process
        processor._timed_step("setup", lambda: None)
        audio = _make_speech_like(duration=0.2)
        processor.process(audio, SR)  # This resets step_timings

        # After process(), timings should be reset
        assert processor.step_timings == []

        # Now manually time after process
        processor._timed_step("cleanup", lambda: None)
        assert len(processor.step_timings) == 1
        assert processor.step_timings[0][0] == "cleanup"

    def test_process_with_different_configs_resets_timings(self):
        """Different process() calls should each reset timings independently."""
        config1 = PostProcessConfig(enable_dynamics=True)
        processor = TTSPostProcessor(config1)

        # Set initial timings
        processor.step_timings = [("step1", 0.1), ("step2", 0.2)]

        audio = _make_speech_like(duration=0.2)
        processor.process(audio, SR)

        # Timings should be reset
        assert processor.step_timings == []

        # Add new timings and process again
        processor._timed_step("new_step", lambda: 42)
        assert len(processor.step_timings) == 1
        processor.process(audio, SR)
        assert processor.step_timings == []
