"""Tests for Fish Speech guard logic in TTS synthesis.

These tests verify that the helper functions are properly integrated
into the TTS synthesis pipeline for Fish Speech engine protection.
"""

import pytest

from audiosmith.pipeline.helpers import (
    _is_fish_skippable,
    _validate_tts_duration,
)


class TestFishSpeechGuardIntegration:
    """Test Fish Speech guard functions used in TTS synthesis.

    These are unit tests for the guard functions themselves.
    Integration tests with the full pipeline are handled separately.
    """

    def test_is_fish_skippable_guards_short_text(self):
        """Test that short segments are marked for skipping."""
        # < 3 words should be skipped
        assert _is_fish_skippable("Hi there", 0.0, 1.0) is True

    def test_is_fish_skippable_allows_normal_segments(self):
        """Test that normal segments are not skipped."""
        # >= 3 words in normal position should not be skipped
        assert _is_fish_skippable("Hi there world", 0.0, 1.0) is False

    def test_is_fish_skippable_guards_gap_and_short(self):
        """Test that short text after large gaps is skipped."""
        # Large gap (15s) + short text (2 words) = skip
        assert _is_fish_skippable(
            "Hi there",
            seg_start=20.0,
            seg_end=21.0,
            prev_end=5.0
        ) is True

    def test_is_fish_skippable_guards_end_of_stream(self):
        """Test that short text near end is skipped."""
        # Last 30s with short text
        assert _is_fish_skippable(
            "Hi",
            seg_start=50.0,
            seg_end=55.0,
            total_duration=60.0
        ) is True

    def test_validate_tts_duration_accepts_normal(self):
        """Test that normal durations pass validation."""
        # 10 words in ~3 seconds is normal
        audio_samples = 48000  # 3s at 16kHz
        result = _validate_tts_duration(audio_samples, 16000, 10)
        assert result is True

    def test_validate_tts_duration_rejects_too_short(self):
        """Test that impossibly short durations fail validation."""
        # 100 words in 0.01s is impossible
        audio_samples = 160  # 0.01s at 16kHz
        result = _validate_tts_duration(audio_samples, 16000, 100)
        assert result is False

    def test_validate_tts_duration_rejects_too_long(self):
        """Test that impossibly long durations fail validation."""
        # 1 word in 60s is impossible (2x margin on slowest)
        audio_samples = 960000  # 60s at 16kHz
        result = _validate_tts_duration(audio_samples, 16000, 1)
        assert result is False

    def test_validate_tts_duration_zero_samples(self):
        """Test that zero samples fail validation."""
        result = _validate_tts_duration(0, 16000, 10)
        assert result is False

    def test_validate_tts_duration_zero_words(self):
        """Test that zero words fail validation."""
        result = _validate_tts_duration(48000, 16000, 0)
        assert result is False
