"""Tests for audiosmith.pipeline.helpers module."""

import pytest

from audiosmith.pipeline.helpers import (
    _clean_tts_text,
    _dedup_repeated_words,
    _is_fish_skippable,
    _validate_tts_duration,
)


class TestCleanTtsText:
    """Test _clean_tts_text function."""

    def test_removes_bracketed_content(self):
        """Test removal of bracketed stage directions."""
        text = "[Marty] Hello there"
        result = _clean_tts_text(text)
        assert "[Marty]" not in result
        assert "Hello there" in result

    def test_removes_parenthetical_content(self):
        """Test removal of parenthetical directions."""
        text = "Hello (laughing) there"
        result = _clean_tts_text(text)
        assert "(laughing)" not in result
        assert "Hello" in result
        assert "there" in result

    def test_removes_music_symbols(self):
        """Test removal of music/lyrics with note symbols."""
        text = "♪ La la la ♪ End"
        result = _clean_tts_text(text)
        assert "♪" not in result
        assert "End" in result

    def test_removes_leading_emdash(self):
        """Test removal of leading dialogue em-dashes."""
        text = "— This is dialogue"
        result = _clean_tts_text(text)
        assert "This is dialogue" in result
        assert not result.startswith("—")

    def test_collapses_whitespace(self):
        """Test whitespace collapse."""
        text = "Hello    there    world"
        result = _clean_tts_text(text)
        assert result == "Hello there world"

    def test_strips_leading_trailing_whitespace(self):
        """Test whitespace trimming."""
        text = "  Hello world  "
        result = _clean_tts_text(text)
        assert result == "Hello world"

    def test_empty_string(self):
        """Test with empty string."""
        result = _clean_tts_text("")
        assert result == ""

    def test_none_handling(self):
        """Test handling of None input."""
        result = _clean_tts_text(None)
        assert result is None

    def test_complex_example(self):
        """Test complex text with multiple cleaning needs."""
        text = "[Stage] — (loudly)  Hello ♪ music ♪ world  (applause)"
        result = _clean_tts_text(text)
        # Should have removed all markers and collapsed whitespace
        assert "[" not in result
        assert "]" not in result
        assert "(" not in result
        assert ")" not in result
        assert "♪" not in result
        assert "  " not in result


class TestDedupRepeatedWords:
    """Test _dedup_repeated_words function."""

    def test_collapses_three_identical_words(self):
        """Test collapsing 3+ identical consecutive words to 2."""
        text = "hello hello hello world"
        result = _dedup_repeated_words(text)
        assert result == "hello hello world"

    def test_keeps_two_identical_words(self):
        """Test that 2 identical words are preserved."""
        text = "hello hello world"
        result = _dedup_repeated_words(text)
        assert result == "hello hello world"

    def test_collapses_five_identical_words(self):
        """Test collapsing 5 identical words to 2."""
        text = "hello hello hello hello hello world"
        result = _dedup_repeated_words(text)
        assert result == "hello hello world"

    def test_case_insensitive_matching(self):
        """Test case-insensitive word comparison."""
        text = "Hello HELLO hello world"
        result = _dedup_repeated_words(text)
        assert result == "Hello HELLO world"

    def test_multiple_sequences(self):
        """Test multiple sequences of repeated words."""
        text = "hello hello hello world world world great"
        result = _dedup_repeated_words(text)
        assert result == "hello hello world world great"

    def test_short_text_no_change(self):
        """Test that texts with < 3 words are unchanged."""
        text = "hello world"
        result = _dedup_repeated_words(text)
        assert result == "hello world"

    def test_custom_max_repeats(self):
        """Test custom max_repeats parameter."""
        text = "hello hello hello hello world"
        result = _dedup_repeated_words(text, max_repeats=3)
        assert result == "hello hello hello world"

    def test_empty_string(self):
        """Test with empty string."""
        result = _dedup_repeated_words("")
        assert result == ""


class TestIsFishSkippable:
    """Test _is_fish_skippable function."""

    def test_too_few_words_returns_true(self):
        """Test that segments with < 3 words are marked skippable."""
        text = "Hi there"
        result = _is_fish_skippable(text, 0.0, 1.0)
        assert result is True

    def test_three_words_returns_false(self):
        """Test that segments with exactly 3 words are not skippable."""
        text = "Hello there world"
        result = _is_fish_skippable(text, 0.0, 1.0)
        assert result is False

    def test_large_silence_gap_with_short_text(self):
        """Test skippability with large gap before segment."""
        text = "Hello world there"  # 3 words, not skippable normally
        # But with 15s gap and 5 words max for gap case
        text_short = "Hi there"  # 2 words
        result = _is_fish_skippable(
            text_short, seg_start=20.0, seg_end=25.0, prev_end=5.0
        )
        assert result is True

    def test_small_gap_not_skippable(self):
        """Test that small gaps don't trigger skippability."""
        text = "Hi there"
        result = _is_fish_skippable(
            text, seg_start=5.0, seg_end=6.0, prev_end=4.5
        )
        assert result is True  # Still skippable due to word count

    def test_large_gap_with_long_text_not_skippable(self):
        """Test that large gaps don't cause skip if text is long enough."""
        text = "Hello there world this is great"  # 6 words
        result = _is_fish_skippable(
            text, seg_start=20.0, seg_end=25.0, prev_end=5.0
        )
        assert result is False

    def test_last_30s_of_timeline_short_text(self):
        """Test that short text in last 30s is skippable."""
        text = "Hi"
        total_duration = 60.0
        # Segment at 50s-55s, so (60 - 55) = 5s < 30s window
        result = _is_fish_skippable(
            text, seg_start=50.0, seg_end=55.0, total_duration=total_duration
        )
        assert result is True

    def test_last_30s_with_adequate_text(self):
        """Test that adequate text in last 30s is not skippable."""
        text = "Hello world this is great stuff"  # 6 words
        total_duration = 60.0
        result = _is_fish_skippable(
            text, seg_start=50.0, seg_end=55.0, total_duration=total_duration
        )
        assert result is False

    def test_early_timeline_short_text_not_skippable(self):
        """Test that short text early in timeline is not skippable (unless < 3 words)."""
        text = "Hi there"
        total_duration = 120.0
        # Segment at 5s-10s, so (120 - 10) = 110s >> 30s window
        result = _is_fish_skippable(
            text, seg_start=5.0, seg_end=10.0, total_duration=total_duration
        )
        assert result is True  # Still skippable due to word count < 3

    def test_empty_total_duration_skips_end_check(self):
        """Test that zero total_duration disables end-of-stream check."""
        text = "Hi"
        result = _is_fish_skippable(
            text, seg_start=100.0, seg_end=105.0, total_duration=0.0
        )
        assert result is True  # Skippable due to word count


class TestValidateTtsDuration:
    """Test _validate_tts_duration function."""

    def test_reasonable_duration_returns_true(self):
        """Test that reasonable TTS duration is validated."""
        # 100 words at 3 wps = ~33 seconds
        # 100 samples at 16000 Hz = 100/16000 = 0.00625s
        # Let's use: 48000 samples at 16000 Hz = 3 seconds for 10 words
        audio_samples = 48000  # 3 seconds
        sample_rate = 16000
        word_count = 10  # 3 seconds / 10 words = 0.3 wps (normal)
        result = _validate_tts_duration(audio_samples, sample_rate, word_count)
        assert result is True

    def test_too_short_duration_returns_false(self):
        """Test that suspiciously short duration is rejected."""
        audio_samples = 1000  # ~0.06 seconds
        sample_rate = 16000
        word_count = 100  # 100 words in 0.06s is impossible
        result = _validate_tts_duration(audio_samples, sample_rate, word_count)
        assert result is False

    def test_too_long_duration_returns_false(self):
        """Test that suspiciously long duration is rejected."""
        audio_samples = 1_000_000  # ~62 seconds
        sample_rate = 16000
        word_count = 1  # 1 word in 62s is impossible (max 2x margin)
        result = _validate_tts_duration(audio_samples, sample_rate, word_count)
        assert result is False

    def test_zero_audio_samples_returns_false(self):
        """Test that zero audio samples is invalid."""
        result = _validate_tts_duration(0, 16000, 10)
        assert result is False

    def test_zero_word_count_returns_false(self):
        """Test that zero word count is invalid."""
        result = _validate_tts_duration(48000, 16000, 0)
        assert result is False

    def test_negative_audio_samples_returns_false(self):
        """Test that negative audio samples is invalid."""
        result = _validate_tts_duration(-1000, 16000, 10)
        assert result is False

    def test_boundary_slow_speech(self):
        """Test boundary case: very slow speech (1.5 wps)."""
        # 10 words at 1.5 wps = 6.67 seconds
        audio_samples = int(6.67 * 16000)  # ~106720
        result = _validate_tts_duration(audio_samples, 16000, 10)
        assert result is True

    def test_boundary_fast_speech(self):
        """Test boundary case: very fast speech (5 wps)."""
        # 10 words at 5 wps = 2 seconds
        audio_samples = int(2.0 * 16000)  # 32000
        result = _validate_tts_duration(audio_samples, 16000, 10)
        assert result is True

    def test_different_sample_rate(self):
        """Test with different sample rate (48 kHz)."""
        # 10 words, 3 seconds at 48kHz = 144000 samples
        audio_samples = 144000
        sample_rate = 48000
        word_count = 10
        result = _validate_tts_duration(audio_samples, sample_rate, word_count)
        assert result is True

    def test_margin_on_low_end(self):
        """Test 2x margin on lower end: duration can be 0.5x minimum expected."""
        # 10 words at fastest (5 wps) = 2s minimum expected
        # 0.5x margin = 1s minimum acceptable
        audio_samples = int(1.0 * 16000)  # 16000
        result = _validate_tts_duration(audio_samples, 16000, 10)
        assert result is True

    def test_margin_on_high_end(self):
        """Test 2x margin on upper end: duration can be 2x maximum expected."""
        # 10 words at slowest (1.5 wps) = 6.67s maximum expected
        # 2x margin = 13.33s maximum acceptable
        audio_samples = int(13.33 * 16000)  # ~213280
        result = _validate_tts_duration(audio_samples, 16000, 10)
        assert result is True
