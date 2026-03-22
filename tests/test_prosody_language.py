"""Tests for language-parameterized audiosmith.prosody module."""

import numpy as np

from audiosmith.prosody import (
    POLISH_VOWELS,
    _find_vowel_clusters,
    apply_penultimate_stress,
    apply_question_intonation,
    normalize_syllable_timing,
)

SR = 24000


def _make_tone(freq: float = 440.0, duration: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _make_speech_like(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    tone = _make_tone(200, 0.15, sr)
    gap = np.zeros(int(0.05 * sr), dtype=np.float32)
    pattern = np.concatenate([tone, gap])
    repeats = max(1, int(duration * sr / len(pattern)))
    return np.tile(pattern, repeats)[:int(sr * duration)].astype(np.float32)


class TestBackwardCompatibilityAlias:
    """Verify POLISH_VOWELS alias still works."""

    def test_polish_vowels_alias_exists(self):
        """POLISH_VOWELS should be available for backward compatibility."""
        assert len(POLISH_VOWELS) == 9

    def test_alias_has_standard_vowels(self):
        for v in "aeiouy":
            assert v in POLISH_VOWELS

    def test_alias_has_nasal_vowels(self):
        assert "ą" in POLISH_VOWELS
        assert "ę" in POLISH_VOWELS

    def test_alias_has_o_kreska(self):
        assert "ó" in POLISH_VOWELS


class TestFindVowelClustersWithVowelSet:
    """Test _find_vowel_clusters with explicit vowel set."""

    def test_with_explicit_vowels(self):
        """_find_vowel_clusters should work with explicit vowel set."""
        vowels = frozenset("aeiou")
        clusters = _find_vowel_clusters("mama", vowels)
        assert len(clusters) == 2  # a, a

    def test_with_polish_vowels(self):
        """Should handle Polish vowels when passed."""
        vowels = frozenset("aeiouyóąę")
        clusters = _find_vowel_clusters("mąka", vowels)
        assert len(clusters) == 2  # ą, a

    def test_no_vowels(self):
        vowels = frozenset("aeiou")
        clusters = _find_vowel_clusters("krk", vowels)
        assert len(clusters) == 0

    def test_empty_string(self):
        vowels = frozenset("aeiou")
        clusters = _find_vowel_clusters("", vowels)
        assert len(clusters) == 0


class TestPenultimateStressWithLanguage:
    """Test apply_penultimate_stress with language parameter."""

    def test_polish_modifies_multisyllabic(self):
        wav = _make_speech_like(1.0)
        result = apply_penultimate_stress(
            wav, SR, "dzień dobry wszystkim", 0.8, language="pl"
        )
        assert not np.allclose(wav, result, atol=1e-4)

    def test_polish_default_language(self):
        """Language should default to Polish."""
        wav = _make_speech_like(1.0)
        result_explicit = apply_penultimate_stress(
            wav, SR, "dzień dobry wszystkim", 0.8, language="pl"
        )
        result_default = apply_penultimate_stress(
            wav, SR, "dzień dobry wszystkim", 0.8
        )
        np.testing.assert_array_equal(result_explicit, result_default)

    def test_english_should_skip_penultimate(self):
        """English has variable stress, should not apply penultimate."""
        wav = _make_speech_like(1.0)
        result = apply_penultimate_stress(wav, SR, "hello world today", 0.8, language="en")
        # Should return unchanged because English doesn't have penultimate stress
        np.testing.assert_array_equal(wav, result)

    def test_intensity_zero_no_change(self):
        wav = _make_speech_like(0.5)
        result = apply_penultimate_stress(wav, SR, "dzień dobry", 0.0, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_monosyllabic_unchanged(self):
        wav = _make_speech_like(0.5)
        result = apply_penultimate_stress(wav, SR, "tak nie", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_empty_text_unchanged(self):
        wav = _make_tone(duration=0.3)
        result = apply_penultimate_stress(wav, SR, "", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_short_audio_unchanged(self):
        wav = _make_tone(duration=0.05)
        result = apply_penultimate_stress(wav, SR, "bardzo", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)


class TestQuestionIntonationWithLanguage:
    """Test apply_question_intonation with language parameter."""

    def test_question_modifies_audio(self):
        wav = _make_speech_like(1.0)
        result = apply_question_intonation(
            wav, SR, "Jak się masz?", 0.8, language="pl"
        )
        assert not np.allclose(wav, result, atol=1e-4)

    def test_question_default_language(self):
        """Language parameter should have a default."""
        wav = _make_speech_like(1.0)
        result_explicit = apply_question_intonation(
            wav, SR, "Jak się masz?", 0.8, language="pl"
        )
        result_default = apply_question_intonation(wav, SR, "Jak się masz?", 0.8)
        np.testing.assert_array_equal(result_explicit, result_default)

    def test_english_question_applies_intonation(self):
        """English questions should also get pitch rise (universal)."""
        wav = _make_speech_like(1.0)
        result = apply_question_intonation(wav, SR, "Hello?", 0.8, language="en")
        # Should apply because questions are universal
        assert not np.allclose(wav, result, atol=1e-4)

    def test_declarative_unchanged(self):
        wav = _make_speech_like(1.0)
        result = apply_question_intonation(wav, SR, "Dzień dobry.", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_intensity_zero_no_change(self):
        wav = _make_speech_like(0.5)
        result = apply_question_intonation(wav, SR, "Czy to prawda?", 0.0, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_empty_text_unchanged(self):
        wav = _make_tone(duration=0.3)
        result = apply_question_intonation(wav, SR, "", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_short_audio_unchanged(self):
        wav = _make_tone(duration=0.05)
        result = apply_question_intonation(wav, SR, "Co?", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)


class TestSyllableTimingWithLanguage:
    """Test normalize_syllable_timing with language parameter."""

    def test_polish_reduces_energy_variance(self):
        wav = _make_speech_like(1.0)
        text = "Bardzo ładna pogoda dzisiaj jest"
        result = normalize_syllable_timing(wav, SR, text, 0.8, language="pl")
        assert not np.allclose(wav, result, atol=1e-4)

    def test_polish_default_language(self):
        """Language should default to Polish."""
        wav = _make_speech_like(1.0)
        text = "Bardzo ładna pogoda dzisiaj jest"
        result_explicit = normalize_syllable_timing(
            wav, SR, text, 0.8, language="pl"
        )
        result_default = normalize_syllable_timing(wav, SR, text, 0.8)
        np.testing.assert_array_equal(result_explicit, result_default)

    def test_english_should_skip_syllable_timing(self):
        """English is stress-timed, should not apply syllable timing normalization."""
        wav = _make_speech_like(1.0)
        text = "Hello there my friends today"
        result = normalize_syllable_timing(wav, SR, text, 0.8, language="en")
        # Should return unchanged because English is stress-timed, not syllable-timed
        np.testing.assert_array_equal(wav, result)

    def test_french_applies_syllable_timing(self):
        """French is syllable-timed, should apply normalization."""
        wav = _make_speech_like(1.0)
        text = "Bonjour mon ami comment allez-vous"
        result = normalize_syllable_timing(wav, SR, text, 0.8, language="fr")
        assert not np.allclose(wav, result, atol=1e-4)

    def test_intensity_zero_no_change(self):
        wav = _make_speech_like(0.5)
        result = normalize_syllable_timing(wav, SR, "cześć świecie", 0.0, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_single_word_unchanged(self):
        wav = _make_speech_like(0.5)
        result = normalize_syllable_timing(wav, SR, "cześć", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_empty_text_unchanged(self):
        wav = _make_tone(duration=0.3)
        result = normalize_syllable_timing(wav, SR, "", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_short_audio_unchanged(self):
        wav = _make_tone(duration=0.05)
        result = normalize_syllable_timing(wav, SR, "dzień dobry", 0.8, language="pl")
        np.testing.assert_array_equal(wav, result)

    def test_empty_audio(self):
        wav = np.array([], dtype=np.float32)
        result = normalize_syllable_timing(wav, SR, "test", 0.8, language="pl")
        assert len(result) == 0
