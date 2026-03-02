"""Tests for audiosmith.polish_prosody module."""

import numpy as np
import pytest

from audiosmith.polish_prosody import (
    POLISH_VOWELS,
    _find_vowel_clusters,
    _estimate_word_boundaries,
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


class TestPolishVowels:
    def test_has_9_vowels(self):
        assert len(POLISH_VOWELS) == 9

    def test_includes_standard(self):
        for v in 'aeiouy':
            assert v in POLISH_VOWELS

    def test_includes_nasal(self):
        assert 'ą' in POLISH_VOWELS
        assert 'ę' in POLISH_VOWELS

    def test_includes_o_kreska(self):
        assert 'ó' in POLISH_VOWELS


class TestFindVowelClusters:
    def test_simple_word(self):
        clusters = _find_vowel_clusters("mama")
        assert len(clusters) == 2  # a, a

    def test_no_vowels(self):
        clusters = _find_vowel_clusters("krk")
        assert len(clusters) == 0

    def test_diphthong(self):
        clusters = _find_vowel_clusters("auto")
        assert len(clusters) == 2  # au, o

    def test_polish_nasal(self):
        clusters = _find_vowel_clusters("mąka")
        assert len(clusters) == 2  # ą, a

    def test_empty_string(self):
        clusters = _find_vowel_clusters("")
        assert len(clusters) == 0


class TestEstimateWordBoundaries:
    def test_single_word(self):
        bounds = _estimate_word_boundaries("hello", 24000)
        assert len(bounds) == 1
        assert bounds[0][2] == "hello"

    def test_two_words(self):
        bounds = _estimate_word_boundaries("cześć świecie", 24000)
        assert len(bounds) == 2

    def test_empty_text(self):
        bounds = _estimate_word_boundaries("", 24000)
        assert len(bounds) == 0


class TestPenultimateStress:
    def test_modifies_multisyllabic(self):
        wav = _make_speech_like(1.0)
        result = apply_penultimate_stress(wav, SR, "dzień dobry wszystkim", 0.8)
        assert not np.allclose(wav, result, atol=1e-4)

    def test_intensity_zero_no_change(self):
        wav = _make_speech_like(0.5)
        result = apply_penultimate_stress(wav, SR, "dzień dobry", 0.0)
        np.testing.assert_array_equal(wav, result)

    def test_monosyllabic_unchanged(self):
        wav = _make_speech_like(0.5)
        result = apply_penultimate_stress(wav, SR, "tak nie", 0.8)
        # "tak" and "nie" are monosyllabic — no penultimate to stress
        np.testing.assert_array_equal(wav, result)

    def test_empty_text_unchanged(self):
        wav = _make_tone(duration=0.3)
        result = apply_penultimate_stress(wav, SR, "", 0.8)
        np.testing.assert_array_equal(wav, result)

    def test_short_audio_unchanged(self):
        wav = _make_tone(duration=0.05)
        result = apply_penultimate_stress(wav, SR, "bardzo", 0.8)
        np.testing.assert_array_equal(wav, result)


class TestQuestionIntonation:
    def test_question_modifies_audio(self):
        wav = _make_speech_like(1.0)
        result = apply_question_intonation(wav, SR, "Jak się masz?", 0.8)
        assert not np.allclose(wav, result, atol=1e-4)

    def test_declarative_unchanged(self):
        wav = _make_speech_like(1.0)
        result = apply_question_intonation(wav, SR, "Dzień dobry.", 0.8)
        np.testing.assert_array_equal(wav, result)

    def test_intensity_zero_no_change(self):
        wav = _make_speech_like(0.5)
        result = apply_question_intonation(wav, SR, "Czy to prawda?", 0.0)
        np.testing.assert_array_equal(wav, result)

    def test_empty_text_unchanged(self):
        wav = _make_tone(duration=0.3)
        result = apply_question_intonation(wav, SR, "", 0.8)
        np.testing.assert_array_equal(wav, result)

    def test_short_audio_unchanged(self):
        wav = _make_tone(duration=0.05)
        result = apply_question_intonation(wav, SR, "Co?", 0.8)
        np.testing.assert_array_equal(wav, result)


class TestSyllableTiming:
    def test_reduces_energy_variance(self):
        wav = _make_speech_like(1.0)
        text = "Bardzo ładna pogoda dzisiaj jest"
        result = normalize_syllable_timing(wav, SR, text, 0.8)
        assert not np.allclose(wav, result, atol=1e-4)

    def test_intensity_zero_no_change(self):
        wav = _make_speech_like(0.5)
        result = normalize_syllable_timing(wav, SR, "cześć świecie", 0.0)
        np.testing.assert_array_equal(wav, result)

    def test_single_word_unchanged(self):
        wav = _make_speech_like(0.5)
        result = normalize_syllable_timing(wav, SR, "cześć", 0.8)
        np.testing.assert_array_equal(wav, result)

    def test_empty_text_unchanged(self):
        wav = _make_tone(duration=0.3)
        result = normalize_syllable_timing(wav, SR, "", 0.8)
        np.testing.assert_array_equal(wav, result)

    def test_short_audio_unchanged(self):
        wav = _make_tone(duration=0.05)
        result = normalize_syllable_timing(wav, SR, "dzień dobry", 0.8)
        np.testing.assert_array_equal(wav, result)

    def test_empty_audio(self):
        wav = np.array([], dtype=np.float32)
        result = normalize_syllable_timing(wav, SR, "test", 0.8)
        assert len(result) == 0
