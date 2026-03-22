"""Language-aware prosody transformations for TTS post-processing.

Supports multiple language-specific stress and rhythm patterns:
    - Penultimate stress: Polish, Swahili, Welsh (accent on second-to-last syllable)
    - Ultimate stress: French (accent on final syllable)
    - Syllable-timed rhythm: Polish, French, Spanish (equal syllable durations)
    - Question intonation: Universal (pitch rise on questions)

All functions are numpy-only and independently callable. Each function accepts
a language parameter (defaults to Polish for backward compatibility).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from audiosmith.language_data import get_language

logger = logging.getLogger("audiosmith.prosody")


def _find_vowel_clusters(word: str, vowels: set[str] | frozenset[str]) -> List[Tuple[int, int]]:
    """Find start/end indices of vowel clusters (syllable nuclei) in a word.

    Args:
        word: The word to analyze.
        vowels: Set of characters to treat as vowels (language-specific).

    Returns:
        List of (start_index, end_index) tuples for each vowel cluster.
    """
    clusters: List[Tuple[int, int]] = []
    i = 0
    lower = word.lower()
    while i < len(lower):
        if lower[i] in vowels:
            start = i
            while i < len(lower) and lower[i] in vowels:
                i += 1
            clusters.append((start, i))
        else:
            i += 1
    return clusters


def _estimate_word_boundaries(
    text: str, total_samples: int,
) -> List[Tuple[int, int, str]]:
    """Estimate sample-level word boundaries from text proportionally.

    Returns list of (start_sample, end_sample, word).
    """
    words = text.split()
    if not words:
        return []

    total_chars = sum(len(w) for w in words) + max(0, len(words) - 1)
    if total_chars == 0:
        return []

    boundaries: List[Tuple[int, int, str]] = []
    char_pos = 0
    for word in words:
        word_len = len(word) + 1  # +1 for space
        start = int((char_pos / total_chars) * total_samples)
        end = int(((char_pos + word_len) / total_chars) * total_samples)
        end = min(end, total_samples)
        boundaries.append((start, end, word))
        char_pos += word_len

    return boundaries


def apply_penultimate_stress(
    audio: np.ndarray,
    sr: int,
    text: str,
    intensity: float,
    language: str = "pl",
) -> np.ndarray:
    """Boost amplitude on the penultimate (or language-specific) stressed syllable.

    For languages with fixed penultimate stress (e.g., Polish), this emphasizes
    the second-to-last syllable of each word. For other languages with different
    stress patterns, this function skips processing automatically.

    Args:
        audio: Mono float32 numpy array.
        sr: Sample rate in Hz.
        text: Text to analyze for stress positions.
        intensity: Processing strength (0.0-1.0).
        language: ISO 639-1 language code (default "pl" for Polish).

    Returns:
        Audio with stressed syllables emphasized (or unchanged if language
        doesn't have penultimate stress).
    """
    config = get_language(language)

    # Early return for languages without penultimate stress
    if config.stress_position != "penultimate":
        return audio

    if intensity < 0.01 or len(audio) < sr // 10 or not text.strip():
        return audio

    result = audio.copy()
    fade_samples = min(int(0.01 * sr), 128)  # 10ms crossfade
    boundaries = _estimate_word_boundaries(text, len(audio))

    for start, end, word in boundaries:
        clusters = _find_vowel_clusters(word, config.vowels)
        if len(clusters) < 2:
            continue  # Monosyllabic — no penultimate

        # Penultimate syllable = second-to-last vowel cluster
        penult = clusters[-2]
        word_len = len(word)
        if word_len == 0:
            continue

        # Map syllable position within word to sample range
        syllable_start_ratio = penult[0] / word_len
        syllable_end_ratio = penult[1] / word_len
        seg_len = end - start
        syl_start = start + int(syllable_start_ratio * seg_len)
        syl_end = start + int(syllable_end_ratio * seg_len)

        # Extend syllable region to include surrounding consonants
        extend = int(0.3 * (syl_end - syl_start))
        syl_start = max(start, syl_start - extend)
        syl_end = min(end, syl_end + extend)

        syl_len = syl_end - syl_start
        if syl_len < 2 * fade_samples:
            continue

        # Boost: 1.5-2.5 dB scaled by intensity
        boost_db = 1.5 + 1.0 * intensity
        boost_linear = 10 ** (boost_db / 20.0)

        gains = np.ones(syl_len, dtype=np.float32) * boost_linear
        if fade_samples > 0 and syl_len > 2 * fade_samples:
            gains[:fade_samples] = np.linspace(1.0, boost_linear, fade_samples)
            gains[-fade_samples:] = np.linspace(boost_linear, 1.0, fade_samples)

        result[syl_start:syl_end] *= gains

    return result


def apply_question_intonation(
    audio: np.ndarray,
    sr: int,
    text: str,
    intensity: float,
    language: str = "pl",
) -> np.ndarray:
    """Apply question intonation — pitch rise on questions.

    Questions universally receive a pitch rise. The implementation applies
    a subtle frequency shift to the last 30% of the audio when the text
    ends with a question mark.

    Args:
        audio: Mono float32 numpy array.
        sr: Sample rate in Hz.
        text: Text to check for question marks.
        intensity: Processing strength (0.0-1.0).
        language: ISO 639-1 language code (default "pl" for Polish).
            Included for extensibility; question intonation applies universally.

    Returns:
        Audio with question intonation applied (if text ends with ?).
    """
    if intensity < 0.01 or not text.strip() or len(audio) < sr // 10:
        return audio

    stripped = text.strip()
    if not stripped.endswith("?"):
        return audio

    # Apply pitch rise to last 30% of audio
    rise_start = int(len(audio) * 0.7)
    rise_region = audio[rise_start:]

    if len(rise_region) < 256:
        return audio

    # FFT-domain frequency shift (5-15 Hz upward, scaled by intensity)
    shift_hz = 5.0 + 10.0 * intensity
    np.fft.rfft(rise_region)
    np.fft.rfftfreq(len(rise_region), d=1.0 / sr)

    # Phase shift for frequency displacement
    phase_shift = 2 * np.pi * shift_hz * np.arange(len(rise_region)) / sr
    shifted = rise_region * np.cos(phase_shift).astype(np.float32)

    # Blend with original (50% wet to keep natural quality)
    blend = 0.5 * intensity
    result = audio.copy()
    result[rise_start:] = rise_region * (1.0 - blend) + shifted * blend

    return result


def normalize_syllable_timing(
    audio: np.ndarray,
    sr: int,
    text: str,
    intensity: float,
    language: str = "pl",
) -> np.ndarray:
    """Normalize energy across syllables for syllable-timed languages.

    Languages like Polish and French are syllable-timed (each syllable has
    roughly equal duration), while English is stress-timed with large energy
    variation. TTS engines trained on English often produce exaggerated
    energy differences. This function compresses inter-syllable variance
    toward equal energy for syllable-timed languages.

    Args:
        audio: Mono float32 numpy array.
        sr: Sample rate in Hz.
        text: Text for word boundary estimation.
        intensity: Processing strength (0.0-1.0).
        language: ISO 639-1 language code (default "pl" for Polish).

    Returns:
        Audio with normalized syllable timing (or unchanged if language
        is not syllable-timed).
    """
    config = get_language(language)

    # Early return for stress-timed languages
    if not config.syllable_timed:
        return audio

    if intensity < 0.01 or len(audio) < sr // 10 or not text.strip():
        return audio

    boundaries = _estimate_word_boundaries(text, len(audio))
    if len(boundaries) < 2:
        return audio

    # Measure per-word RMS
    word_rms = []
    for start, end, _ in boundaries:
        seg = audio[start:end]
        if len(seg) > 0:
            word_rms.append(float(np.sqrt(np.mean(seg ** 2))))
        else:
            word_rms.append(0.0)

    mean_rms = np.mean(word_rms)
    if mean_rms < 1e-8:
        return audio

    # Compress each word's amplitude toward the mean
    result = audio.copy()
    compression = intensity * 0.3  # 30% compression at full intensity
    fade_samples = min(int(0.005 * sr), 64)  # 5ms crossfade

    for i, (start, end, _) in enumerate(boundaries):
        if word_rms[i] < 1e-8:
            continue

        # Target = blend between current RMS and mean
        target_rms = word_rms[i] + (mean_rms - word_rms[i]) * compression
        gain = target_rms / word_rms[i]
        gain = np.clip(gain, 0.7, 1.4)

        seg_len = end - start
        if seg_len < 2 * fade_samples:
            result[start:end] *= gain
        else:
            gains = np.ones(seg_len, dtype=np.float32) * gain
            if fade_samples > 0:
                gains[:fade_samples] = np.linspace(1.0, gain, fade_samples)
                gains[-fade_samples:] = np.linspace(gain, 1.0, fade_samples)
            result[start:end] *= gains

    return result


# Backward compatibility — will be removed in v2.0
POLISH_VOWELS = get_language("pl").vowels
