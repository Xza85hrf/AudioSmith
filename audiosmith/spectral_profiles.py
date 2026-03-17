"""Emotion-specific spectral reference profiles for TTS post-processing.

Targets derived from ElevenLabs eleven_v3 output analysis (Feb 2026).
Each profile captures the spectral and loudness characteristics of
natural human speech for a given emotion, enabling the post-processor
to shape local TTS output toward professional quality.

9-band octave-spaced frequency decomposition:
    Band 0: 0-150 Hz    (sub-bass, chest resonance)
    Band 1: 150-350 Hz  (bass, fundamental F0)
    Band 2: 350-700 Hz  (low-mid, warmth)
    Band 3: 700-1.4 kHz (mid, vowel formants F1)
    Band 4: 1.4-2.8 kHz (upper-mid, presence, F2)
    Band 5: 2.8-4 kHz   (presence, consonant clarity)
    Band 6: 4-6 kHz     (brilliance, sibilance)
    Band 7: 6-8 kHz     (air, breathiness)
    Band 8: 8+ kHz      (ultra-high, sparkle)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# Octave band edges in Hz (9 bands)
BAND_EDGES: List[float] = [0, 150, 350, 700, 1400, 2800, 4000, 6000, 8000]


@dataclass(frozen=True)
class EmotionProfile:
    """Spectral and loudness targets for a specific emotion."""

    emotion: str
    centroid_hz: float
    target_rms: float
    target_dynamics_db: float
    band_energies_db: tuple  # 9 relative dB values per band
    brightness: float  # ratio of energy above 2kHz to total (0.0-1.0)


# ── Reference profiles from ElevenLabs eleven_v3 analysis ──

_PROFILES: Dict[str, EmotionProfile] = {
    "angry": EmotionProfile(
        emotion="angry",
        centroid_hz=2631.0,
        target_rms=0.158,
        target_dynamics_db=15.5,
        band_energies_db=(0.0, -2.1, -3.5, -5.8, -8.2, -12.0, -18.5, -24.0, -30.0),
        brightness=0.68,
    ),
    "neutral": EmotionProfile(
        emotion="neutral",
        centroid_hz=2373.0,
        target_rms=0.145,
        target_dynamics_db=15.8,
        band_energies_db=(0.0, -1.8, -3.0, -5.2, -9.0, -13.5, -19.0, -25.0, -31.0),
        brightness=0.55,
    ),
    "sad": EmotionProfile(
        emotion="sad",
        centroid_hz=2579.0,
        target_rms=0.127,
        target_dynamics_db=17.3,
        band_energies_db=(0.0, -2.0, -3.2, -5.5, -8.8, -12.8, -18.0, -23.5, -29.5),
        brightness=0.62,
    ),
    "excited": EmotionProfile(
        emotion="excited",
        centroid_hz=2416.0,
        target_rms=0.173,
        target_dynamics_db=14.7,
        band_energies_db=(0.0, -1.5, -2.8, -5.0, -8.5, -13.0, -19.5, -25.5, -31.5),
        brightness=0.56,
    ),
    "whisper": EmotionProfile(
        emotion="whisper",
        centroid_hz=2819.0,
        target_rms=0.105,
        target_dynamics_db=17.4,
        band_energies_db=(0.0, -3.0, -4.5, -6.0, -7.5, -10.5, -15.0, -20.0, -26.0),
        brightness=0.72,
    ),
    "happy": EmotionProfile(
        emotion="happy",
        centroid_hz=2450.0,
        target_rms=0.160,
        target_dynamics_db=15.0,
        band_energies_db=(0.0, -1.6, -2.9, -5.1, -8.3, -12.5, -18.8, -24.5, -30.5),
        brightness=0.58,
    ),
    "fearful": EmotionProfile(
        emotion="fearful",
        centroid_hz=2700.0,
        target_rms=0.115,
        target_dynamics_db=16.5,
        band_energies_db=(0.0, -2.5, -3.8, -5.9, -8.0, -11.5, -16.5, -22.0, -28.0),
        brightness=0.65,
    ),
}

# Default profile for unknown emotions
_DEFAULT = _PROFILES["neutral"]


# Language-specific spectral modifiers (adjustments to English-calibrated profiles)
LANGUAGE_SPECTRAL_MODIFIERS: Dict[str, Dict[str, float]] = {
    "pl": {
        "band_0_boost_db": -6.0,   # 0-150 Hz: cut excess sub-bass (Fish overproduces)
        "band_1_boost_db": -3.0,   # 150-350 Hz: cut bass (shifts energy ratio toward highs)
        "band_2_boost_db": -5.0,   # 350-700 Hz: cut excess low-mid (Fish dominant band)
        "band_3_boost_db": 1.5,    # 700-1.4 kHz: slight mid lift (compensate low cuts)
        "band_4_boost_db": 3.0,    # 1.4-2.8 kHz: upper-mid presence (F2 formants)
        "band_5_boost_db": 7.0,    # 2.8-4 kHz: consonant clarity (sz, cz, ś, ź)
        "band_6_boost_db": 7.0,    # 4-6 kHz: sibilance (ż, rz, high-freq fricatives)
        "band_7_boost_db": 5.0,    # 6-8 kHz: air/breathiness
        "band_8_boost_db": 5.0,    # 8+ kHz: sparkle/sibilant tail
        "brightness_offset": 0.08,  # Polish speech is brighter than English
        "centroid_offset_hz": 100.0,  # Reduced: low-band cuts already shift centroid up
        "spectral_intensity_cap": 0.3,  # Cap per-emotion English correction
        "correction_clip_max": 4.0,  # Allow stronger per-band correction
    },
}


def get_profile(emotion: str) -> EmotionProfile:
    """Get spectral profile for an emotion, falling back to neutral."""
    return _PROFILES.get(emotion, _DEFAULT)


def get_language_modifier(language: Optional[str]) -> Dict[str, float]:
    """Get language-specific spectral modifier, or empty dict if unsupported."""
    if not language:
        return {}
    return LANGUAGE_SPECTRAL_MODIFIERS.get(language, {})


def list_emotions() -> List[str]:
    """Return all available emotion profile names."""
    return list(_PROFILES.keys())
