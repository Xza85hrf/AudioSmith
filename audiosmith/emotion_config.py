"""Centralized emotion configuration for AudioSmith.

All emotion-related mappings, intensities, and spectral profiles live here.
Individual modules import from this single source of truth.
"""

from __future__ import annotations

from typing import Dict

# Emotion → Chatterbox TTS parameter offsets (exaggeration, cfg_weight)
EMOTION_TTS_MAP: Dict[str, Dict[str, float]] = {
    'happy': {'exaggeration': 0.7, 'cfg_weight': 0.5},
    'sad': {'exaggeration': 0.3, 'cfg_weight': 0.4},
    'angry': {'exaggeration': 0.9, 'cfg_weight': 0.7},
    'fearful': {'exaggeration': 0.6, 'cfg_weight': 0.6},
    'surprised': {'exaggeration': 0.8, 'cfg_weight': 0.5},
    'whisper': {'exaggeration': 0.2, 'cfg_weight': 0.3},
    'sarcastic': {'exaggeration': 0.6, 'cfg_weight': 0.5},
    'tender': {'exaggeration': 0.3, 'cfg_weight': 0.4},
    'excited': {'exaggeration': 0.8, 'cfg_weight': 0.6},
    'determined': {'exaggeration': 0.7, 'cfg_weight': 0.6},
}

# Emotion → ElevenLabs style parameter (0.0 = neutral, 1.0 = expressive)
EMOTION_STYLE_MAP: Dict[str, float] = {
    'neutral': 0.0,
    'happy': 0.3,
    'sad': 0.2,
    'angry': 0.5,
    'fearful': 0.4,
    'surprised': 0.4,
    'whisper': 0.1,
    'excited': 0.5,
    'tender': 0.2,
    'sarcastic': 0.3,
    'determined': 0.3,
}

# Emotion → processing intensity multiplier
EMOTION_INTENSITY: Dict[str, float] = {
    "angry": 1.2,
    "excited": 1.0,
    "happy": 1.0,
    "determined": 0.9,
    "surprised": 0.8,
    "neutral": 0.7,
    "sarcastic": 0.7,
    "fearful": 0.6,
    "sad": 0.5,
    "tender": 0.3,
    "whisper": 0.1,
}

# Per-emotion spectral intensity overrides.
# Emotions with large centroid gaps need stronger correction.
# Calibrated against 12kHz-capped centroid measurements:
#   angry: raw 2569Hz vs target 2631Hz (-2.4%) — barely needs correction
#   excited: raw 3106Hz vs target 2416Hz (+28.6%) — needs darkening
#   neutral: raw 2665Hz vs target 2373Hz (+12.3%) — needs moderate darkening
#   sad: raw 3179Hz vs target 2579Hz (+23.3%) — needs darkening
#   whisper: raw 2682Hz vs target 2819Hz (-4.8%) — barely needs correction
EMOTION_SPECTRAL_INTENSITY: Dict[str, float] = {
    "angry": 0.3,     # raw already close to target
    "whisper": 0.3,   # raw already close to target
    "sad": 0.6,       # +23% bright, moderate correction to darken
    "neutral": 0.4,   # +12% bright, light correction to darken
    "excited": 0.95,  # +29% bright, aggressive correction to darken
    "happy": 0.4,
    "fearful": 0.5,
}
