"""Post-processing configuration dataclass.

Defines all parameters for TTS audio post-processing pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PostProcessConfig:
    """Configuration for TTS post-processing.

    Attributes:
        enable_silence: Insert silence at punctuation (default: True).
        enable_dynamics: Expand dynamic range (default: True).
        enable_breath: Add breath noise in gaps (default: True).
        enable_warmth: Boost spectral brightness (default: True).
        enable_normalize: Normalize audio to target RMS (default: False).
        enable_spectral_matching: Apply emotion-aware spectral correction (default: False).
        enable_micro_dynamics: Add per-word amplitude variation (default: False).
        enable_silence_trim: Trim excess pauses (default: False).
        max_silence_ms: Maximum silence duration before trimming (default: 200).
        emotion_aware: Use emotion metadata for processing (default: True).
        global_intensity: Processing intensity multiplier (default: 0.7).
        target_rms: Target RMS level when normalization enabled (default: 0.0).
        target_rms_adaptive: Use emotion profile RMS targets (default: False).
        spectral_tilt: Spectral tilt amount [-1.0, +1.0] (default: 0.0).
        spectral_intensity: Spectral matching intensity [0.0, 1.0] (default: 0.7).
        language: Language code for language-specific processing (default: None).
    """

    enable_silence: bool = True
    enable_dynamics: bool = True
    enable_breath: bool = True
    enable_warmth: bool = True
    enable_normalize: bool = False
    enable_spectral_matching: bool = False
    enable_micro_dynamics: bool = False
    enable_silence_trim: bool = False
    max_silence_ms: int = 200
    emotion_aware: bool = True
    global_intensity: float = 0.7
    target_rms: float = 0.0
    target_rms_adaptive: bool = False
    spectral_tilt: float = 0.0
    spectral_intensity: float = 0.7
    language: Optional[str] = None
