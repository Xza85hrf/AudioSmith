"""Pipeline configuration constants for AudioSmith.

Engine-specific post-processing presets and language-specific overrides.
"""

from __future__ import annotations

from typing import Any, Dict

# Per-engine post-processing presets (calibrated to match ElevenLabs quality)
ENGINE_PP_PRESETS: Dict[str, Dict] = {
    'piper': dict(
        enable_silence=True, enable_dynamics=True, enable_breath=True,
        enable_warmth=False, enable_spectral_matching=True,
        enable_micro_dynamics=True, enable_normalize=True,
        target_rms_adaptive=True, spectral_intensity=0.8,
    ),
    'chatterbox': dict(
        enable_silence=True, enable_dynamics=True, enable_breath=True,
        enable_warmth=True, enable_spectral_matching=True,
        enable_micro_dynamics=True, spectral_intensity=0.6,
    ),
    'fish': dict(
        enable_silence=False, enable_dynamics=True, enable_breath=True,
        enable_warmth=False, enable_spectral_matching=True,
        enable_micro_dynamics=False, enable_normalize=True,
        enable_silence_trim=True, max_silence_ms=100,
        target_rms_adaptive=True, spectral_intensity=0.5,
    ),
    'qwen3': dict(
        enable_silence=True, enable_dynamics=True, enable_breath=True,
        enable_warmth=True, enable_spectral_matching=True,
        enable_micro_dynamics=True, spectral_intensity=0.5,
    ),
    'f5': dict(
        enable_silence=True, enable_dynamics=True, enable_breath=True,
        enable_warmth=False, enable_spectral_matching=True,
        enable_micro_dynamics=True, enable_normalize=True,
        target_rms_adaptive=True, spectral_intensity=0.5,
    ),
    'indextts': dict(
        enable_silence=False, enable_dynamics=False, enable_breath=False,
        enable_warmth=False, enable_spectral_matching=False,
        enable_micro_dynamics=False, enable_normalize=False,
    ),
    'cosyvoice': dict(
        enable_silence=False, enable_dynamics=False, enable_breath=False,
        enable_warmth=False, enable_spectral_matching=False,
        enable_micro_dynamics=False, enable_normalize=True,
    ),
    'orpheus': dict(
        enable_silence=False, enable_dynamics=False, enable_breath=False,
        enable_warmth=False, enable_spectral_matching=False,
        enable_micro_dynamics=False, enable_normalize=True,
    ),
}


# Per-language post-processing overrides (stronger correction for non-English)
LANGUAGE_PP_OVERRIDES: Dict[str, Dict[str, Any]] = {
    'pl': {
        'spectral_intensity': 0.3,
        'enable_spectral_matching': True,
        'enable_dynamics': True,
        'enable_breath': False,
        'enable_normalize': True,
        'target_rms_adaptive': False,
        'target_rms': 0.13,
    },
}
