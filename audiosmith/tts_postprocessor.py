"""AudioSmith TTS Post-Processor (Backward Compatibility Re-export).

This module re-exports from audiosmith.postprocessing for backward compatibility.
New code should import from audiosmith.postprocessing directly.

Improves local TTS engine output quality via spectral matching,
micro-dynamics, silence injection, dynamic range expansion,
breath simulation, and spectral warmth/tilt.

Processing chain per segment:
    raw TTS → spectral correction → presence synthesis →
    silence trim → micro-dynamics → warmth → silence injection →
    dynamic range → DR reshape → post-DR spectral re-correction →
    breath noise → spectral tilt → RMS normalization (adaptive) → output

When spectral matching is enabled (Fish, Piper), warmth should be disabled
to avoid fighting the spectral correction. Warmth is a blunt high-shelf
boost; spectral matching provides per-emotion, per-band frequency shaping.

All algorithms are numpy-only (no scipy dependency).
"""

# Re-export public API for backward compatibility
from audiosmith.postprocessing import PostProcessConfig, TTSPostProcessor
from audiosmith.postprocessing.silence import inject_silence as _inject_silence
from audiosmith.postprocessing.silence import trim_excess_silence as _trim_excess_silence
from audiosmith.postprocessing.dynamics import (
    expand_dynamic_range as _expand_dynamic_range,
)
from audiosmith.postprocessing.dynamics import (
    reshape_dynamic_range as _reshape_dynamic_range,
)
from audiosmith.postprocessing.spectral import (
    add_breath_noise as _add_breath_noise,
)
from audiosmith.postprocessing.spectral import (
    add_micro_dynamics as _add_micro_dynamics,
)
from audiosmith.postprocessing.spectral import (
    apply_spectral_correction as _apply_spectral_correction,
)
from audiosmith.postprocessing.spectral import (
    apply_spectral_tilt as _apply_spectral_tilt,
)
from audiosmith.postprocessing.spectral import (
    boost_warmth as _boost_warmth,
)
from audiosmith.postprocessing.spectral import (
    compute_spectral_correction as _compute_spectral_correction,
)
from audiosmith.postprocessing.spectral import (
    measure_spectral_envelope as _measure_spectral_envelope,
)
from audiosmith.postprocessing.spectral import (
    pink_noise as _pink_noise,
)
from audiosmith.postprocessing.spectral import (
    synthesize_presence as _synthesize_presence,
)

__all__ = [
    "PostProcessConfig",
    "TTSPostProcessor",
    "_inject_silence",
    "_trim_excess_silence",
    "_expand_dynamic_range",
    "_reshape_dynamic_range",
    "_add_breath_noise",
    "_add_micro_dynamics",
    "_apply_spectral_correction",
    "_apply_spectral_tilt",
    "_boost_warmth",
    "_compute_spectral_correction",
    "_measure_spectral_envelope",
    "_pink_noise",
    "_synthesize_presence",
]
