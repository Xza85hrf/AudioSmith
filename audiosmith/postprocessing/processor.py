"""TTS post-processor orchestrator.

Main entry point that chains all spectral and dynamic processing filters.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from audiosmith.emotion_config import (
    EMOTION_INTENSITY,
    EMOTION_SPECTRAL_INTENSITY,
)
from audiosmith.exceptions import TTSError
from audiosmith.postprocessing.config import PostProcessConfig
from audiosmith.postprocessing.dynamics import (
    expand_dynamic_range,
    reshape_dynamic_range,
)
from audiosmith.postprocessing.silence import inject_silence, trim_excess_silence
from audiosmith.postprocessing.spectral import (
    add_breath_noise,
    add_micro_dynamics,
    apply_spectral_correction,
    apply_spectral_tilt,
    boost_warmth,
    compute_spectral_correction,
    measure_spectral_envelope,
    synthesize_presence,
)

logger = logging.getLogger("audiosmith.tts_postprocessor")


class TTSPostProcessor:
    """Post-process TTS audio for naturalness and expressiveness.

    Applies a comprehensive processing chain: spectral correction, presence synthesis,
    silence trimming, micro-dynamics, warmth boosting, silence injection, dynamic range
    expansion, breath simulation, and spectral tilt. Each step can be independently
    enabled/disabled via PostProcessConfig.

    All processing is in-memory on numpy arrays. No file I/O.
    """

    def __init__(self, config: Optional[PostProcessConfig] = None) -> None:
        """Initialize post-processor.

        Args:
            config: PostProcessConfig instance (uses defaults if None).
        """
        self.config = config or PostProcessConfig()

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
        emotion: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
    ) -> np.ndarray:
        """Post-process TTS audio.

        Processes through a multi-stage pipeline:
            1. Spectral correction (FFT-domain, on raw speech)
            2. Presence synthesis (add high-freq energy if centroid too low)
            3. Silence trimming (trim excess pauses)
            4. Micro-dynamics (word-level emphasis variation)
            5. Language-specific prosody (stress, intonation, timing)
            6. Warmth boost (spectral brightness)
            7. Silence injection (punctuation pauses)
            8. Dynamic range expansion
            9. DR reshaping (secondary pass for stubborn gaps)
           10. Post-DR spectral re-correction (compensate for centroid drift)
           11. Breath simulation (pink noise in gaps)
           12. Spectral tilt (legacy brightness adjustment)
           13. Normalization (RMS or peak preservation)

        Args:
            audio: Mono float32 numpy array.
            sample_rate: Sample rate in Hz (22050, 24000, etc.).
            text: Source text (used for silence injection at punctuation).
            emotion: Emotion metadata dict with 'primary' and 'intensity' keys.
            language: Language code (e.g., 'en', 'pl') for language-specific processing.

        Returns:
            Processed audio array (mono float32).

        Raises:
            TTSError: If all processing steps fail.
        """
        if audio.size == 0:
            return audio

        intensity = self._resolve_intensity(emotion)
        wav = audio.copy()
        orig_peak = np.max(np.abs(wav))
        steps_applied = 0

        # Resolve emotion-adaptive targets
        emotion_profile = None
        if emotion and (self.config.enable_spectral_matching or self.config.target_rms_adaptive):
            from audiosmith.spectral_profiles import get_profile
            emotion_profile = get_profile(emotion.get("primary", "neutral"))

        # Resolve language modifier once for all steps
        lang = language or self.config.language
        lang_mod = {}
        if lang:
            from audiosmith.spectral_profiles import get_language_modifier
            lang_mod = get_language_modifier(lang)

        # 1. Spectral correction (FFT-domain, on raw speech first)
        if self.config.enable_spectral_matching and emotion_profile:
            try:
                measured = measure_spectral_envelope(wav, sample_rate)
                # Use per-emotion spectral intensity if available
                emotion_name = emotion.get("primary", "neutral") if emotion else "neutral"
                spec_intensity = EMOTION_SPECTRAL_INTENSITY.get(
                    emotion_name, self.config.spectral_intensity,
                )
                # Cap per-emotion intensity for non-English languages
                if lang_mod:
                    cap = lang_mod.get("spectral_intensity_cap", 1.0)
                    spec_intensity = min(spec_intensity, cap)
                correction = compute_spectral_correction(
                    measured, emotion_profile, spec_intensity,  # type: ignore[arg-type]
                )
                # Apply language-specific spectral boosts
                if lang_mod:
                    for key, boost_db in lang_mod.items():
                        if key.startswith("band_") and key.endswith("_boost_db"):
                            band_idx = int(key.split("_")[1])
                            if 0 <= band_idx < len(correction):
                                correction[band_idx] *= 10 ** (boost_db / 20.0)
                    clip_max = lang_mod.get("correction_clip_max", 3.0)
                    correction = np.clip(correction, 0.3, clip_max)
                wav = apply_spectral_correction(wav, sample_rate, correction)
                steps_applied += 1
            except Exception as e:
                logger.warning("Spectral matching failed, skipping: %s", e)

        # 1b. Presence synthesis (add high-freq energy when centroid is too low)
        if self.config.enable_spectral_matching and emotion_profile:
            try:
                # Adjust target centroid and presence band for language
                centroid_target = emotion_profile.centroid_hz
                centroid_target += lang_mod.get("centroid_offset_hz", 0.0)
                # Widen presence band for sibilant-heavy languages
                presence_high = 5000.0
                if lang_mod.get("band_6_boost_db", 0) > 0:
                    presence_high = 6500.0
                # Lower activation threshold for languages with known gaps
                gap_threshold = 0.05 if lang_mod else 0.15
                wav = synthesize_presence(
                    wav, sample_rate, centroid_target, intensity,
                    high_freq=presence_high, gap_threshold=gap_threshold,
                )
            except Exception as e:
                logger.warning("Presence synthesis failed, skipping: %s", e)

        # 2. Silence trimming (trim engine's excess pauses, skip if already sparse)
        if self.config.enable_silence_trim:
            try:
                # Only trim if current silence exceeds a reasonable threshold
                # This prevents trimming audio that already has sparse pauses
                wav = trim_excess_silence(
                    wav, sample_rate, self.config.max_silence_ms,
                    min_silence_pct=20.0,
                )
                steps_applied += 1
            except Exception as e:
                logger.warning("Silence trimming failed, skipping: %s", e)

        # 3. Micro-dynamics (word-level emphasis variation)
        if self.config.enable_micro_dynamics and text:
            try:
                wav = add_micro_dynamics(wav, sample_rate, text, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Micro-dynamics failed, skipping: %s", e)

        # 3b. Language-specific prosody (multi-language support)
        if text and lang:
            try:
                from audiosmith.prosody import (
                    apply_penultimate_stress, apply_question_intonation,
                    normalize_syllable_timing)
                wav = apply_penultimate_stress(wav, sample_rate, text, intensity, language=lang)
                wav = apply_question_intonation(wav, sample_rate, text, intensity, language=lang)
                wav = normalize_syllable_timing(wav, sample_rate, text, intensity, language=lang)
                steps_applied += 1
            except Exception as e:
                logger.warning("Language-specific prosody failed for '%s', skipping: %s", lang, e)

        # 4. Warmth boost (spectral brightness — skip when spectral matching handles it)
        if self.config.enable_warmth:
            try:
                wav = boost_warmth(wav, sample_rate, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Warmth boost failed, skipping: %s", e)

        # 5. Silence injection (punctuation pauses)
        if self.config.enable_silence and text:
            try:
                wav = inject_silence(wav, sample_rate, text, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Silence injection failed, skipping: %s", e)

        # 6. Dynamic range expansion (target-driven when emotion profile available)
        if self.config.enable_dynamics:
            try:
                target_dr = emotion_profile.target_dynamics_db if emotion_profile else None
                wav = expand_dynamic_range(wav, intensity, target_dr=target_dr)
                steps_applied += 1
            except Exception as e:
                logger.warning("Dynamic range expansion failed, skipping: %s", e)

        # 6b. DR reshaping (secondary pass for stubborn gaps)
        if self.config.enable_dynamics and emotion_profile:
            try:
                wav = reshape_dynamic_range(
                    wav, sample_rate, emotion_profile.target_dynamics_db, intensity,
                )
            except Exception as e:
                logger.warning("DR reshaping failed, skipping: %s", e)

        # 6c. Post-DR spectral re-correction
        # DR expansion changes relative frame amplitudes, shifting centroid.
        # This light correction pass compensates for centroid drift.
        if self.config.enable_spectral_matching and emotion_profile:
            try:
                re_measured = measure_spectral_envelope(wav, sample_rate)
                emotion_name = emotion.get("primary", "neutral") if emotion else "neutral"
                # Flat intensity for re-correction: this corrects DR expansion
                # damage (uniform across emotions), not raw engine gaps.
                # Per-emotion overrides would be wrong here.
                re_intensity = 0.4
                re_correction = compute_spectral_correction(
                    re_measured, emotion_profile, re_intensity,  # type: ignore[arg-type]
                )
                wav = apply_spectral_correction(wav, sample_rate, re_correction)
            except Exception as e:
                logger.warning("Post-DR spectral re-correction failed: %s", e)

        # 7. Breath simulation
        if self.config.enable_breath:
            try:
                wav = add_breath_noise(wav, sample_rate, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Breath simulation failed, skipping: %s", e)

        # 8. Spectral tilt (legacy, for configs not using spectral matching)
        if self.config.spectral_tilt != 0.0:
            try:
                wav = apply_spectral_tilt(wav, sample_rate, self.config.spectral_tilt, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Spectral tilt failed, skipping: %s", e)

        if steps_applied == 0 and any([
            self.config.enable_silence,
            self.config.enable_dynamics,
            self.config.enable_breath,
            self.config.enable_warmth,
            self.config.enable_spectral_matching,
            self.config.enable_micro_dynamics,
            self.config.enable_silence_trim,
            self.config.spectral_tilt != 0.0,
        ]):
            raise TTSError(
                "All post-processing steps failed",
                error_code="TTS_PP_ERR",
            )

        # 8. Normalization — emotion-adaptive RMS or peak preservation
        target_rms = self.config.target_rms
        if self.config.target_rms_adaptive and emotion_profile:
            target_rms = emotion_profile.target_rms

        if self.config.enable_normalize and target_rms > 0:
            current_rms = np.sqrt(np.mean(wav ** 2))
            if current_rms > 1e-8:
                wav = wav * (target_rms / current_rms)
                wav = np.clip(wav, -1.0, 1.0)
        else:
            new_peak = np.max(np.abs(wav))
            if new_peak > 1e-8 and orig_peak > 1e-8:
                wav = wav * (orig_peak / new_peak)

        return wav

    def _resolve_intensity(
        self, emotion: Optional[Dict[str, Any]]
    ) -> float:
        """Resolve processing intensity from config and emotion.

        Args:
            emotion: Emotion dict with 'primary' and 'intensity' keys (optional).

        Returns:
            Computed intensity multiplier (0.0 to 1.0+).
        """
        base = self.config.global_intensity

        if not self.config.emotion_aware or not emotion:
            return base

        primary = emotion.get("primary", "neutral")
        emotion_mult = EMOTION_INTENSITY.get(primary, 0.7)
        emotion_intensity = emotion.get("intensity", 1.0)

        return base * emotion_mult * emotion_intensity  # type: ignore[no-any-return]
