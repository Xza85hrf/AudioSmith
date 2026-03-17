"""AudioSmith TTS Post-Processor.

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

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from audiosmith.exceptions import TTSError

logger = logging.getLogger("audiosmith.tts_postprocessor")

# Emotion → processing intensity multiplier
_EMOTION_INTENSITY: Dict[str, float] = {
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

# Punctuation → silence duration in ms (min, max)
# Calibrated to match ElevenLabs eleven_v3 measured pauses
_PAUSE_DURATIONS: Dict[str, tuple] = {
    ".": (100, 150),
    "!": (100, 140),
    "?": (100, 140),
    ";": (80, 120),
    ":": (70, 100),
    ",": (40, 70),
    "—": (30, 50),
    "-": (30, 50),
}

# Regex to split text at punctuation boundaries while keeping the delimiter
_PUNCT_SPLIT = re.compile(r"(?<=[.!?;:,\-—])\s+")

# Per-emotion spectral intensity overrides.
# Emotions with large centroid gaps need stronger correction.
# Per-emotion spectral intensity overrides.
# Calibrated against 12kHz-capped centroid measurements:
#   angry: raw 2569Hz vs target 2631Hz (-2.4%) — barely needs correction
#   excited: raw 3106Hz vs target 2416Hz (+28.6%) — needs darkening
#   neutral: raw 2665Hz vs target 2373Hz (+12.3%) — needs moderate darkening
#   sad: raw 3179Hz vs target 2579Hz (+23.3%) — needs darkening
#   whisper: raw 2682Hz vs target 2819Hz (-4.8%) — barely needs correction
_EMOTION_SPECTRAL_INTENSITY: Dict[str, float] = {
    "angry": 0.3,     # raw already close to target
    "whisper": 0.3,   # raw already close to target
    "sad": 0.6,       # +23% bright, moderate correction to darken
    "neutral": 0.4,   # +12% bright, light correction to darken
    "excited": 0.95,  # +29% bright, aggressive correction to darken
    "happy": 0.4,
    "fearful": 0.5,
}


@dataclass
class PostProcessConfig:
    """Configuration for TTS post-processing."""

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


class TTSPostProcessor:
    """Post-process TTS audio for naturalness and expressiveness.

    Applies a 4-step chain: silence injection, dynamic range expansion,
    breath simulation, and spectral warmth boost. Each step can be
    independently enabled/disabled via PostProcessConfig.

    All processing is in-memory on numpy arrays. No file I/O.
    """

    def __init__(self, config: Optional[PostProcessConfig] = None) -> None:
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

        Args:
            audio: Mono float32 numpy array.
            sample_rate: Sample rate in Hz (22050, 24000, etc.).
            text: Source text (used for silence injection at punctuation).
            emotion: Emotion metadata dict with 'primary' and 'intensity' keys.

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
                measured = _measure_spectral_envelope(wav, sample_rate)
                # Use per-emotion spectral intensity if available
                emotion_name = emotion.get("primary", "neutral") if emotion else "neutral"
                spec_intensity = _EMOTION_SPECTRAL_INTENSITY.get(
                    emotion_name, self.config.spectral_intensity,
                )
                # Cap per-emotion intensity for non-English languages
                if lang_mod:
                    cap = lang_mod.get("spectral_intensity_cap", 1.0)
                    spec_intensity = min(spec_intensity, cap)
                correction = _compute_spectral_correction(
                    measured, emotion_profile, spec_intensity,
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
                wav = _apply_spectral_correction(wav, sample_rate, correction)
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
                wav = _synthesize_presence(
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
                wav = _trim_excess_silence(
                    wav, sample_rate, self.config.max_silence_ms,
                    min_silence_pct=20.0,
                )
                steps_applied += 1
            except Exception as e:
                logger.warning("Silence trimming failed, skipping: %s", e)

        # 3. Micro-dynamics (word-level emphasis variation)
        if self.config.enable_micro_dynamics and text:
            try:
                wav = _add_micro_dynamics(wav, sample_rate, text, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Micro-dynamics failed, skipping: %s", e)

        # 3b. Language-specific prosody (Polish)
        if lang == "pl" and text:
            try:
                from audiosmith.polish_prosody import (
                    apply_penultimate_stress, apply_question_intonation,
                    normalize_syllable_timing)
                wav = apply_penultimate_stress(wav, sample_rate, text, intensity)
                wav = apply_question_intonation(wav, sample_rate, text, intensity)
                wav = normalize_syllable_timing(wav, sample_rate, text, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Polish prosody failed, skipping: %s", e)

        # 4. Warmth boost (spectral brightness — skip when spectral matching handles it)
        if self.config.enable_warmth:
            try:
                wav = _boost_warmth(wav, sample_rate, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Warmth boost failed, skipping: %s", e)

        # 5. Silence injection (punctuation pauses)
        if self.config.enable_silence and text:
            try:
                wav = _inject_silence(wav, sample_rate, text, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Silence injection failed, skipping: %s", e)

        # 6. Dynamic range expansion (target-driven when emotion profile available)
        if self.config.enable_dynamics:
            try:
                target_dr = emotion_profile.target_dynamics_db if emotion_profile else None
                wav = _expand_dynamic_range(wav, intensity, target_dr=target_dr)
                steps_applied += 1
            except Exception as e:
                logger.warning("Dynamic range expansion failed, skipping: %s", e)

        # 6b. DR reshaping (secondary pass for stubborn gaps)
        if self.config.enable_dynamics and emotion_profile:
            try:
                wav = _reshape_dynamic_range(
                    wav, sample_rate, emotion_profile.target_dynamics_db, intensity,
                )
            except Exception as e:
                logger.warning("DR reshaping failed, skipping: %s", e)

        # 6c. Post-DR spectral re-correction
        # DR expansion changes relative frame amplitudes, shifting centroid.
        # This light correction pass compensates for centroid drift.
        if self.config.enable_spectral_matching and emotion_profile:
            try:
                re_measured = _measure_spectral_envelope(wav, sample_rate)
                emotion_name = emotion.get("primary", "neutral") if emotion else "neutral"
                # Flat intensity for re-correction: this corrects DR expansion
                # damage (uniform across emotions), not raw engine gaps.
                # Per-emotion overrides would be wrong here.
                re_intensity = 0.4
                re_correction = _compute_spectral_correction(
                    re_measured, emotion_profile, re_intensity,
                )
                wav = _apply_spectral_correction(wav, sample_rate, re_correction)
            except Exception as e:
                logger.warning("Post-DR spectral re-correction failed: %s", e)

        # 7. Breath simulation
        if self.config.enable_breath:
            try:
                wav = _add_breath_noise(wav, sample_rate, intensity)
                steps_applied += 1
            except Exception as e:
                logger.warning("Breath simulation failed, skipping: %s", e)

        # 8. Spectral tilt (legacy, for configs not using spectral matching)
        if self.config.spectral_tilt != 0.0:
            try:
                wav = _apply_spectral_tilt(wav, sample_rate, self.config.spectral_tilt, intensity)
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
        """Resolve processing intensity from config and emotion."""
        base = self.config.global_intensity

        if not self.config.emotion_aware or not emotion:
            return base

        primary = emotion.get("primary", "neutral")
        emotion_mult = _EMOTION_INTENSITY.get(primary, 0.7)
        emotion_intensity = emotion.get("intensity", 1.0)

        return base * emotion_mult * emotion_intensity


def _measure_spectral_envelope(
    wav: np.ndarray, sr: int,
) -> Dict[str, float]:
    """Measure spectral characteristics in 9 octave bands.

    Returns centroid (Hz), brightness (0-1), and per-band RMS (dB).
    """
    from audiosmith.spectral_profiles import BAND_EDGES

    fft_mag = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)

    # Cap analysis at 12kHz to match 24kHz reference profiles.
    # This prevents high-sample-rate audio (44.1/48kHz) from showing
    # artificially high centroid/brightness from extended frequencies.
    analysis_cap = min(12000.0, sr / 2.0)
    cap_mask = freqs <= analysis_cap
    capped_mag = fft_mag.copy()
    capped_mag[~cap_mask] = 0.0

    total_energy = np.sum(capped_mag ** 2)
    if total_energy < 1e-16:
        return {"centroid": 0.0, "brightness": 0.0, "bands_db": (0.0,) * 9}

    # Centroid (within analysis cap)
    centroid = float(np.sum(freqs * capped_mag) / np.sum(capped_mag))

    # Per-band energy (9 bands, capped)
    edges = BAND_EDGES + [analysis_cap]
    band_energies = []
    for i in range(len(edges) - 1):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        band_e = np.sum(capped_mag[mask] ** 2) if np.any(mask) else 1e-16
        band_energies.append(band_e)

    # Convert to dB relative to strongest band
    max_e = max(band_energies)
    bands_db = tuple(
        10 * np.log10(e / max_e) if e > 1e-16 else -60.0
        for e in band_energies
    )

    # Brightness = energy above 2kHz / total (within cap)
    bright_mask = (freqs >= 2000.0) & cap_mask
    brightness = float(np.sum(capped_mag[bright_mask] ** 2) / total_energy)

    return {"centroid": centroid, "brightness": brightness, "bands_db": bands_db}


def _compute_spectral_correction(
    measured: Dict[str, float],
    target_profile: "EmotionProfile",
    spectral_intensity: float,
) -> np.ndarray:
    """Compute per-band gain correction to match target spectral profile.

    Returns an array of 9 gain values (linear) for each frequency band.
    """
    measured_db = measured["bands_db"]
    target_db = target_profile.band_energies_db

    gains = np.ones(len(target_db), dtype=np.float32)
    for i in range(len(target_db)):
        diff_db = target_db[i] - measured_db[i]
        # Blend toward target by spectral_intensity
        correction_db = diff_db * spectral_intensity
        gains[i] = 10 ** (correction_db / 20.0)

    # Clamp — wider range allows stronger correction for extreme gaps
    gains = np.clip(gains, 0.3, 3.0)
    return gains


def _apply_spectral_correction(
    wav: np.ndarray, sr: int, band_gains: np.ndarray,
) -> np.ndarray:
    """Apply per-band gain correction in FFT domain with smooth interpolation."""
    from audiosmith.spectral_profiles import BAND_EDGES

    fft = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)

    # Band center frequencies for interpolation
    edges = BAND_EDGES + [sr / 2.0]
    centers = np.array(
        [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)],
        dtype=np.float32,
    )

    # Interpolate band gains to full frequency resolution (smooth curve)
    gain_curve = np.interp(freqs, centers, band_gains).astype(np.float32)

    # Cap correction at 10kHz — profiles are calibrated from 24kHz audio.
    # Above 10kHz, set gain to 1.0 to avoid overcutting extended high
    # frequencies in higher sample rate audio (44.1kHz, 48kHz).
    high_mask = freqs > 10000.0
    if np.any(high_mask):
        # Smooth transition: ramp from corrected gain to 1.0 over 10-12kHz
        transition = (freqs >= 8000) & (freqs <= 10000)
        if np.any(transition):
            t = (freqs[transition] - 8000) / 2000  # 0→1 over 8-10kHz
            gain_curve[transition] = gain_curve[transition] * (1.0 - t) + t
        gain_curve[high_mask] = 1.0

    result = np.fft.irfft(fft * gain_curve, n=len(wav)).astype(np.float32)
    return result


def _add_micro_dynamics(
    wav: np.ndarray, sr: int, text: str, intensity: float,
) -> np.ndarray:
    """Add subtle per-word amplitude variation to simulate natural emphasis.

    TTS engines often produce flat emphasis across all words. Natural speech
    emphasizes ~20% of words (louder) and de-emphasizes ~20% (softer).
    """
    if intensity < 0.01 or len(wav) < sr // 10:
        return wav

    words = text.split()
    if len(words) < 2:
        return wav

    total_chars = sum(len(w) for w in words) + len(words) - 1
    if total_chars == 0:
        return wav

    result = wav.copy()
    fade_samples = min(int(0.005 * sr), 64)  # 5ms crossfade
    rng = np.random.RandomState(hash(text) & 0x7FFFFFFF)  # deterministic per text

    char_pos = 0
    for word in words:
        word_len = len(word) + 1  # +1 for space
        start_sample = int((char_pos / total_chars) * len(wav))
        end_sample = int(((char_pos + word_len) / total_chars) * len(wav))
        end_sample = min(end_sample, len(wav))

        # Random emphasis: 50% normal, 25% boost, 25% reduce
        emphasis = rng.choice([0.78, 0.85, 0.90, 1.0, 1.0, 1.0, 1.0, 1.10, 1.18, 1.25])
        # Scale by intensity
        emphasis = 1.0 + (emphasis - 1.0) * intensity

        seg_len = end_sample - start_sample
        if seg_len > 2 * fade_samples and seg_len > 0:
            gains = np.ones(seg_len, dtype=np.float32) * emphasis
            # Smooth fade at boundaries
            if fade_samples > 0:
                gains[:fade_samples] = np.linspace(1.0, emphasis, fade_samples)
                gains[-fade_samples:] = np.linspace(emphasis, 1.0, fade_samples)
            result[start_sample:end_sample] *= gains

        char_pos += word_len

    return result


def _synthesize_presence(
    wav: np.ndarray, sr: int, target_centroid: float, intensity: float,
    low_freq: float = 2000.0, high_freq: float = 5000.0,
    gap_threshold: float = 0.15,
) -> np.ndarray:
    """Synthesize high-frequency presence energy modulated by speech envelope.

    When the audio lacks energy in the presence band (low centroid),
    this generates bandpass-filtered noise shaped by the speech envelope
    and mixes it in. This simulates natural consonant energy and sibilance
    that TTS engines sometimes undergenerate.

    Args:
        low_freq: Lower edge of presence band (Hz). Default 2000.
        high_freq: Upper edge of presence band (Hz). Default 5000.
            Wider bands (e.g. 6500) cover sibilant-heavy languages.
        gap_threshold: Minimum centroid gap ratio to activate. Default 0.15.
            Lower values (e.g. 0.05) activate for smaller gaps.
    """
    if intensity < 0.01 or len(wav) < sr // 10:
        return wav

    # Measure current centroid
    fft_mag = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)
    total_mag = np.sum(fft_mag)
    if total_mag < 1e-8:
        return wav
    current_centroid = float(np.sum(freqs * fft_mag) / total_mag)

    # Only synthesize if centroid is below target by gap_threshold
    gap_ratio = (target_centroid - current_centroid) / target_centroid
    if gap_ratio < gap_threshold:
        return wav

    # Generate white noise, bandpass filter to presence band
    noise = np.random.RandomState(len(wav) & 0x7FFFFFFF).normal(0, 1, len(wav)).astype(np.float32)
    noise_fft = np.fft.rfft(noise)

    # Bandpass with smooth ramps
    ramp_width = 500.0
    rolloff_width = 1000.0
    bp_gain = np.zeros_like(freqs)
    ramp_up = (freqs >= low_freq - ramp_width) & (freqs < low_freq)
    flat = (freqs >= low_freq) & (freqs <= high_freq)
    ramp_down = (freqs > high_freq) & (freqs <= high_freq + rolloff_width)
    if np.any(ramp_up):
        bp_gain[ramp_up] = (freqs[ramp_up] - (low_freq - ramp_width)) / ramp_width
    bp_gain[flat] = 1.0
    if np.any(ramp_down):
        bp_gain[ramp_down] = 1.0 - (freqs[ramp_down] - high_freq) / rolloff_width

    filtered = np.fft.irfft(noise_fft * bp_gain, n=len(wav)).astype(np.float32)

    # Modulate by speech envelope (so presence only appears during voiced segments)
    frame_size = 512
    n_frames = max(1, len(wav) // frame_size)
    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * frame_size
        e = min(s + frame_size, len(wav))
        envelope[i] = np.sqrt(np.mean(wav[s:e] ** 2))

    # Smooth envelope
    kernel_size = min(3, n_frames)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    envelope = np.convolve(envelope, kernel, mode="same")

    # Interpolate to sample level
    frame_centers = np.arange(n_frames) * frame_size + frame_size // 2
    env_samples = np.interp(np.arange(len(wav)), frame_centers, envelope).astype(np.float32)

    # Scale: proportional to gap, capped at -20dB below signal
    # Larger centroid gap → more presence added
    mix_level = min(gap_ratio * 0.5, 0.3) * intensity
    peak = np.max(np.abs(wav))
    presence = filtered * env_samples * mix_level * peak

    result = wav + presence
    return result


def _reshape_dynamic_range(
    wav: np.ndarray, sr: int, target_dr: float, intensity: float,
) -> np.ndarray:
    """Explicitly reshape envelope to hit a target dynamic range.

    Unlike _expand_dynamic_range which amplifies deviations by a factor,
    this computes the exact per-frame gain needed to achieve target DR.
    Applied as a secondary pass after standard DR expansion.
    """
    if intensity < 0.01 or len(wav) == 0:
        return wav

    rms = float(np.sqrt(np.mean(wav ** 2)))
    peak = float(np.max(np.abs(wav)))
    if rms < 1e-8:
        return wav

    current_dr = 20 * np.log10(peak / rms)
    dr_gap = target_dr - current_dr

    # Only activate for significant gaps (>2dB shortfall)
    if dr_gap < 2.0:
        return wav

    # Strategy: boost loud frames, attenuate quiet frames
    # to widen the peak-to-RMS ratio
    frame_size = 512
    n_frames = max(1, len(wav) // frame_size)

    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * frame_size
        e = min(s + frame_size, len(wav))
        envelope[i] = np.sqrt(np.mean(wav[s:e] ** 2))

    env_mean = np.mean(envelope)
    if env_mean < 1e-8:
        return wav

    # Compute normalized deviation, scale to close the DR gap
    # Each dB of gap needs ~0.15 expansion factor
    expansion = min(dr_gap * 0.15, 1.5) * intensity
    deviation = (envelope - env_mean) / env_mean
    gain_per_frame = (1.0 + expansion * deviation).astype(np.float32)
    gain_per_frame = np.clip(gain_per_frame, 0.5, 2.0)

    frame_centers = np.arange(n_frames) * frame_size + frame_size // 2
    gain = np.interp(np.arange(len(wav)), frame_centers, gain_per_frame).astype(np.float32)

    return wav * gain


def _trim_excess_silence(
    wav: np.ndarray, sr: int, max_silence_ms: int = 200,
    min_silence_pct: float = 0.0,
) -> np.ndarray:
    """Trim silence runs that exceed max_silence_ms to that duration.

    Fish Speech and other engines sometimes insert pauses 2-3x longer
    than natural speech. This trims them to a maximum duration while
    preserving short pauses that contribute to natural rhythm.

    If min_silence_pct > 0, skip trimming when the audio's current
    silence percentage is already at or below that threshold.
    """
    if len(wav) == 0:
        return wav

    max_silence_samples = int(max_silence_ms * sr / 1000)
    frame_size = 256
    silence_threshold = 0.01

    # Guard: skip if already below minimum silence threshold
    if min_silence_pct > 0:
        total_frames = max(1, len(wav) // frame_size)
        silent_frames = 0
        for i in range(total_frames):
            s = i * frame_size
            e = min(s + frame_size, len(wav))
            if np.sqrt(np.mean(wav[s:e] ** 2)) < silence_threshold:
                silent_frames += 1
        current_pct = silent_frames / total_frames * 100
        if current_pct <= min_silence_pct:
            return wav

    # Find silent regions
    regions = []
    i = 0
    while i < len(wav):
        end = min(i + frame_size, len(wav))
        rms = np.sqrt(np.mean(wav[i:end] ** 2))
        if rms < silence_threshold:
            silence_start = i
            while i < len(wav):
                end = min(i + frame_size, len(wav))
                rms = np.sqrt(np.mean(wav[i:end] ** 2))
                if rms >= silence_threshold:
                    break
                i += frame_size
            regions.append((silence_start, min(i, len(wav))))
        else:
            i += frame_size

    if not regions:
        return wav

    # Build output by copying audio and trimming long silences
    parts = []
    prev_end = 0
    for sil_start, sil_end in regions:
        # Copy audio before this silence
        parts.append(wav[prev_end:sil_start])
        sil_len = sil_end - sil_start
        if sil_len > max_silence_samples:
            # Trim to max, keep start of silence (has natural fade)
            parts.append(wav[sil_start:sil_start + max_silence_samples])
        else:
            parts.append(wav[sil_start:sil_end])
        prev_end = sil_end

    # Remainder after last silence
    if prev_end < len(wav):
        parts.append(wav[prev_end:])

    return np.concatenate(parts) if parts else wav


def _inject_silence(
    wav: np.ndarray, sr: int, text: str, intensity: float
) -> np.ndarray:
    """Insert silence at punctuation boundaries.

    Splits text by punctuation, estimates chunk proportions, and inserts
    zero-filled arrays at corresponding positions in the audio.
    """
    chunks = _PUNCT_SPLIT.split(text)
    if len(chunks) <= 1:
        return wav

    # Find which punctuation ends each chunk (except the last)
    pauses = []
    pos = 0
    for chunk in chunks[:-1]:
        pos += len(chunk)
        # Look at the character just before the split
        last_char = chunk.rstrip()[-1] if chunk.rstrip() else "."
        duration_range = _PAUSE_DURATIONS.get(last_char, (75, 100))
        pause_ms = duration_range[0] + (duration_range[1] - duration_range[0]) * intensity
        pause_ms = min(pause_ms, duration_range[1])
        pauses.append(int(pause_ms * sr / 1000))

    # Distribute audio proportionally across text chunks
    total_chars = sum(len(c) for c in chunks)
    if total_chars == 0:
        return wav

    parts = []
    sample_pos = 0
    for i, chunk in enumerate(chunks):
        chunk_ratio = len(chunk) / total_chars
        chunk_samples = int(chunk_ratio * len(wav))

        # Last chunk gets remaining samples
        if i == len(chunks) - 1:
            chunk_samples = len(wav) - sample_pos

        chunk_audio = wav[sample_pos : sample_pos + chunk_samples]
        parts.append(chunk_audio)
        sample_pos += chunk_samples

        if i < len(pauses):
            silence = np.zeros(pauses[i], dtype=np.float32)
            parts.append(silence)

    return np.concatenate(parts)


def _expand_dynamic_range(
    wav: np.ndarray, intensity: float, target_dr: Optional[float] = None,
) -> np.ndarray:
    """Expand dynamic range toward a target peak-to-RMS ratio.

    When target_dr is provided (from emotion profiles), measures current DR
    and scales expansion strength to close the gap. Without a target, uses
    a fixed expansion factor.
    """
    if intensity < 0.01 or len(wav) == 0:
        return wav

    frame_size = 512
    n_frames = max(1, len(wav) // frame_size)

    # Compute per-frame RMS envelope
    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(wav))
        frame = wav[start:end]
        envelope[i] = np.sqrt(np.mean(frame ** 2))

    if envelope.max() < 1e-8:
        return wav

    # Smooth envelope with 5-frame boxcar
    kernel_size = min(5, n_frames)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    envelope_smooth = np.convolve(envelope, kernel, mode="same")

    env_mean = envelope_smooth.mean()
    if env_mean < 1e-8:
        return wav

    # Target-driven: scale expansion factor to close DR gap
    if target_dr is not None and target_dr > 0:
        peak = float(np.max(np.abs(wav)))
        rms = float(np.sqrt(np.mean(wav ** 2)))
        current_dr = 20 * np.log10(peak / rms) if rms > 1e-8 else 0.0
        dr_gap = target_dr - current_dr  # positive = need more DR
        # Scale expansion: 0 gap → factor 0.3, large gap → factor 2.5
        expansion_factor = np.clip(0.3 + dr_gap * 0.4, 0.3, 2.5) * intensity
    else:
        expansion_factor = 1.2 * intensity

    deviation = (envelope_smooth - env_mean) / env_mean
    gain_per_frame = (1.0 + expansion_factor * deviation).astype(np.float32)
    gain_per_frame = np.clip(gain_per_frame, 0.4, 2.0)

    # Interpolate gain to sample level
    frame_centers = np.arange(n_frames) * frame_size + frame_size // 2
    sample_indices = np.arange(len(wav))
    gain = np.interp(sample_indices, frame_centers, gain_per_frame).astype(np.float32)

    return wav * gain


def _add_breath_noise(
    wav: np.ndarray, sr: int, intensity: float
) -> np.ndarray:
    """Add subtle pink noise in silent regions to simulate breathing.

    Only injects noise in silence gaps longer than 80ms,
    at approximately -40dB below the signal peak.
    """
    if intensity < 0.01:
        return wav

    min_silence_samples = int(0.08 * sr)  # 80ms minimum silence
    frame_size = 256
    silence_threshold = 0.01

    # Detect silent frames
    result = wav.copy()
    peak = np.max(np.abs(wav))
    if peak < 1e-8:
        return result

    # Target breath amplitude: -40dB below peak, scaled by intensity
    breath_amp = peak * 10 ** (-40.0 / 20.0) * intensity

    i = 0
    while i < len(wav) - min_silence_samples:
        frame = wav[i : i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))

        if rms < silence_threshold:
            # Find extent of silence
            silence_start = i
            while i < len(wav):
                end = min(i + frame_size, len(wav))
                rms = np.sqrt(np.mean(wav[i:end] ** 2))
                if rms >= silence_threshold:
                    break
                i += frame_size

            # Clamp to actual array bounds
            actual_end = min(i, len(wav))
            silence_len = actual_end - silence_start
            if silence_len >= min_silence_samples:
                noise = _pink_noise(silence_len) * breath_amp
                # Fade in/out (5ms)
                fade_samples = min(int(0.005 * sr), silence_len // 4)
                if fade_samples > 0:
                    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
                    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
                    noise[:fade_samples] *= fade_in
                    noise[-fade_samples:] *= fade_out
                result[silence_start:actual_end] += noise
        else:
            i += frame_size

    return result


def _boost_warmth(wav: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Boost spectral centroid via FFT-domain high-shelf tilt.

    Applies a smooth gain curve in frequency domain: flat below 1kHz,
    linearly rising above 1kHz up to +4dB at Nyquist. This shifts
    the spectral centroid upward without creating time-domain peak spikes.
    """
    if intensity < 0.01 or len(wav) < 2:
        return wav

    fft = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)

    # Gain curve: 1.0 below 500Hz, rising to 1.0 + boost above
    # Starts at 500Hz to catch upper formants (F2, F3) where brightness lives
    max_boost_db = 10.0 * intensity
    max_boost_linear = 10 ** (max_boost_db / 20.0)
    shelf_start = 500.0  # Hz
    shelf_end = sr / 2.0  # Nyquist

    gain = np.ones_like(freqs)
    above_shelf = freqs > shelf_start
    if np.any(above_shelf):
        # Linear ramp from 1.0 at shelf_start to max_boost at Nyquist
        ramp = (freqs[above_shelf] - shelf_start) / (shelf_end - shelf_start)
        ramp = np.clip(ramp, 0, 1)
        gain[above_shelf] = 1.0 + (max_boost_linear - 1.0) * ramp

    result = np.fft.irfft(fft * gain, n=len(wav)).astype(np.float32)
    return result


def _pink_noise(n_samples: int) -> np.ndarray:
    """Generate pink noise using FFT spectral shaping.

    Pink noise has equal energy per octave (1/f power spectrum),
    approximating natural breathing and ambient room tone.
    """
    white = np.random.normal(0, 1, n_samples).astype(np.float32)
    fft = np.fft.rfft(white)

    freqs = np.fft.rfftfreq(n_samples)
    # 1/sqrt(f) amplitude scaling = 1/f power spectrum (pink)
    scale = 1.0 / np.sqrt(freqs + 1e-8)
    scale[0] = 0  # Remove DC offset

    pink = np.fft.irfft(fft * scale, n=n_samples).astype(np.float32)

    # Normalize to unit peak
    peak = np.max(np.abs(pink))
    if peak > 1e-8:
        pink /= peak

    return pink


def _apply_spectral_tilt(
    wav: np.ndarray, sr: int, tilt: float, intensity: float
) -> np.ndarray:
    """Apply spectral tilt to shift brightness.

    Positive tilt = brighten (boost highs, same as warmth).
    Negative tilt = darken (cut highs, boost lows).

    Args:
        wav: Audio array.
        sr: Sample rate.
        tilt: Tilt amount (-1.0 to +1.0). Negative = darken.
        intensity: Overall processing intensity.
    """
    if abs(tilt) < 0.01 or len(wav) < 2:
        return wav

    fft = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)

    # Crossover at 1kHz: above gets boosted/cut, below gets the inverse
    crossover = 1000.0
    nyquist = sr / 2.0
    max_db = 8.0 * abs(tilt) * intensity

    gain = np.ones_like(freqs)

    above = freqs > crossover
    below = (freqs > 20) & (freqs <= crossover)  # skip DC

    if tilt > 0:
        # Brighten: boost above crossover
        if np.any(above):
            ramp = (freqs[above] - crossover) / (nyquist - crossover)
            ramp = np.clip(ramp, 0, 1)
            gain[above] = 10 ** (max_db * ramp / 20.0)
    else:
        # Darken: cut above crossover, slight boost below
        if np.any(above):
            ramp = (freqs[above] - crossover) / (nyquist - crossover)
            ramp = np.clip(ramp, 0, 1)
            gain[above] = 10 ** (-max_db * ramp / 20.0)
        if np.any(below):
            ramp = 1.0 - (freqs[below] / crossover)
            gain[below] = 10 ** (max_db * 0.3 * ramp / 20.0)

    result = np.fft.irfft(fft * gain, n=len(wav)).astype(np.float32)
    return result
