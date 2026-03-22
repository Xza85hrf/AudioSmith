"""Spectral processing filters.

Handles spectral matching, warmth boosting, presence synthesis, and spectral tilt.
"""

from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from audiosmith.spectral_profiles import EmotionProfile


def measure_spectral_envelope(
    wav: np.ndarray, sr: int,
) -> Dict[str, float]:
    """Measure spectral characteristics in 9 octave bands.

    Returns centroid (Hz), brightness (0-1), and per-band RMS (dB).

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.

    Returns:
        Dict with keys:
            - centroid: Spectral centroid in Hz
            - brightness: Brightness ratio (0.0 to 1.0)
            - bands_db: Tuple of 9 per-band RMS values in dB
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


def compute_spectral_correction(
    measured: Dict[str, float],
    target_profile: "EmotionProfile",
    spectral_intensity: float,
) -> np.ndarray:
    """Compute per-band gain correction to match target spectral profile.

    Returns an array of 9 gain values (linear) for each frequency band.

    Args:
        measured: Measured spectral envelope (from measure_spectral_envelope).
        target_profile: Target EmotionProfile with band_energies_db.
        spectral_intensity: Intensity of correction (0.0 to 1.0+).

    Returns:
        Array of 9 gain values (linear scale).
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


def apply_spectral_correction(
    wav: np.ndarray, sr: int, band_gains: np.ndarray,
) -> np.ndarray:
    """Apply per-band gain correction in FFT domain with smooth interpolation.

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        band_gains: Array of 9 gain values (linear scale).

    Returns:
        Audio with spectral correction applied.
    """
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


def synthesize_presence(
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
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        target_centroid: Target spectral centroid in Hz.
        intensity: Processing intensity (0.0 to 1.0+).
        low_freq: Lower edge of presence band (Hz). Default 2000.
        high_freq: Upper edge of presence band (Hz). Default 5000.
        gap_threshold: Minimum centroid gap ratio to activate. Default 0.15.

    Returns:
        Audio with presence synthesis applied.
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


def boost_warmth(wav: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Boost spectral centroid via FFT-domain high-shelf tilt.

    Applies a smooth gain curve in frequency domain: flat below 1kHz,
    linearly rising above 1kHz up to +4dB at Nyquist. This shifts
    the spectral centroid upward without creating time-domain peak spikes.

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        intensity: Processing intensity (0.0 to 1.0+).

    Returns:
        Audio with warmth boost applied.
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


def apply_spectral_tilt(
    wav: np.ndarray, sr: int, tilt: float, intensity: float
) -> np.ndarray:
    """Apply spectral tilt to shift brightness.

    Positive tilt = brighten (boost highs, same as warmth).
    Negative tilt = darken (cut highs, boost lows).

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        tilt: Tilt amount (-1.0 to +1.0). Negative = darken.
        intensity: Overall processing intensity.

    Returns:
        Audio with spectral tilt applied.
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


def pink_noise(n_samples: int) -> np.ndarray:
    """Generate pink noise using FFT spectral shaping.

    Pink noise has equal energy per octave (1/f power spectrum),
    approximating natural breathing and ambient room tone.

    Args:
        n_samples: Number of samples to generate.

    Returns:
        Pink noise array (float32).
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


def add_breath_noise(
    wav: np.ndarray, sr: int, intensity: float
) -> np.ndarray:
    """Add subtle pink noise in silent regions to simulate breathing.

    Only injects noise in silence gaps longer than 80ms,
    at approximately -40dB below the signal peak.

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        intensity: Processing intensity (0.0 to 1.0+).

    Returns:
        Audio with breath noise added.
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
                noise = pink_noise(silence_len) * breath_amp
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


def add_micro_dynamics(
    wav: np.ndarray, sr: int, text: str, intensity: float,
) -> np.ndarray:
    """Add subtle per-word amplitude variation to simulate natural emphasis.

    TTS engines often produce flat emphasis across all words. Natural speech
    emphasizes ~20% of words (louder) and de-emphasizes ~20% (softer).

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        text: Source text to analyze for word boundaries.
        intensity: Processing intensity (0.0 to 1.0+).

    Returns:
        Audio with micro-dynamics applied.
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
