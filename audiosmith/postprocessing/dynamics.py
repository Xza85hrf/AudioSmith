"""Dynamic range processing filters.

Handles dynamic range expansion and reshaping for natural prosody.
"""

from typing import Optional

import numpy as np


def expand_dynamic_range(
    wav: np.ndarray, intensity: float, target_dr: Optional[float] = None,
) -> np.ndarray:
    """Expand dynamic range toward a target peak-to-RMS ratio.

    When target_dr is provided (from emotion profiles), measures current DR
    and scales expansion strength to close the gap. Without a target, uses
    a fixed expansion factor.

    Args:
        wav: Audio array (mono float32).
        intensity: Processing intensity (0.0 to 1.0+).
        target_dr: Target dynamic range in dB (optional). If provided,
            expansion is scaled to close the gap from current to target.

    Returns:
        Audio with expanded dynamic range.
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

    return wav * gain  # type: ignore[no-any-return]


def reshape_dynamic_range(
    wav: np.ndarray, sr: int, target_dr: float, intensity: float,
) -> np.ndarray:
    """Explicitly reshape envelope to hit a target dynamic range.

    Unlike expand_dynamic_range which amplifies deviations by a factor,
    this computes the exact per-frame gain needed to achieve target DR.
    Applied as a secondary pass after standard DR expansion.

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        target_dr: Target dynamic range in dB.
        intensity: Processing intensity (0.0 to 1.0+).

    Returns:
        Audio with reshaped dynamic range.
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

    return wav * gain  # type: ignore[no-any-return]
