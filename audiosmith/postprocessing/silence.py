"""Silence insertion and trimming filters.

Handles punctuation-based silence injection and excess silence trimming.
"""

import re
from typing import Dict

import numpy as np

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


def inject_silence(
    wav: np.ndarray, sr: int, text: str, intensity: float
) -> np.ndarray:
    """Insert silence at punctuation boundaries.

    Splits text by punctuation, estimates chunk proportions, and inserts
    zero-filled arrays at corresponding positions in the audio.

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        text: Source text to analyze for punctuation.
        intensity: Processing intensity (0.0 to 1.0+). Controls pause duration.

    Returns:
        Audio with silence inserted at punctuation boundaries.
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


def trim_excess_silence(
    wav: np.ndarray, sr: int, max_silence_ms: int = 200,
    min_silence_pct: float = 0.0,
) -> np.ndarray:
    """Trim silence runs that exceed max_silence_ms to that duration.

    Fish Speech and other engines sometimes insert pauses 2-3x longer
    than natural speech. This trims them to a maximum duration while
    preserving short pauses that contribute to natural rhythm.

    Args:
        wav: Audio array (mono float32).
        sr: Sample rate in Hz.
        max_silence_ms: Maximum silence duration before trimming (default: 200).
        min_silence_pct: Skip trimming if audio's silence % is below this (default: 0.0).

    Returns:
        Audio with excess silence trimmed.
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
