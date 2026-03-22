"""Sequential audio scheduling, drift correction, and buffer rendering for dubbing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from audiosmith.models import DubbingConfig, DubbingSegment, ScheduledSegment

logger = logging.getLogger(__name__)


class AudioMixer:
    """Schedules TTS segments sequentially with drift correction, then renders to a numpy buffer."""

    def __init__(self, config: DubbingConfig, background_path: Optional[Path] = None) -> None:
        self.min_gap_ms = config.min_gap_ms
        self.max_speedup = config.max_speedup
        self.silence_reset_gap = config.silence_reset_gap
        self.sample_rate = config.dubbed_sample_rate
        self.background_path = background_path
        self.allow_extended_timing = getattr(config, 'allow_extended_timing', False)

    def schedule(self, segments: List[DubbingSegment]) -> List[ScheduledSegment]:
        """Build a schedule that keeps segments within their original time windows.

        Segments are sped up (up to max_speedup) to fit.  If even max speedup
        is not enough the audio is truncated with a fade-out so that subsequent
        segments stay aligned with the source video.

        When allow_extended_timing is True, segments can extend up to 500ms
        past their original end time, reducing the need for speedup when
        translation is naturally longer.
        """
        scheduled = []
        prev_end_ms = 0

        for seg in segments:
            if seg.tts_audio_path is None or not seg.tts_duration_ms or seg.tts_duration_ms <= 0:
                continue

            orig_start_ms = int(seg.start_time * 1000)
            orig_end_ms = int(seg.end_time * 1000)
            gap_from_prev = orig_start_ms - prev_end_ms

            if gap_from_prev >= self.silence_reset_gap * 1000:
                earliest_start = orig_start_ms
            else:
                earliest_start = max(orig_start_ms, prev_end_ms + self.min_gap_ms)

            # Allow extended timing: segment can go 500ms past original end
            max_end_ms = orig_end_ms + (500 if self.allow_extended_timing else 0)
            window_ms = max(max_end_ms - earliest_start, 1)

            if seg.tts_duration_ms <= window_ms:
                # Fits naturally — no speedup needed
                speed = 1.0
                actual_dur = seg.tts_duration_ms
            else:
                needed_speed = seg.tts_duration_ms / window_ms
                speed = min(needed_speed, self.max_speedup)
                # Constrain to window — truncate if max speedup isn't enough
                actual_dur = window_ms
                if needed_speed > self.max_speedup:
                    logger.warning(
                        "Segment %d needs %.1fx speedup (max %.1fx), will truncate "
                        "(%dms TTS → %dms window)",
                        seg.index, needed_speed, self.max_speedup,
                        seg.tts_duration_ms, window_ms,
                    )

            scheduled.append(ScheduledSegment(
                segment=seg,
                place_at_ms=earliest_start,
                speed_factor=speed,
                actual_duration_ms=actual_dur,
            ))
            prev_end_ms = earliest_start + actual_dur

        return scheduled

    def _load_background(self, total_samples: int) -> np.ndarray:
        """Load background audio and fit to total_samples length."""
        import librosa
        import soundfile as sf

        bg_audio, bg_sr = sf.read(str(self.background_path))
        bg_audio = bg_audio.astype(np.float32)

        if bg_audio.ndim == 1:
            bg_audio = np.column_stack([bg_audio, bg_audio])

        if bg_sr != self.sample_rate:
            bg_audio = librosa.resample(
                bg_audio.T, orig_sr=bg_sr, target_sr=self.sample_rate,
            ).T

        if len(bg_audio) >= total_samples:
            return bg_audio[:total_samples]

        buffer = np.zeros((total_samples, 2), dtype=np.float32)
        buffer[:len(bg_audio)] = bg_audio
        return buffer

    def render(self, scheduled: List[ScheduledSegment], total_duration_s: float) -> np.ndarray:
        """Render scheduled segments into a float32 stereo buffer, peak-normalized to 0.95.

        Uses librosa for pitch-preserving time-stretch and proper resampling.
        Segments that exceed their window are truncated with a 30ms fade-out.
        When a background track is set, TTS segments are layered on top with
        -12 dB ducking of the background during speech windows.
        """
        import librosa
        import soundfile as sf

        total_samples = int(total_duration_s * self.sample_rate)

        if self.background_path is not None:
            try:
                buffer = self._load_background(total_samples)
                logger.info("Loaded background audio from %s", self.background_path.name)
            except Exception as exc:
                logger.warning("Failed to load background audio: %s — using silence", exc)
                buffer = np.zeros((total_samples, 2), dtype=np.float32)
        else:
            buffer = np.zeros((total_samples, 2), dtype=np.float32)

        rendered = 0
        fade_samples = int(0.05 * self.sample_rate)  # 50ms crossfade for ducking

        for item in scheduled:
            try:
                audio_data, file_sr = sf.read(str(item.segment.tts_audio_path))
                mono = audio_data.astype(np.float32)

                # Convert to mono for processing
                if mono.ndim == 2:
                    mono = mono.mean(axis=1)

                # Resample to target rate
                if file_sr != self.sample_rate:
                    mono = librosa.resample(
                        mono, orig_sr=file_sr, target_sr=self.sample_rate,
                    )

                # Pitch-preserving time-stretch
                if item.speed_factor > 1.01:
                    mono = librosa.effects.time_stretch(mono, rate=item.speed_factor)

                # Per-segment RMS normalization for consistent volume
                rms = np.sqrt(np.mean(mono ** 2))
                if rms > 1e-6:
                    mono = mono * (0.12 / rms)  # target RMS ~0.12

                # Truncate to window with fade-out
                max_samples = int(item.actual_duration_ms / 1000 * self.sample_rate)
                if max_samples > 0 and len(mono) > max_samples:
                    fade_len = min(int(0.03 * self.sample_rate), max_samples // 4)
                    mono = mono[:max_samples]
                    if fade_len > 0:
                        mono[-fade_len:] *= np.linspace(
                            1.0, 0.0, fade_len, dtype=np.float32,
                        )

                # Mono -> stereo
                stereo = np.column_stack([mono, mono])

                # Place in buffer
                s = int(item.place_at_ms / 1000 * self.sample_rate)
                e = min(s + len(stereo), total_samples)

                if e > s:
                    # Duck the background during TTS segment (-12 dB = 0.25)
                    if self.background_path is not None:
                        region_len = e - s
                        envelope = np.full(region_len, 0.25, dtype=np.float32)
                        fl = min(fade_samples, region_len // 4)
                        if fl > 0:
                            # Ramp down 1.0→0.25 at start, ramp up 0.25→1.0 at end
                            envelope[:fl] = np.linspace(1.0, 0.25, fl, dtype=np.float32)
                            envelope[-fl:] = np.linspace(0.25, 1.0, fl, dtype=np.float32)
                        buffer[s:e] *= envelope[:, np.newaxis]

                    buffer[s:e] += stereo[:e - s]

                rendered += 1
            except Exception as exc:
                logger.warning("Render failed for segment %d: %s", item.segment.index, exc)

        logger.info("Rendered %d/%d segments", rendered, len(scheduled))

        peak = np.max(np.abs(buffer))
        if peak > 0.01:
            buffer = buffer / peak * 0.95

        return buffer

    def render_to_file(
        self,
        scheduled: List[ScheduledSegment],
        total_duration_s: float,
        output_path: Path,
    ) -> Path:
        """Render scheduled segments and write to a WAV file."""
        import soundfile as sf

        buffer = self.render(scheduled, total_duration_s)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), buffer, self.sample_rate)
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 ** 2)
            logger.info("Dubbed audio saved: %s (%.1f MB)", output_path, size_mb)
        return output_path
