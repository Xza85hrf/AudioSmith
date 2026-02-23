"""Sequential audio scheduling, drift correction, and buffer rendering for dubbing."""

import logging
from pathlib import Path
from typing import List

import numpy as np

from audiosmith.models import DubbingSegment, ScheduledSegment, DubbingConfig

logger = logging.getLogger(__name__)


class AudioMixer:
    """Schedules TTS segments sequentially with drift correction, then renders to a numpy buffer."""

    def __init__(self, config: DubbingConfig):
        self.min_gap_ms = config.min_gap_ms
        self.max_speedup = config.max_speedup
        self.silence_reset_gap = config.silence_reset_gap
        self.sample_rate = config.dubbed_sample_rate

    def schedule(self, segments: List[DubbingSegment]) -> List[ScheduledSegment]:
        """Build a sequential schedule with drift correction."""
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

            remaining_ms = max(orig_end_ms - earliest_start, 1)
            speed = 1.0
            if seg.tts_duration_ms > remaining_ms:
                speed = min(seg.tts_duration_ms / remaining_ms, self.max_speedup)

            actual_dur = int(seg.tts_duration_ms / speed)
            scheduled.append(ScheduledSegment(
                segment=seg,
                place_at_ms=earliest_start,
                speed_factor=speed,
                actual_duration_ms=actual_dur,
            ))
            prev_end_ms = earliest_start + actual_dur

        return scheduled

    def render(self, scheduled: List[ScheduledSegment], total_duration_s: float) -> np.ndarray:
        """Render scheduled segments into a float32 stereo buffer, peak-normalized to 0.95."""
        import soundfile as sf

        total_samples = int(total_duration_s * self.sample_rate)
        buffer = np.zeros((total_samples, 2), dtype=np.float32)
        rendered = 0

        for item in scheduled:
            try:
                audio_data, file_sr = sf.read(str(item.segment.tts_audio_path))
                samples = audio_data.astype(np.float32)

                # Mono -> stereo
                if samples.ndim == 1:
                    samples = np.column_stack([samples, samples])
                elif samples.ndim == 2 and samples.shape[1] == 1:
                    samples = np.column_stack([samples[:, 0], samples[:, 0]])

                # Speed up if needed
                if item.speed_factor > 1.01:
                    new_len = int(len(samples) / item.speed_factor)
                    if new_len > 0:
                        indices = np.linspace(0, len(samples) - 1, new_len).astype(int)
                        samples = samples[indices]

                # Resample if needed
                if file_sr != self.sample_rate:
                    new_len = int(len(samples) * self.sample_rate / file_sr)
                    if new_len > 0:
                        indices = np.linspace(0, len(samples) - 1, new_len).astype(int)
                        samples = samples[indices]

                # Place in buffer
                s = int(item.place_at_ms / 1000 * self.sample_rate)
                e = min(s + len(samples), total_samples)
                if e > s:
                    buffer[s:e] = samples[:e - s]
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
