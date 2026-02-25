"""Voice extraction pipeline — extract, catalog, and manage voice samples for TTS cloning."""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from audiosmith.exceptions import ProcessingError

logger = logging.getLogger("audiosmith.voice_extractor")


@dataclass
class VoiceSample:
    """A voice sample extracted from an audio source."""

    speaker_id: str
    sample_path: Path
    source_file: Path
    start_time: float
    end_time: float
    mean_volume_db: float = 0.0
    description: str = ""
    language: str = "auto"
    created_at: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VoiceCatalog:
    """Collection of voice samples with speaker metadata."""

    samples: List[VoiceSample] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_sample(self, sample: VoiceSample):
        self.samples.append(sample)
        src = str(sample.source_file)
        if src not in self.source_files:
            self.source_files.append(src)

    def get_speakers(self) -> List[str]:
        return list(set(s.speaker_id for s in self.samples))

    def get_samples_for_speaker(self, speaker_id: str) -> List[VoiceSample]:
        return [s for s in self.samples if s.speaker_id == speaker_id]

    def get_best_sample(self, speaker_id: str) -> Optional[VoiceSample]:
        """Get the loudest non-clipping sample for a speaker.

        Filters out samples with mean volume above -3 dB (likely clipping),
        then returns the loudest remaining sample (closest to 0 dB).
        """
        samples = self.get_samples_for_speaker(speaker_id)
        if not samples:
            return None

        safe = [s for s in samples if s.mean_volume_db <= -3.0]
        if not safe:
            safe = samples

        # Sort by mean_volume_db descending (closest to 0 = loudest)
        safe.sort(key=lambda s: s.mean_volume_db, reverse=True)
        return safe[0]

    def save(self, path: Path):
        """Save catalog to JSON."""
        data = {
            "samples": [
                {
                    "speaker_id": s.speaker_id,
                    "sample_path": str(s.sample_path),
                    "source_file": str(s.source_file),
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "mean_volume_db": s.mean_volume_db,
                    "description": s.description,
                    "language": s.language,
                    "created_at": s.created_at,
                }
                for s in self.samples
            ],
            "source_files": self.source_files,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "VoiceCatalog":
        """Load catalog from JSON."""
        data = json.loads(path.read_text())
        samples = []
        for item in data.get("samples", []):
            item["sample_path"] = Path(item["sample_path"])
            item["source_file"] = Path(item["source_file"])
            samples.append(VoiceSample(**item))
        return cls(
            samples=samples,
            source_files=data.get("source_files", []),
            metadata=data.get("metadata", {}),
        )


class VoiceExtractor:
    """Extract voice samples from audio files for TTS voice cloning."""

    def __init__(
        self,
        output_dir: Path,
        sample_duration: float = 5.0,
        sample_rate: int = 24000,
        min_volume_db: float = -30.0,
    ):
        self.output_dir = Path(output_dir)
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate
        self.min_volume_db = min_volume_db
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_with_diarization(
        self, audio_path: Path, num_speakers: Optional[int] = None,
    ) -> VoiceCatalog:
        """Extract voice samples using speaker diarization."""
        from audiosmith.diarizer import Diarizer

        diarizer = Diarizer()
        segments = diarizer.diarize(audio_path, num_speakers=num_speakers)

        speaker_segs: Dict[str, List[Tuple[float, float]]] = {}
        for seg in segments:
            sid = seg.get("speaker", "unknown")
            speaker_segs.setdefault(sid, []).append((seg["start"], seg["end"]))

        catalog = VoiceCatalog()
        catalog.source_files.append(str(audio_path))

        for speaker_id, segs in speaker_segs.items():
            best_seg = None
            best_vol = -999.0
            for start, end in segs:
                if (end - start) < 1.0:
                    continue
                temp_path = self._extract_segment(audio_path, start, end, f"_tmp_{speaker_id}")
                vol = self._measure_volume(temp_path)
                temp_path.unlink(missing_ok=True)
                if vol > self.min_volume_db and vol > best_vol and vol <= -3.0:
                    best_seg = (start, end)
                    best_vol = vol

            if best_seg:
                s, e = best_seg
                path = self._extract_segment(audio_path, s, e, f"{speaker_id}_best")
                catalog.add_sample(VoiceSample(
                    speaker_id=speaker_id, sample_path=path,
                    source_file=audio_path, start_time=s, end_time=e,
                    mean_volume_db=best_vol,
                ))

        return catalog

    def extract_at_intervals(
        self,
        audio_path: Path,
        intervals: List[Tuple[float, float]],
        speaker_ids: Optional[List[str]] = None,
    ) -> VoiceCatalog:
        """Extract voice samples at specified time intervals."""
        catalog = VoiceCatalog()
        catalog.source_files.append(str(audio_path))

        if speaker_ids is None:
            speaker_ids = [f"speaker_{i:02d}" for i in range(len(intervals))]

        for i, (start, end) in enumerate(intervals):
            sid = speaker_ids[i] if i < len(speaker_ids) else f"speaker_{i:02d}"
            name = f"{sid}_{start:.0f}_{end:.0f}"
            try:
                path = self._extract_segment(audio_path, start, end, name)
                vol = self._measure_volume(path)
                catalog.add_sample(VoiceSample(
                    speaker_id=sid, sample_path=path, source_file=audio_path,
                    start_time=start, end_time=end, mean_volume_db=vol,
                ))
            except ProcessingError as e:
                logger.warning("Skipping interval %.1f-%.1f: %s", start, end, e)

        return catalog

    def extract_evenly(
        self, audio_path: Path, num_samples: int = 5, speaker_prefix: str = "voice",
    ) -> VoiceCatalog:
        """Extract evenly-spaced samples from the middle of each segment."""
        total = self._get_duration(audio_path)
        seg_len = total / num_samples

        intervals = []
        for i in range(num_samples):
            mid = (i + 0.5) * seg_len
            s = max(0, mid - self.sample_duration / 2)
            e = min(total, mid + self.sample_duration / 2)
            intervals.append((s, e))

        ids = [f"{speaker_prefix}_{i:02d}" for i in range(num_samples)]
        return self.extract_at_intervals(audio_path, intervals, ids)

    def _extract_segment(self, audio_path: Path, start: float, end: float, output_name: str) -> Path:
        """Extract audio segment using ffmpeg."""
        out = self.output_dir / f"{output_name}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start), "-t", str(end - start),
            "-ar", str(self.sample_rate), "-ac", "1", str(out),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise ProcessingError(f"FFmpeg extraction failed: {e.stderr}")
        return out

    def _measure_volume(self, audio_path: Path) -> float:
        """Measure mean volume using ffmpeg volumedetect."""
        cmd = ["ffmpeg", "-i", str(audio_path), "-af", "volumedetect", "-f", "null", "-"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in result.stderr.splitlines():
                if "mean_volume" in line:
                    return float(line.split("mean_volume:")[1].split("dB")[0].strip())
        except (subprocess.CalledProcessError, ValueError, IndexError):
            pass
        return -60.0

    def _get_duration(self, audio_path: Path) -> float:
        """Get audio duration using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            raise ProcessingError(f"Could not get duration: {e}")


def create_voice_profiles(catalog: VoiceCatalog) -> Dict[str, Any]:
    """Create voice profile dict from catalog for use with Qwen3TTS.create_voice_clone()."""
    profiles = {}
    for speaker_id in catalog.get_speakers():
        best = catalog.get_best_sample(speaker_id)
        if best:
            profiles[speaker_id] = {
                "sample_path": str(best.sample_path),
                "description": best.description,
                "language": best.language,
            }
    return profiles
