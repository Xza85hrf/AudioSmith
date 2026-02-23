"""Data models for the AudioSmith 6-step dubbing pipeline."""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class DubbingStep(Enum):
    """Pipeline step identifiers."""
    EXTRACT_AUDIO = 'extract_audio'
    TRANSCRIBE = 'transcribe'
    TRANSLATE = 'translate'
    GENERATE_TTS = 'generate_tts'
    MIX_AUDIO = 'mix_audio'
    ENCODE_VIDEO = 'encode_video'


@dataclass
class DubbingConfig:
    """Configuration for the dubbing pipeline."""
    video_path: Path
    output_dir: Path
    source_language: str = 'en'
    target_language: str = 'pl'
    max_duration: Optional[float] = None
    min_gap_ms: int = 150
    max_speedup: float = 1.5
    silence_reset_gap: float = 1.0
    dubbed_sample_rate: int = 48000
    whisper_model: str = 'large-v3'
    whisper_compute_type: str = 'float16'
    whisper_device: str = 'cuda'
    audio_prompt_path: Optional[Path] = None
    chatterbox_exaggeration: float = 0.5
    chatterbox_cfg_weight: float = 0.5
    burn_subtitles: bool = True
    resume: bool = False


@dataclass
class DubbingSegment:
    """Represents a segment to be processed in the dubbing pipeline."""
    index: int
    start_time: float
    end_time: float
    original_text: str
    translated_text: str = ''
    speaker_id: Optional[str] = None
    is_speech: bool = True
    is_hallucination: bool = False
    tts_audio_path: Optional[Path] = None
    tts_duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        """Return the segment duration in milliseconds."""
        return int((self.end_time - self.start_time) * 1000)


@dataclass
class ScheduledSegment:
    """A segment scheduled for placement with timing adjustments."""
    segment: DubbingSegment
    place_at_ms: int
    speed_factor: float = 1.0
    actual_duration_ms: int = 0


@dataclass
class DubbingResult:
    """Result of a dubbing pipeline run."""
    success: bool
    output_video_path: Optional[Path] = None
    dubbed_audio_path: Optional[Path] = None
    subtitle_path: Optional[Path] = None
    total_segments: int = 0
    segments_dubbed: int = 0
    segments_skipped: int = 0
    segments_failed: int = 0
    step_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineState:
    """Checkpoint state for pipeline resume support."""
    completed_steps: List[str] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    audio_path: Optional[str] = None
    dubbed_audio_path: Optional[str] = None
    subtitle_path: Optional[str] = None

    def save(self, path: Path) -> None:
        """Persist state to a JSON file."""
        data = asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'PipelineState':
        """Load state from a JSON checkpoint file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def is_step_done(self, step: str) -> bool:
        return step in self.completed_steps

    def mark_step_done(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
