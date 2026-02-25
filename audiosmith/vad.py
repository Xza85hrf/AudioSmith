"""Voice Activity Detection using Silero VAD model."""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment in audio."""
    start: float   # seconds
    end: float     # seconds
    confidence: float = 1.0

    @property
    def duration_ms(self) -> int:
        return int((self.end - self.start) * 1000)


class VADProcessor:
    """Voice Activity Detection using Silero VAD."""

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        sample_rate: int = 16000,
    ) -> None:
        self.device = device
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        self._model: Optional[Any] = None
        self._get_speech_ts: Optional[Any] = None

    def _load_model(self) -> None:
        """Lazy-load Silero VAD model."""
        if self._model is not None:
            return
        import torch
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True,
        )
        self._model = model
        self._get_speech_ts = utils[0]  # get_speech_timestamps
        logger.info("Silero VAD model loaded")

    def detect_speech(self, audio_path: Path) -> List[SpeechSegment]:
        """Detect speech segments in audio file."""
        self._load_model()
        import torch

        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            audio_tensor = waveform.squeeze(0)
        except ImportError:
            import soundfile as sf
            import numpy as np
            data, sr = sf.read(str(audio_path), dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            audio_tensor = torch.from_numpy(data)

        timestamps = self._get_speech_ts(
            audio_tensor, self._model,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            threshold=self.threshold,
        )

        segments = []
        for ts in timestamps:
            segments.append(SpeechSegment(
                start=ts["start"] / self.sample_rate,
                end=ts["end"] / self.sample_rate,
                confidence=ts.get("confidence", 1.0),
            ))
        logger.info("Detected %d speech segments in %s", len(segments), audio_path.name)
        return segments

    def filter_silence(
        self, segments: List[Any], speech_regions: List[SpeechSegment],
    ) -> List[Any]:
        """Mark segments as non-speech if they don't overlap any speech region."""
        for seg in segments:
            seg.is_speech = any(
                seg.start_time < sr.end and seg.end_time > sr.start
                for sr in speech_regions
            )
        return segments

    def unload(self) -> None:
        """Free model from memory."""
        if self._model is not None:
            del self._model
            del self._get_speech_ts
            self._model = None
            self._get_speech_ts = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
