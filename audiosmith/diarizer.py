"""Speaker diarization using pyannote.audio — lean port for AudioSmith."""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from audiosmith.exceptions import DiarizationError
from audiosmith.error_codes import ErrorCode

logger = logging.getLogger(__name__)


class Diarizer:
    """Speaker diarization using pyannote/speaker-diarization-3.1.

    Lazy-loads the model on first use. Supports GPU with automatic
    fallback to CPU when CUDA is unavailable.
    """

    def __init__(
        self,
        device: str = 'auto',
        hf_token: Optional[str] = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        merge_threshold: float = 0.5,
    ):
        self.device = device
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.merge_threshold = merge_threshold
        self._pipeline = None

    def _resolve_device(self) -> str:
        """Resolve 'auto' to the best available device."""
        if self.device != 'auto':
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'

    def load_model(self) -> None:
        """Lazy-load the pyannote speaker diarization pipeline."""
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise DiarizationError(
                "pyannote-audio not installed. Run: pip install 'audiosmith[quality]'",
                error_code=str(ErrorCode.MODEL_LOAD_ERROR.value),
                original_error=e,
            )

        if not self.hf_token:
            raise DiarizationError(
                "HuggingFace token required for pyannote models. "
                "Set HF_TOKEN env var or pass hf_token parameter.",
                error_code=str(ErrorCode.MODEL_LOAD_ERROR.value),
            )

        device = self._resolve_device()
        t0 = time.time()

        try:
            self._pipeline = Pipeline.from_pretrained(
                'pyannote/speaker-diarization-3.1',
                use_auth_token=self.hf_token,
            )

            import torch
            self._pipeline = self._pipeline.to(torch.device(device))

            # GPU optimizations for Volta+ (compute capability >= 7.0)
            if device.startswith('cuda'):
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 7:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            logger.info(
                "Diarization pipeline loaded on %s in %.1fs",
                device, time.time() - t0,
            )
        except DiarizationError:
            raise
        except Exception as e:
            raise DiarizationError(
                f"Failed to load diarization pipeline: {e}",
                error_code=str(ErrorCode.MODEL_LOAD_ERROR.value),
                original_error=e,
            )

    def diarize(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Run speaker diarization on an audio file.

        Returns list of dicts: {speaker, start, end, duration, confidence}.
        Segments are sorted by start time and merged within merge_threshold.
        """
        self.load_model()
        t0 = time.time()
        audio_path = Path(audio_path)

        try:
            diarization = self._pipeline(
                str(audio_path),
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
        except Exception as e:
            raise DiarizationError(
                f"Diarization inference failed: {e}",
                error_code=str(ErrorCode.MODEL_INFERENCE_ERROR.value),
                original_error=e,
            )

        # Extract segments from pyannote Annotation
        segments: List[Dict[str, Any]] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'speaker': speaker,
                'start': float(segment.start),
                'end': float(segment.end),
                'duration': float(segment.duration),
                'confidence': 1.0,
            })

        segments.sort(key=lambda s: s['start'])
        merged = self._merge_segments(segments, self.merge_threshold)

        speakers = set(s['speaker'] for s in merged)
        logger.info(
            "Diarized %s: %d segments, %d speakers in %.1fs",
            audio_path.name, len(merged), len(speakers), time.time() - t0,
        )
        return merged

    @staticmethod
    def _merge_segments(
        segments: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Merge consecutive segments from the same speaker within threshold seconds."""
        if not segments:
            return []

        merged: List[Dict[str, Any]] = []
        current = segments[0].copy()

        for seg in segments[1:]:
            if (
                seg['speaker'] == current['speaker']
                and seg['start'] - current['end'] <= threshold
            ):
                current['end'] = seg['end']
                current['duration'] = current['end'] - current['start']
                current['confidence'] = (current['confidence'] + seg['confidence']) / 2
            else:
                merged.append(current)
                current = seg.copy()

        merged.append(current)
        return merged

    @staticmethod
    def apply_to_transcription(
        transcription_segments: List[Dict[str, Any]],
        diarization_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assign speaker IDs to transcription segments by maximum timing overlap.

        Each transcription segment gets a 'speaker' key set to the diarization
        speaker with the largest temporal overlap. Segments with no overlap
        get speaker=None.
        """
        for tseg in transcription_segments:
            t_start = tseg['start']
            t_end = tseg['end']
            best_speaker: Optional[str] = None
            best_overlap: float = 0.0

            for dseg in diarization_segments:
                overlap_start = max(t_start, dseg['start'])
                overlap_end = min(t_end, dseg['end'])
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = dseg['speaker']

            tseg['speaker'] = best_speaker

        return transcription_segments

    def unload(self) -> None:
        """Release model and GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
