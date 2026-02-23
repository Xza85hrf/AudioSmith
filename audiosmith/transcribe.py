"""Faster-Whisper transcriber — single class, no manager stack."""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from audiosmith.exceptions import TranscriptionError
from audiosmith.error_codes import ErrorCode

logger = logging.getLogger(__name__)


class Transcriber:
    """High-performance transcriber using faster-whisper with BatchedInferencePipeline."""

    SUPPORTED_MODELS = [
        'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
        'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3',
        'distil-large-v2', 'distil-large-v3',
    ]

    def __init__(
        self,
        model: str = 'large-v3',
        compute_type: str = 'float16',
        device: str = 'cuda',
        device_index: int = 0,
        batch_size: int = 16,
        vad_filter: bool = True,
    ):
        self.model = model
        self.compute_type = compute_type
        self.device = device
        self.device_index = device_index
        self.batch_size = batch_size
        self.vad_filter = vad_filter
        self._model = None
        self._batched = None

    def load_model(self) -> None:
        """Load the Faster-Whisper model with BatchedInferencePipeline."""
        if self._batched is not None:
            return

        try:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
        except ImportError as e:
            raise TranscriptionError(
                "faster-whisper not installed. Run: pip install faster-whisper",
                error_code=str(ErrorCode.MODEL_LOAD_ERROR.value),
                original_error=e,
            )

        try:
            device_str = "cuda" if self.device.startswith("cuda") else "cpu"
            self._model = WhisperModel(
                self.model,
                device=device_str,
                device_index=self.device_index,
                compute_type=self.compute_type,
            )
            self._batched = BatchedInferencePipeline(model=self._model)
            logger.info("Faster-Whisper model loaded: %s on %s", self.model, device_str)
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load Faster-Whisper model: {e}",
                error_code=str(ErrorCode.MODEL_LOAD_ERROR.value),
                original_error=e,
            )

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Transcribe audio file. Returns list of segment dicts with text/start/end/words."""
        self.load_model()
        t0 = time.time()

        lang = None if language in (None, 'auto') else language

        segments, info = self._batched.transcribe(
            str(audio_path),
            batch_size=self.batch_size,
            word_timestamps=True,
            language=lang,
            vad_filter=self.vad_filter,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

        result = []
        for seg in segments:
            entry = {
                'text': seg.text.strip(),
                'start': seg.start,
                'end': seg.end,
                'language': info.language,
                'words': [],
            }
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    entry['words'].append({
                        'text': getattr(w, 'word', '').strip(),
                        'start': getattr(w, 'start', 0.0),
                        'end': getattr(w, 'end', 0.0),
                        'confidence': getattr(w, 'probability', 0.0),
                    })
            result.append(entry)

        elapsed = time.time() - t0
        logger.info("Transcribed %d segments in %.1fs", len(result), elapsed)
        return result

    def unload(self) -> None:
        """Release model and GPU memory."""
        if self._batched is not None:
            del self._batched
        if self._model is not None:
            del self._model
        self._batched = self._model = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
