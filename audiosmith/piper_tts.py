"""Piper TTS engine — lightweight ONNX-based TTS with Polish support."""

import gc
import logging
from pathlib import Path
from typing import Any, List, Optional

from audiosmith.exceptions import TTSError

logger = logging.getLogger(__name__)

POLISH_VOICES = ["pl_PL-gosia-medium", "pl_PL-aleksandra-medium"]
ENGLISH_VOICES = ["en_US-lessac-medium", "en_US-amy-medium", "en_GB-alba-medium"]


class PiperTTS:
    """Piper TTS — CPU-friendly ONNX synthesis with Polish voice support."""

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        model_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ) -> None:
        self.voice = voice
        self.model_path = model_path
        self.data_path = data_path
        self._model: Optional[Any] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from piper import PiperVoice
        except ImportError:
            raise TTSError("piper not installed. Install: pip install piper-tts")
        if self.model_path is None:
            raise TTSError("model_path required to load Piper model")
        logger.info("Loading Piper model: %s", self.model_path)
        self._model = PiperVoice.load(str(self.model_path))

    @property
    def sample_rate(self) -> int:
        return 22050

    def synthesize(self, text: str, voice: Optional[str] = None) -> Any:
        """Synthesize text to audio numpy array (float32, mono)."""
        if not text or not text.strip():
            raise TTSError("Text cannot be empty")
        if voice and voice != self.voice:
            self.voice = voice
            self._model = None
        self._load_model()

        import numpy as np

        chunks = list(self._model.synthesize(text))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        return audio

    def list_voices(self) -> List[str]:
        """Return available voice identifiers."""
        voices = list(POLISH_VOICES + ENGLISH_VOICES)
        if self.data_path and self.data_path.is_dir():
            for f in self.data_path.glob("*.onnx"):
                voices.append(f.stem)
        return sorted(set(voices))

    def cleanup(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            logger.info("Piper TTS model unloaded")
