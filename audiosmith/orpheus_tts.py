"""AudioSmith Orpheus Text-to-Speech Engine.

Local TTS engine with the most expressive emotion control via inline tags.
Supports 13 languages, 8 preset voices, and streaming audio generation.

Features:
- 13 languages (en, zh, es, fr, de, it, pt, hi, ko, tr, ja, th, ar)
- 8 preset voices (tara, leah, jess, leo, dan, mia, zac, zoe)
- Inline emotion tags (<laugh>, <sigh>, <gasp>, <cough>, <groan>, <yawn>)
- Streaming audio chunk generation
- 3B parameter model, ~15GB VRAM

Requires: pip install orpheus-speech
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from audiosmith.exceptions import TTSError

logger = logging.getLogger("audiosmith.orpheus_tts")


def _get_orpheus():
    """Lazy import and return OrpheusModel class."""
    try:
        from orpheus_tts import OrpheusModel
    except ImportError:
        raise TTSError(
            "orpheus-speech not installed. Install: pip install orpheus-speech",
            error_code="ORPH_IMPORT_ERR",
        )
    return OrpheusModel


ORPHEUS_LANGS: Set[str] = {
    'en', 'zh', 'es', 'fr', 'de', 'it', 'pt', 'hi', 'ko', 'tr', 'ja', 'th', 'ar',
}

ORPHEUS_VOICES: List[str] = [
    'tara', 'leah', 'jess', 'leo', 'dan', 'mia', 'zac', 'zoe',
]

# Map AudioSmith emotion labels → Orpheus inline tags
ORPHEUS_EMOTION_TAGS: Dict[str, str] = {
    'happy': '<laugh>',
    'excited': '<laugh>',
    'sad': '<sigh>',
    'fearful': '<gasp>',
    'surprised': '<gasp>',
    'angry': '<groan>',
    'tired': '<yawn>',
    'sick': '<cough>',
}


class OrpheusTTS:
    """Orpheus TTS with expressive emotion tags and preset voices.

    Uses the orpheus-speech package which requires a vLLM backend.
    ~15GB VRAM required for the 3B model.
    """

    def __init__(
        self,
        model_name: str = 'canopylabs/orpheus-3b-0.1-ft',
        voice: str = 'tara',
        temperature: float = 0.7,
    ) -> None:
        if voice not in ORPHEUS_VOICES:
            raise TTSError(
                f"Unknown voice '{voice}'. Available: {ORPHEUS_VOICES}",
                error_code="ORPH_VOICE_ERR",
            )
        self.model_name = model_name
        self.default_voice = voice
        self.temperature = temperature
        self._model: Any = None
        self._cloned_voices: Dict[str, Path] = {}
        self.initialized = False

    @property
    def sample_rate(self) -> int:
        """Orpheus outputs at 24000 Hz."""
        return 24000

    def _ensure_model(self) -> None:
        """Load the Orpheus model (lazy)."""
        if self._model is not None:
            return

        try:
            OrpheusModel = _get_orpheus()
            self._model = OrpheusModel(model_name=self.model_name)
            self.initialized = True
            logger.info("Orpheus model loaded: %s", self.model_name)
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"Failed to load Orpheus model: {e}. "
                "Orpheus requires ~15GB VRAM and a vLLM backend.",
                error_code="ORPH_MODEL_ERR",
                original_error=e,
            )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Preset voice name (default: 'tara').
            language: Language code (ISO 639-1).
            emotion: Emotion label to inject as inline tag.

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        if not text or not text.strip():
            raise TTSError("Text cannot be empty", error_code="ORPH_SYNTH_ERR")

        if language and language not in ORPHEUS_LANGS:
            raise TTSError(
                f"Language '{language}' not supported by Orpheus. "
                f"Supported: {sorted(ORPHEUS_LANGS)}",
                error_code="ORPH_LANG_ERR",
            )

        # Inject emotion tag at the start of text
        if emotion:
            tag = ORPHEUS_EMOTION_TAGS.get(emotion)
            if tag:
                text = f"{tag} {text}"

        voice_name = voice or self.default_voice
        if voice_name in self._cloned_voices:
            # For cloned voices, we still use a preset voice but could
            # inject reference context in future API versions
            logger.debug("Using cloned voice '%s' (preset fallback)", voice_name)

        # Validate preset voice
        if voice_name not in ORPHEUS_VOICES and voice_name not in self._cloned_voices:
            raise TTSError(
                f"Voice '{voice_name}' not found. Available presets: {ORPHEUS_VOICES}",
                error_code="ORPH_VOICE_ERR",
            )

        self._ensure_model()

        try:
            output_iter = self._model.generate_speech(
                prompt=text,
                voice=voice_name if voice_name in ORPHEUS_VOICES else self.default_voice,
            )
            audio = self._collect_chunks(output_iter)
            return audio, self.sample_rate

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"Orpheus synthesis failed: {e}",
                error_code="ORPH_SYNTH_ERR",
                original_error=e,
            )

    def create_voice_clone(
        self,
        voice_name: str,
        ref_audio: Union[str, Path],
    ) -> str:
        """Register a reference audio for voice context.

        Note: Orpheus primarily uses preset voices. Voice cloning stores
        the reference path for potential future prompt-context injection.

        Args:
            voice_name: Name for the voice.
            ref_audio: Path to reference audio file.

        Returns:
            The voice name.
        """
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise TTSError(
                f"Reference audio not found: {ref_audio}",
                error_code="ORPH_VOICE_ERR",
            )
        self._cloned_voices[voice_name] = ref_path
        logger.info("Orpheus voice '%s' registered (ref: %s)", voice_name, ref_path.name)
        return voice_name

    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get all available voices (presets + cloned)."""
        voices: Dict[str, Dict[str, Any]] = {}
        for name in ORPHEUS_VOICES:
            voices[name] = {'type': 'preset'}
        for name in self._cloned_voices:
            voices[name] = {'type': 'cloned'}
        return voices

    def cleanup(self) -> None:
        """Release model and clear state."""
        if self._model is not None:
            del self._model
            self._model = None
        self._cloned_voices.clear()
        self.initialized = False
        logger.info("Orpheus TTS cleaned up")

    def __enter__(self) -> "OrpheusTTS":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.cleanup()
        return False

    def _collect_chunks(self, output_iter: Any) -> np.ndarray:
        """Collect streaming audio chunks into a single numpy array.

        Orpheus generate_speech() returns an iterator of audio chunks
        (bytes or numpy arrays depending on version).
        """
        chunks: List[np.ndarray] = []
        for chunk in output_iter:
            if isinstance(chunk, bytes):
                audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(chunk, np.ndarray):
                audio_np = chunk.astype(np.float32)
            else:
                # torch tensor or similar
                audio_np = np.array(chunk, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            chunks.append(audio_np)

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)
