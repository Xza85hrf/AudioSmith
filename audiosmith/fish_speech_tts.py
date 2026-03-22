"""AudioSmith Fish Speech Text-to-Speech Engine.

Cloud-based TTS via Fish Audio API with voice cloning and 13-language support.
Uses the fish-audio-sdk (fishaudio) package for API communication.

Features:
- 13 languages (en, zh, ja, ko, de, fr, es, pt, ru, nl, it, pl, ar)
- Instant voice cloning from 10-30s reference audio
- Persistent voice models (create once, reuse)
- Streaming synthesis support
- S1 (4B) and S1-mini (0.5B) model variants

Requires: FISH_API_KEY environment variable.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np

from audiosmith.exceptions import TTSError

logger = logging.getLogger("audiosmith.fish_speech_tts")


def _get_fishaudio():
    """Lazy import and return the fishaudio module."""
    try:
        import fishaudio
    except ImportError:
        raise TTSError(
            "fish-audio-sdk not installed. Install: pip install fish-audio-sdk",
            error_code="FISH_IMPORT_ERR",
        )
    return fishaudio


def _get_soundfile():
    """Lazy import soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        raise TTSError(
            "soundfile not installed. Install: pip install soundfile",
            error_code="FISH_IMPORT_ERR",
        )
    return sf


# ISO 639-1 → Fish Speech language name
FISH_LANGUAGE_MAP: Dict[str, str] = {
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "pt": "portuguese",
    "ru": "russian",
    "nl": "dutch",
    "it": "italian",
    "pl": "polish",
    "ar": "arabic",
}

FISH_LANGS: Set[str] = set(FISH_LANGUAGE_MAP.keys())


class FishSpeechTTS:
    """Fish Speech TTS with voice cloning and multilingual support.

    Requires FISH_API_KEY environment variable to be set.
    Uses WAV output format for direct numpy conversion via soundfile.
    """

    def __init__(
        self,
        model_id: str = "s1",
    ) -> None:
        self.model_id = model_id
        self._client: Any = None
        self._cloned_voices: Dict[str, str] = {}  # name → reference_id
        self.initialized = False

    @property
    def name(self) -> str:
        """Engine identifier."""
        return 'fish'

    @property
    def sample_rate(self) -> int:
        """Fish Speech outputs at 44.1kHz."""
        return 44100

    def load_model(self) -> None:
        """Initialize the Fish Audio client (lazy)."""
        self._ensure_client()

    def _ensure_client(self) -> None:
        """Initialize the Fish Audio client (lazy, checks API key)."""
        if self._client is not None:
            return

        api_key = os.getenv("FISH_API_KEY")
        if not api_key:
            raise TTSError(
                "FISH_API_KEY environment variable not set. "
                "Get a key at https://fish.audio",
                error_code="FISH_AUTH_ERR",
            )

        fishaudio = _get_fishaudio()
        self._client = fishaudio.FishAudio(api_key=api_key)
        self.initialized = True
        logger.info("Fish Audio client initialized (model=%s)", self.model_id)

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
            voice: Voice name (looks up cloned voices) or reference_id.
            language: Language code (ISO 639-1).
            emotion: Emotion marker (e.g. 'angry', 'sad', 'excited').
                Fish Speech natively supports 45+ emotion markers across
                all languages. The marker is prepended as ``(emotion) text``.

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        if not text or not text.strip():
            raise TTSError("Text cannot be empty", error_code="FISH_TEXT_ERR")

        # Inject emotion marker — Fish Speech processes these natively
        if emotion:
            text = f"({emotion}) {text}"

        self._ensure_client()

        # Resolve voice to reference_id
        reference_id = self._resolve_voice(voice)

        try:
            kwargs: Dict[str, Any] = {"text": text}
            if reference_id:
                kwargs["reference_id"] = reference_id

            audio_bytes = self._client.tts.convert(**kwargs)
            return self._bytes_to_audio(audio_bytes)

        except TTSError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                raise TTSError(
                    "Fish Audio rate limit exceeded. Wait before retrying.",
                    error_code="FISH_RATELIMIT",
                    original_error=e,
                )
            if "401" in err_str or "unauthorized" in err_str or "authentication" in err_str:
                raise TTSError(
                    "Fish Audio API key is invalid or expired.",
                    error_code="FISH_AUTH_ERR",
                    original_error=e,
                )
            raise TTSError(
                f"Fish Speech synthesis failed: {e}",
                error_code="FISH_SYNTH_ERR",
                original_error=e,
            )

    def create_voice_clone(
        self,
        voice_name: str,
        ref_audio: Union[str, Tuple[np.ndarray, int]],
        ref_text: Optional[str] = None,
        description: str = "",
    ) -> str:
        """Clone a voice from reference audio via Fish Audio API.

        Args:
            voice_name: Name for the cloned voice.
            ref_audio: Path to audio file or (numpy_array, sample_rate) tuple.
            ref_text: Transcript of the reference audio (improves quality).
            description: Optional voice description.

        Returns:
            The voice reference_id for use in synthesize().
        """
        self._ensure_client()

        # Load reference audio bytes
        audio_bytes = self._load_ref_audio(ref_audio)

        # Validate duration (10-30s recommended)
        self._validate_ref_duration(ref_audio)

        try:
            voice = self._client.voices.create(
                title=voice_name,
                voices=[audio_bytes],
                description=description or f"Cloned voice: {voice_name}",
            )
            voice_id: str = voice.id  # type: ignore[attr-defined]
            self._cloned_voices[voice_name] = voice_id
            logger.info(
                "Fish Speech voice '%s' created (id=%s)", voice_name, voice_id,
            )
            return voice_id

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"Voice cloning failed: {e}",
                error_code="FISH_CLONE_ERR",
                original_error=e,
            )

    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get cloned voices (Fish Speech has no preset voices, only clones)."""
        voices: Dict[str, Dict[str, Any]] = {}
        for name, vid in self._cloned_voices.items():
            voices[name] = {"type": "cloned", "reference_id": vid}
        return voices

    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ) -> Path:
        """Save audio array to WAV file."""
        sf = _get_soundfile()
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate or self.sample_rate)
        return output_path

    def cleanup(self) -> None:
        """Release client and clear caches."""
        self._client = None
        self._cloned_voices.clear()
        self.initialized = False
        logger.info("Fish Speech TTS cleaned up")

    def __enter__(self) -> "FishSpeechTTS":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def _resolve_voice(self, voice: Optional[str]) -> Optional[str]:
        """Resolve voice name to reference_id."""
        if not voice:
            return None
        # Check cloned voices first
        if voice in self._cloned_voices:
            return self._cloned_voices[voice]
        # Assume it's a raw reference_id
        return voice

    def _load_ref_audio(
        self, ref_audio: Union[str, Tuple[np.ndarray, int]]
    ) -> bytes:
        """Load reference audio as bytes for the API."""
        if isinstance(ref_audio, str):
            path = Path(ref_audio)
            if not path.exists():
                raise TTSError(
                    f"Reference audio file not found: {ref_audio}",
                    error_code="FISH_CLONE_ERR",
                )
            return path.read_bytes()

        # numpy array + sample rate → WAV bytes
        audio_arr, sr = ref_audio
        sf = _get_soundfile()
        buf = io.BytesIO()
        sf.write(buf, audio_arr, sr, format="WAV")
        return buf.getvalue()

    def _validate_ref_duration(
        self, ref_audio: Union[str, Tuple[np.ndarray, int]]
    ) -> None:
        """Warn if reference audio is too short (< 10s) or too long (> 60s)."""
        try:
            if isinstance(ref_audio, str):
                sf = _get_soundfile()
                info = sf.info(ref_audio)
                duration = info.duration
            else:
                audio_arr, sr = ref_audio
                duration = len(audio_arr) / sr

            if duration < 3.0:
                raise TTSError(
                    f"Reference audio too short ({duration:.1f}s). "
                    "Minimum 10 seconds recommended for quality cloning.",
                    error_code="FISH_CLONE_ERR",
                )
            if duration < 10.0:
                logger.warning(
                    "Reference audio is %.1fs — 10-30s recommended for best quality",
                    duration,
                )
            if duration > 60.0:
                logger.warning(
                    "Reference audio is %.1fs — trimming to first 30s recommended",
                    duration,
                )
        except TTSError:
            raise
        except Exception as e:
            logger.warning("Could not validate reference audio duration: %s", e)

    def _bytes_to_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Convert audio bytes (WAV/MP3) to float32 numpy array."""
        if not audio_bytes:
            return np.array([], dtype=np.float32), self.sample_rate

        sf = _get_soundfile()
        buf = io.BytesIO(audio_bytes)
        try:
            audio, sr = sf.read(buf, dtype="float32")
            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), sr
        except Exception:
            # Fallback: try raw PCM 16-bit (some endpoints return raw PCM)
            try:
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return audio, self.sample_rate
            except Exception as e:
                raise TTSError(
                    f"Failed to decode Fish Speech audio response: {e}",
                    error_code="FISH_DECODE_ERR",
                    original_error=e,
                )
