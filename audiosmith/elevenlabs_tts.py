"""AudioSmith ElevenLabs Text-to-Speech Engine.

Cloud-based TTS via ElevenLabs API with voice cloning, preset voices,
and 70+ language support across multiple models.

Features:
- Preset voices (Rachel, Adam, Bella, etc.) with human-friendly names
- Instant voice cloning from audio samples
- Configurable models: eleven_v3, eleven_multilingual_v2, eleven_flash_v2_5
- Streaming synthesis support
- 70+ languages with eleven_v3

Requires: ELEVENLABS_API_KEY environment variable.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np

from audiosmith.exceptions import TTSError

# Lazy imports for ElevenLabs SDK
_elevenlabs_client = None

logger = logging.getLogger("audiosmith.elevenlabs_tts")


def _get_elevenlabs():
    """Lazy import and return the ElevenLabs client class."""
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        raise TTSError(
            "elevenlabs not installed. Install: pip install elevenlabs",
            error_code="ELEV_IMPORT_ERR",
        )
    return ElevenLabs


def _get_soundfile():
    """Lazy import soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        raise TTSError(
            "soundfile not installed. Install: pip install soundfile",
            error_code="ELEV_IMPORT_ERR",
        )
    return sf


# Human-friendly name → ElevenLabs voice UUID
VOICE_MAP: Dict[str, str] = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Antoni": "ErXwobaYiN019PkySvjV",
    "Arnold": "VR6AewLTigWG4xSOukaG",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
    "Charlotte": "XB0fDUnXU5powFXDhCwa",
    "Clyde": "2EiwWnXFnvU5JabPnv8n",
    "Daniel": "onwK4e9ZLuTAKqWW03F9",
    "Dave": "CYw3kZ02Hs0563khs1Fj",
    "Domi": "AZnzlk1XvdvUeBnXmlld",
    "Dorothy": "ThT5KcBeYPX3keUQqHPh",
    "Elli": "MF3mGyEYCl7XYWbV9V6O",
    "Emily": "LcfcDJNUP1GQjkzn1xUU",
    "Ethan": "g5CIjZEefAph4nQFvHAz",
    "Fin": "D38z5RcWu1voky8WS1ja",
    "Freya": "jsCqWAovK2LkecY7zXl4",
    "Gigi": "jBpfuIE2acCO8z3wKNLl",
    "Giovanni": "zcAOhNBS3c14rBihAFp1",
    "Glinda": "z9fAnlkpzviPz146aGWa",
    "Grace": "oWAxZDx7w5VEj9dCyTzz",
    "Harry": "SOYHLrjzK2X1ezoPC6cr",
    "James": "ZQe5CZNOzWyzPSCn5a3c",
}

ELEVENLABS_MODELS: Dict[str, str] = {
    "eleven_v3": "Most expressive, 70+ languages, best quality",
    "eleven_multilingual_v2": "29 languages, proven reliability",
    "eleven_flash_v2_5": "Ultra-low ~75ms latency, 32 languages",
    "eleven_turbo_v2_5": "Speed-balanced, 32 languages",
}

# Languages supported by eleven_v3 / eleven_multilingual_v2 (ISO 639-1 codes)
ELEVENLABS_LANGS: Set[str] = {
    "en", "es", "de", "fr", "it", "pt", "pl", "ru", "ja", "ko", "zh",
    "ar", "bg", "cs", "da", "nl", "fi", "el", "hi", "hu", "id", "ms",
    "no", "ro", "sk", "sv", "ta", "th", "tr", "uk", "vi", "hr", "sl",
    "fil", "sw", "he", "ca", "cy", "mk", "sr", "lt", "lv", "et",
}


class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech with voice cloning and preset voices.

    Requires ELEVENLABS_API_KEY environment variable to be set.
    Uses PCM output format for direct numpy conversion (no MP3 decoding).
    """

    def __init__(
        self,
        model_id: str = "eleven_v3",
        voice_id: Optional[str] = None,
        voice_name: Optional[str] = None,
        output_format: str = "pcm_24000",
    ) -> None:
        if model_id not in ELEVENLABS_MODELS:
            raise TTSError(
                f"Unknown model: {model_id}. "
                f"Valid: {list(ELEVENLABS_MODELS.keys())}",
                error_code="ELEV_MODEL_ERR",
            )

        self.model_id = model_id
        self.output_format = output_format
        self._client: Any = None
        self._cloned_voices: Dict[str, str] = {}
        self.initialized = False

        # Resolve voice — name takes precedence over raw ID
        if voice_name:
            resolved = VOICE_MAP.get(voice_name)
            if not resolved:
                raise TTSError(
                    f"Unknown voice name: {voice_name}. "
                    f"Available: {list(VOICE_MAP.keys())}",
                    error_code="ELEV_VOICE_ERR",
                )
            self.voice_id = resolved
        elif voice_id:
            self.voice_id = voice_id
        else:
            self.voice_id = VOICE_MAP["Rachel"]

    @property
    def name(self) -> str:
        """Engine identifier."""
        return 'elevenlabs'

    @property
    def sample_rate(self) -> int:
        """Sample rate derived from output format."""
        rates = {
            "pcm_16000": 16000,
            "pcm_22050": 22050,
            "pcm_24000": 24000,
            "pcm_44100": 44100,
            "mp3_44100_128": 44100,
        }
        return rates.get(self.output_format, 24000)

    def load_model(self) -> None:
        """Initialize the ElevenLabs client (lazy)."""
        self._ensure_client()

    def _ensure_client(self) -> None:
        """Initialize the ElevenLabs client (lazy, checks API key)."""
        if self._client is not None:
            return

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise TTSError(
                "ELEVENLABS_API_KEY environment variable not set. "
                "Get a key at https://elevenlabs.io",
                error_code="ELEV_APIKEY_MISSING",
            )

        ElevenLabs = _get_elevenlabs()
        self._client = ElevenLabs(api_key=api_key)
        self.initialized = True
        logger.info("ElevenLabs client initialized (model=%s)", self.model_id)

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_name: Optional[str] = None,
        style: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice_id: Override voice UUID for this call.
            voice_name: Override voice by human name for this call.
            style: Expressiveness level 0.0-1.0 (higher = more dramatic).
            speed: Speech rate 0.5-2.0 (1.0 = normal).

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        if not text or not text.strip():
            raise TTSError("Text cannot be empty", error_code="ELEV_TEXT_ERR")

        self._ensure_client()

        # Resolve per-call voice override
        vid = self._resolve_voice(voice_id, voice_name)

        # Build voice_settings dict
        voice_settings: Dict[str, float] = {'stability': 0.5, 'similarity_boost': 0.75}
        if style is not None:
            voice_settings['style'] = max(0.0, min(1.0, style))
        if speed is not None:
            voice_settings['speed'] = max(0.5, min(2.0, speed))

        try:
            audio_iter = self._client.text_to_speech.convert(
                text=text,
                voice_id=vid,
                model_id=self.model_id,
                output_format=self.output_format,
                voice_settings=voice_settings,
            )
            # convert() returns an iterator of bytes chunks
            audio_bytes = b"".join(audio_iter)
            return self._pcm_to_audio(audio_bytes)

        except TTSError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                raise TTSError(
                    "ElevenLabs rate limit exceeded. "
                    "Upgrade your plan or wait before retrying.",
                    error_code="ELEV_RATELIMIT",
                    original_error=e,
                )
            if "quota_exceeded" in err_str:
                raise TTSError(
                    "ElevenLabs quota exceeded. "
                    "Check your plan's character/credit limits at https://elevenlabs.io",
                    error_code="ELEV_QUOTA",
                    original_error=e,
                )
            if "401" in err_str or "unauthorized" in err_str:
                raise TTSError(
                    "ElevenLabs API key is invalid or expired.",
                    error_code="ELEV_AUTH_ERR",
                    original_error=e,
                )
            raise TTSError(
                f"ElevenLabs synthesis failed: {e}",
                error_code="ELEV_SYNTH_ERR",
                original_error=e,
            )

    def synthesize_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_name: Optional[str] = None,
        chunk_size: int = 4096,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Synthesize with streaming output (yields audio chunks).

        Args:
            text: Text to synthesize.
            voice_id: Override voice UUID.
            voice_name: Override voice by name.
            chunk_size: Bytes per chunk before yielding.

        Yields:
            Tuple of (audio_chunk float32, sample_rate).
        """
        if not text or not text.strip():
            raise TTSError("Text cannot be empty", error_code="ELEV_TEXT_ERR")

        self._ensure_client()
        vid = self._resolve_voice(voice_id, voice_name)

        try:
            audio_iter = self._client.text_to_speech.convert(
                text=text,
                voice_id=vid,
                model_id=self.model_id,
                output_format=self.output_format,
            )

            buffer = b""
            for chunk in audio_iter:
                buffer += chunk
                while len(buffer) >= chunk_size:
                    yield self._pcm_to_audio(buffer[:chunk_size])
                    buffer = buffer[chunk_size:]

            if buffer:
                yield self._pcm_to_audio(buffer)

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"ElevenLabs streaming synthesis failed: {e}",
                error_code="ELEV_STREAM_ERR",
                original_error=e,
            )

    def create_voice_clone(
        self,
        voice_name: str,
        audio_files: List[str],
        description: str = "",
    ) -> str:
        """Clone a voice from audio samples via ElevenLabs Instant Voice Cloning.

        Args:
            voice_name: Name for the cloned voice.
            audio_files: List of paths to audio sample files.
            description: Optional voice description.

        Returns:
            The cloned voice ID.
        """
        self._ensure_client()

        if not audio_files:
            raise TTSError(
                "At least one audio file required for voice cloning.",
                error_code="ELEV_CLONE_ERR",
            )

        handles = []
        try:
            for path in audio_files:
                p = Path(path)
                if not p.exists():
                    raise TTSError(
                        f"Audio file not found: {path}",
                        error_code="ELEV_CLONE_ERR",
                    )
                handles.append(open(p, "rb"))

            voice = self._client.voices.ivc.create(
                name=voice_name,
                files=handles,
                description=description,
            )
            self._cloned_voices[voice_name] = voice.voice_id
            logger.info(
                "Cloned voice '%s' created (id=%s)", voice_name, voice.voice_id,
            )
            return voice.voice_id

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"Voice cloning failed: {e}",
                error_code="ELEV_CLONE_ERR",
                original_error=e,
            )
        finally:
            for h in handles:
                h.close()

    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get all available voices (preset + cloned)."""
        voices: Dict[str, Dict[str, Any]] = {}
        for name, vid in VOICE_MAP.items():
            voices[name] = {"type": "preset", "voice_id": vid}
        for name, vid in self._cloned_voices.items():
            voices[name] = {"type": "cloned", "voice_id": vid}
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
        logger.info("ElevenLabs TTS cleaned up")

    def _resolve_voice(
        self,
        voice_id: Optional[str] = None,
        voice_name: Optional[str] = None,
    ) -> str:
        """Resolve voice ID from optional overrides."""
        if voice_name:
            # Check cloned voices first, then presets
            vid = self._cloned_voices.get(voice_name) or VOICE_MAP.get(voice_name)
            if not vid:
                raise TTSError(
                    f"Unknown voice: {voice_name}. "
                    f"Preset: {list(VOICE_MAP.keys())}",
                    error_code="ELEV_VOICE_ERR",
                )
            return vid
        return voice_id or self.voice_id

    def _pcm_to_audio(self, pcm_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Convert raw PCM bytes (signed 16-bit LE) to float32 numpy array."""
        if not pcm_bytes:
            return np.array([], dtype=np.float32), self.sample_rate
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, self.sample_rate

    def __enter__(self) -> "ElevenLabsTTS":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.cleanup()
        return False
