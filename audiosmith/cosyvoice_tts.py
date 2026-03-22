"""AudioSmith CosyVoice2 Text-to-Speech Engine.

Local TTS engine with highest reported MOS (5.53) for open-source models.
Supports 9 languages, zero-shot voice cloning, cross-lingual synthesis,
and instruction-based emotion/dialect control.

Features:
- 9 languages (zh, en, ja, ko, de, es, fr, it, ru)
- Zero-shot voice cloning (requires reference audio + transcript)
- Cross-lingual voice transfer
- Instruction-based control (emotion, dialect, speaking style)
- 0.5B parameter model, ~4GB VRAM

Requires: CosyVoice2 installed locally, COSYVOICE_MODEL_DIR env var.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from audiosmith.exceptions import TTSError

logger = logging.getLogger("audiosmith.cosyvoice_tts")


def _get_cosyvoice():
    """Lazy import and return the CosyVoice AutoModel class."""
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError:
        raise TTSError(
            "CosyVoice2 not installed. Clone from: "
            "https://github.com/FunAudioLLM/CosyVoice2",
            error_code="COSYV_IMPORT_ERR",
        )
    return AutoModel


def _get_soundfile():
    """Lazy import soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        raise TTSError(
            "soundfile not installed. Install: pip install soundfile",
            error_code="COSYV_IMPORT_ERR",
        )
    return sf


COSYVOICE_LANGS: Set[str] = {'zh', 'en', 'ja', 'ko', 'de', 'es', 'fr', 'it', 'ru'}


class CosyVoice2TTS:
    """CosyVoice2 TTS with zero-shot cloning and instruction control.

    Requires COSYVOICE_MODEL_DIR environment variable pointing to the
    downloaded model directory, or pass model_dir to __init__.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = 'cuda',
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self._model: Any = None
        self._cloned_voices: Dict[str, Dict[str, Any]] = {}
        self.initialized = False

    @property
    def name(self) -> str:
        """Engine identifier."""
        return 'cosyvoice'

    @property
    def sample_rate(self) -> int:
        """CosyVoice2 outputs at 22050 Hz."""
        return 22050

    def load_model(self) -> None:
        """Load the CosyVoice2 model (lazy)."""
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Load the CosyVoice2 model (lazy)."""
        if self._model is not None:
            return

        model_dir = self.model_dir or os.getenv('COSYVOICE_MODEL_DIR')
        if not model_dir:
            raise TTSError(
                "COSYVOICE_MODEL_DIR not set. Point it to the CosyVoice2 "
                "model directory (e.g. ~/.cache/cosyvoice/CosyVoice2-0.5B).",
                error_code="COSYV_MODEL_ERR",
            )

        model_path = Path(model_dir)
        if not model_path.exists():
            raise TTSError(
                f"CosyVoice2 model directory not found: {model_dir}",
                error_code="COSYV_MODEL_ERR",
            )

        try:
            AutoModel = _get_cosyvoice()
            self._model = AutoModel(model_dir=str(model_path))
            self.initialized = True
            logger.info("CosyVoice2 model loaded from %s", model_path)
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"Failed to load CosyVoice2 model: {e}",
                error_code="COSYV_MODEL_ERR",
                original_error=e,
            )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Cloned voice name or None for cross-lingual mode.
            language: Language code (ISO 639-1).
            instruct: Instruction text for emotion/style control
                (e.g. "Speak happily", "Whisper softly").

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        if not text or not text.strip():
            raise TTSError("Text cannot be empty", error_code="COSYV_SYNTH_ERR")

        if language and language not in COSYVOICE_LANGS:
            raise TTSError(
                f"Language '{language}' not supported by CosyVoice2. "
                f"Supported: {sorted(COSYVOICE_LANGS)}",
                error_code="COSYV_LANG_ERR",
            )

        self._ensure_model()

        voice_info = self._resolve_voice(voice)

        try:
            if instruct and voice_info:
                # Instruction mode with voice reference
                output_iter = self._model.inference_instruct2(
                    text,
                    instruct,
                    voice_info['ref_audio'],
                )
            elif voice_info and voice_info.get('ref_text'):
                # Zero-shot cloning (highest quality, needs transcript)
                output_iter = self._model.inference_zero_shot(
                    text,
                    voice_info['ref_text'],
                    voice_info['ref_audio'],
                )
            elif voice_info:
                # Cross-lingual (no transcript needed)
                output_iter = self._model.inference_cross_lingual(
                    text,
                    voice_info['ref_audio'],
                )
            else:
                raise TTSError(
                    "CosyVoice2 requires a voice reference for synthesis. "
                    "Use create_voice_clone() first or pass --audio-prompt.",
                    error_code="COSYV_VOICE_ERR",
                )

            audio = self._collect_output(output_iter)
            return audio, self.sample_rate

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"CosyVoice2 synthesis failed: {e}",
                error_code="COSYV_SYNTH_ERR",
                original_error=e,
            )

    def create_voice_clone(
        self,
        voice_name: str,
        ref_audio: Union[str, Path],
        ref_text: Optional[str] = None,
    ) -> str:
        """Register a voice clone from reference audio.

        Args:
            voice_name: Name for the cloned voice.
            ref_audio: Path to reference audio file (10-30s recommended).
            ref_text: Transcript of the reference audio. Providing this
                enables zero-shot mode which produces significantly
                higher quality than cross-lingual mode.

        Returns:
            The voice name for use in synthesize().
        """
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise TTSError(
                f"Reference audio not found: {ref_audio}",
                error_code="COSYV_CLONE_ERR",
            )

        # Validate duration
        sf = _get_soundfile()
        try:
            info = sf.info(str(ref_path))
            if info.duration < 3.0:
                raise TTSError(
                    f"Reference audio too short ({info.duration:.1f}s). "
                    "Minimum 3 seconds, 10-30s recommended.",
                    error_code="COSYV_CLONE_ERR",
                )
            if info.duration < 10.0:
                logger.warning(
                    "Reference audio %.1fs — 10-30s recommended for best quality",
                    info.duration,
                )
        except TTSError:
            raise
        except Exception as e:
            logger.warning("Could not validate reference duration: %s", e)

        self._cloned_voices[voice_name] = {
            'ref_audio': str(ref_path),
            'ref_text': ref_text,
        }
        quality = "zero-shot" if ref_text else "cross-lingual"
        logger.info(
            "CosyVoice2 voice '%s' registered (%s mode)", voice_name, quality,
        )
        return voice_name

    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get registered cloned voices."""
        voices: Dict[str, Dict[str, Any]] = {}
        for name, info in self._cloned_voices.items():
            voices[name] = {
                'type': 'cloned',
                'has_transcript': info.get('ref_text') is not None,
                'mode': 'zero-shot' if info.get('ref_text') else 'cross-lingual',
            }
        return voices

    def cleanup(self) -> None:
        """Release model and clear state."""
        if self._model is not None:
            del self._model
            self._model = None
        self._cloned_voices.clear()
        self.initialized = False
        logger.info("CosyVoice2 TTS cleaned up")

    def __enter__(self) -> "CosyVoice2TTS":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.cleanup()
        return False

    def _resolve_voice(self, voice: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve voice name to reference info dict."""
        if not voice:
            return None
        if voice in self._cloned_voices:
            return self._cloned_voices[voice]
        # Try as direct path
        path = Path(voice)
        if path.exists():
            return {'ref_audio': str(path), 'ref_text': None}
        raise TTSError(
            f"Voice '{voice}' not found. Register with create_voice_clone() "
            f"or pass a path to an audio file.",
            error_code="COSYV_VOICE_ERR",
        )

    def _collect_output(self, output_iter: Any) -> np.ndarray:
        """Collect streaming output from CosyVoice2 into a numpy array.

        CosyVoice2 returns an iterator of dicts, each containing
        a 'tts_speech' key with a torch.Tensor.
        """
        chunks: List[np.ndarray] = []
        for chunk in output_iter:
            tensor = chunk['tts_speech']
            # Convert torch Tensor → numpy float32
            audio_np = tensor.cpu().numpy().astype(np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            chunks.append(audio_np)

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)
