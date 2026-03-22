"""AudioSmith IndexTTS-2 Text-to-Speech Engine.

Local TTS with native emotion control via audio prompts and 8D emotion vectors.
Supports independent timbre/emotion control (emotion disentanglement).

Features:
- English + Chinese only (IndexTTS-2 limitation)
- Native emotion control via reference audio or 8D emotion vectors
- Zero-shot voice cloning from reference audio
- Duration control via target_dur parameter
- BigVGAN-v2 vocoder

Requires: indextts library + model checkpoints from HuggingFace.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np

from audiosmith.exceptions import TTSError

logger = logging.getLogger("audiosmith.indextts_tts")

# Supported languages
INDEXTTS_LANGS: Set[str] = {'en', 'zh'}


def _get_indextts():
    """Lazy import IndexTTS inference class."""
    try:
        from indextts.infer import IndexTTS
    except ImportError:
        raise TTSError(
            "indextts not installed. Install: pip install indextts",
            error_code="IDXTTS_IMPORT_ERR",
        )
    return IndexTTS


def _get_soundfile():
    """Lazy import soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        raise TTSError(
            "soundfile not installed. Install: pip install soundfile",
            error_code="IDXTTS_IMPORT_ERR",
        )
    return sf


class IndexTTS2TTS:
    """IndexTTS-2 TTS with emotion disentanglement and voice cloning.

    Requires model checkpoints downloaded from HuggingFace (5.9GB).
    Set INDEXTTS_MODEL_DIR env var or place in ~/.cache/huggingface/indextts/.
    """

    def __init__(
        self,
        model_variant: str = 'base',
        device: str = 'cuda',
        emo_alpha: float = 0.5,
    ) -> None:
        if model_variant not in ('base', 'design'):
            raise TTSError(
                f"Unknown model variant: {model_variant}. Valid: 'base', 'design'",
                error_code="IDXTTS_MODEL_ERR",
            )
        self.model_variant = model_variant
        self.device = device
        self.emo_alpha = emo_alpha
        self._model: Any = None
        self._cloned_voices: Dict[str, Path] = {}
        self._sample_rate: int = 24000  # BigVGAN-v2 default, updated on first synthesis
        self.initialized = False

    @property
    def name(self) -> str:
        """Engine identifier."""
        return 'indextts'

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load_model(self) -> None:
        """Load model from checkpoints directory (lazy)."""
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Load model from checkpoints directory (lazy)."""
        if self._model is not None:
            return

        IndexTTS = _get_indextts()

        model_dir = Path(os.getenv(
            'INDEXTTS_MODEL_DIR',
            str(Path.home() / '.cache' / 'huggingface' / 'indextts'),
        ))

        if not model_dir.exists():
            raise TTSError(
                f"IndexTTS-2 model directory not found: {model_dir}\n"
                "Download: huggingface-cli download IndexTeam/IndexTTS-2 "
                "--local-dir checkpoints\n"
                "Or set INDEXTTS_MODEL_DIR env var.",
                error_code="IDXTTS_MODEL_MISSING",
            )

        cfg_path = model_dir / 'config.yaml'
        if not cfg_path.exists():
            raise TTSError(
                f"IndexTTS-2 config not found: {cfg_path}",
                error_code="IDXTTS_CONFIG_ERR",
            )

        try:
            self._model = IndexTTS(
                model_dir=str(model_dir),
                cfg_path=str(cfg_path),
            )
            self.initialized = True
            logger.info(
                "IndexTTS-2 loaded (variant=%s, device=%s)",
                self.model_variant, self.device,
            )
        except Exception as e:
            raise TTSError(
                f"Failed to load IndexTTS-2: {e}",
                error_code="IDXTTS_LOAD_ERR",
                original_error=e,
            )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = 'en',
        emotion_prompt: Optional[Union[str, Path]] = None,
        target_duration_ms: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio with optional emotion and duration control.

        Args:
            text: Text to synthesize.
            voice: Voice name (from create_voice_clone) or path to ref audio.
            language: 'en' or 'zh'.
            emotion_prompt: Path to emotion reference audio (separate from voice).
            target_duration_ms: Target duration in milliseconds.

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        if not text or not text.strip():
            raise TTSError("Text cannot be empty", error_code="IDXTTS_TEXT_ERR")

        if language not in INDEXTTS_LANGS:
            raise TTSError(
                f"Language '{language}' not supported. "
                f"IndexTTS-2 supports: {sorted(INDEXTTS_LANGS)}",
                error_code="IDXTTS_LANG_ERR",
            )

        self._ensure_model()
        spk_audio = self._resolve_voice(voice)

        if spk_audio is None:
            raise TTSError(
                "IndexTTS-2 requires voice reference audio. "
                "Provide via --audio-prompt or create_voice_clone().",
                error_code="IDXTTS_VOICE_ERR",
            )

        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav')
            os.close(tmp_fd)

            infer_kwargs: Dict[str, Any] = {
                'text': text,
                'spk_audio_prompt': str(spk_audio),
                'output_path': tmp_path,
                'emo_alpha': self.emo_alpha,
                'verbose': False,
            }

            if emotion_prompt:
                emo_path = Path(emotion_prompt)
                if not emo_path.exists():
                    raise TTSError(
                        f"Emotion prompt not found: {emotion_prompt}",
                        error_code="IDXTTS_EMO_ERR",
                    )
                infer_kwargs['emo_audio_prompt'] = str(emo_path)

            if target_duration_ms and target_duration_ms > 0:
                infer_kwargs['target_dur'] = target_duration_ms

            self._model.infer(**infer_kwargs)

            sf = _get_soundfile()
            audio, sr = sf.read(tmp_path, dtype='float32')
            self._sample_rate = sr  # Update from actual output

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            return audio.astype(np.float32), sr

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"IndexTTS-2 synthesis failed: {e}",
                error_code="IDXTTS_SYNTH_ERR",
                original_error=e,
            )
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def create_voice_clone(
        self,
        voice_name: str,
        ref_audio: Union[str, Path],
    ) -> str:
        """Register a voice from reference audio for later synthesis.

        Args:
            voice_name: Name for the cloned voice.
            ref_audio: Path to reference audio file (10-30s recommended).

        Returns:
            Voice name string for use in synthesize().
        """
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise TTSError(
                f"Reference audio not found: {ref_audio}",
                error_code="IDXTTS_CLONE_ERR",
            )

        self._validate_ref_duration(ref_path)
        self._cloned_voices[voice_name] = ref_path
        logger.info("IndexTTS-2 voice '%s' registered (path=%s)", voice_name, ref_path)
        return voice_name

    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get registered cloned voices."""
        return {
            name: {'type': 'cloned', 'ref_audio_path': str(path)}
            for name, path in self._cloned_voices.items()
        }

    def cleanup(self) -> None:
        """Release model and clear caches."""
        self._model = None
        self._cloned_voices.clear()
        self.initialized = False
        logger.info("IndexTTS-2 cleaned up")

    def __enter__(self) -> "IndexTTS2TTS":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def _resolve_voice(self, voice: Optional[str]) -> Optional[Path]:
        """Resolve voice name or path to a Path object."""
        if not voice:
            return None

        if voice in self._cloned_voices:
            return self._cloned_voices[voice]

        path = Path(voice)
        if path.exists():
            return path

        raise TTSError(
            f"Voice not found: {voice}. "
            f"Available: {list(self._cloned_voices.keys())}",
            error_code="IDXTTS_VOICE_ERR",
        )

    def _validate_ref_duration(self, ref_path: Path) -> None:
        """Warn if reference audio is too short or too long."""
        try:
            sf = _get_soundfile()
            info = sf.info(str(ref_path))
            if info.duration < 3.0:
                raise TTSError(
                    f"Reference audio too short ({info.duration:.1f}s). "
                    "Minimum 10s recommended for quality cloning.",
                    error_code="IDXTTS_CLONE_ERR",
                )
            if info.duration < 10.0:
                logger.warning(
                    "Reference audio is %.1fs — 10-30s recommended for best quality",
                    info.duration,
                )
        except TTSError:
            raise
        except Exception as e:
            logger.warning("Could not validate ref duration: %s", e)
