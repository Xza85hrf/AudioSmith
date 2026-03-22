"""AudioSmith F5-TTS Engine.

Local flow-matching TTS with voice cloning and Polish support.
Uses the f5-tts package for inference with vocos vocoder.

Features:
- Flow matching on mel spectrograms (no codec bottleneck)
- Zero-shot voice cloning via reference audio
- Proven Polish checkpoint (Gregniuki)
- 24kHz output, float32

License note: Gregniuki checkpoint is CC-BY-NC-4.0 (non-commercial).
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np

from audiosmith.exceptions import TTSError

logger = logging.getLogger("audiosmith.f5_tts")


F5_LANGUAGE_MAP: Dict[str, str] = {
    "en": "en",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "pt": "pt",
    "ru": "ru",
    "pl": "pl",
    "ar": "ar",
    "it": "it",
    "nl": "nl",
}

F5_LANGS: Set[str] = set(F5_LANGUAGE_MAP.keys())


def _get_f5_infer():
    """Lazy import f5_tts inference utilities."""
    try:
        from f5_tts.infer.utils_infer import (infer_process, load_model,
                                              load_vocoder,
                                              preprocess_ref_audio_text)
        return load_model, load_vocoder, infer_process, preprocess_ref_audio_text
    except ImportError:
        raise TTSError(
            "f5-tts not installed. Install: pip install f5-tts",
            error_code="F5_IMPORT_ERR",
        )


def _get_dit():
    """Lazy import DiT model class."""
    try:
        from f5_tts.model import DiT
        return DiT
    except ImportError:
        raise TTSError(
            "f5-tts not installed. Install: pip install f5-tts",
            error_code="F5_IMPORT_ERR",
        )


def _get_torch():
    """Lazy import torch."""
    try:
        import torch
        return torch
    except ImportError:
        raise TTSError(
            "PyTorch not installed. Install: pip install torch",
            error_code="F5_IMPORT_ERR",
        )


def _get_soundfile():
    """Lazy import soundfile."""
    try:
        import soundfile as sf
        return sf
    except ImportError:
        raise TTSError(
            "soundfile not installed. Install: pip install soundfile",
            error_code="F5_IMPORT_ERR",
        )


# Model presets — HuggingFace repo + checkpoint paths
F5_MODELS: Dict[str, Dict[str, str]] = {
    "f5-tts": {
        "hf_id": "SWivid/F5-TTS",
        "ckpt": "F5TTS_v1_Base/model_1200000.safetensors",
    },
    "f5-polish": {
        "hf_id": "Gregniuki/F5-tts_English_German_Polish",
        "ckpt": "Polish/model_500000.pt",
        "vocab": "Polish/vocab.txt",
    },
}

F5_SAMPLE_RATE = 24000

F5_MODEL_CONFIG = dict(
    dim=1024, depth=22, heads=16, ff_mult=2,
    text_dim=512, conv_layers=4,
)


class F5TTS:
    """F5-TTS engine with flow-matching synthesis and voice cloning.

    Uses DiT backbone with vocos vocoder. Supports custom checkpoints
    and Polish via the Gregniuki fine-tuned model.
    """

    def __init__(
        self,
        model_name: str = "f5-tts",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._checkpoint_path = checkpoint_path
        self._vocab_path = vocab_path
        self._model: Any = None
        self._vocoder: Any = None
        self._voice_cache: Dict[str, Tuple[Any, str]] = {}
        self._device = device
        self.initialized = False

    @property
    def name(self) -> str:
        """Engine identifier."""
        return 'f5'

    @property
    def sample_rate(self) -> int:
        """F5-TTS outputs at 24kHz."""
        return F5_SAMPLE_RATE

    def load_model(self) -> None:
        """Load the F5-TTS model and vocoder (lazy initialization)."""
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Load F5-TTS model and vocoder (lazy, first call only)."""
        if self._model is not None:
            return

        torch = _get_torch()
        load_model, load_vocoder, _, _ = _get_f5_infer()
        DiT = _get_dit()

        # Resolve device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Resolve checkpoint and vocab paths
        ckpt_path, vocab_file = self._resolve_paths()

        # Build model config (load_model adds text_num_embeds and mel_dim)
        model_cfg = dict(F5_MODEL_CONFIG)

        logger.info(
            "Loading F5-TTS model '%s' on %s (ckpt=%s)",
            self.model_name, self.device, ckpt_path,
        )

        self._model = load_model(
            DiT, model_cfg, str(ckpt_path),
            mel_spec_type="vocos",
            vocab_file=str(vocab_file) if vocab_file else "",
            device=self.device,
        )
        self._vocoder = load_vocoder(
            vocoder_name="vocos", is_local=False, device=self.device,
        )
        self.initialized = True
        logger.info("F5-TTS model loaded successfully")

    def _resolve_paths(self) -> Tuple[Path, Optional[Path]]:
        """Resolve checkpoint and vocab file paths from preset or custom."""
        if self._checkpoint_path:
            ckpt = Path(self._checkpoint_path)
            vocab = Path(self._vocab_path) if self._vocab_path else None
            if not ckpt.exists():
                raise TTSError(
                    f"Checkpoint not found: {ckpt}",
                    error_code="F5_CKPT_ERR",
                )
            return ckpt, vocab

        preset = F5_MODELS.get(self.model_name)
        if not preset:
            available = ", ".join(F5_MODELS.keys())
            raise TTSError(
                f"Unknown model '{self.model_name}'. Available: {available}",
                error_code="F5_MODEL_ERR",
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise TTSError(
                "huggingface_hub not installed. Install: pip install huggingface-hub",
                error_code="F5_IMPORT_ERR",
            )

        hf_id = preset["hf_id"]
        ckpt_file = preset["ckpt"]

        logger.info("Downloading checkpoint from %s/%s", hf_id, ckpt_file)
        ckpt_path = Path(hf_hub_download(repo_id=hf_id, filename=ckpt_file))

        vocab_path = None
        if "vocab" in preset:
            vocab_path = Path(hf_hub_download(repo_id=hf_id, filename=preset["vocab"]))

        return ckpt_path, vocab_path

    def _get_vocab_size(self, vocab_file: Optional[Path]) -> int:
        """Read vocab size from vocab.txt or use default."""
        if vocab_file and vocab_file.exists():
            with open(vocab_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return len(lines)
        # Default F5-TTS base vocab size
        return 2546

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice name (from clone_voice cache) for voice cloning.
            language: Language code (unused, F5-TTS is language-agnostic).
            speed: Speech speed multiplier (default 1.0).

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        if not text or not text.strip():
            raise TTSError("Text cannot be empty", error_code="F5_TEXT_ERR")

        self._ensure_model()
        _, _, infer_process, _ = _get_f5_infer()

        # Resolve voice reference
        ref_audio = None
        ref_text = ""
        if voice and voice in self._voice_cache:
            ref_audio, ref_text = self._voice_cache[voice]
        elif voice:
            logger.warning("Voice '%s' not in cache, synthesizing without reference", voice)

        try:
            audio, sr, _spec = infer_process(
                ref_audio, ref_text, text,
                self._model, self._vocoder,
                speed=speed, device=self.device,
            )
            # Ensure float32 mono
            if isinstance(audio, np.ndarray):
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return audio.astype(np.float32), sr

            # torch tensor → numpy
            torch = _get_torch()
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return audio.astype(np.float32), sr

            return np.array(audio, dtype=np.float32), sr

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"F5-TTS synthesis failed: {e}",
                error_code="F5_SYNTH_ERR",
                original_error=e,
            )

    def clone_voice(
        self,
        name: str,
        audio_path_or_array: Union[str, Tuple[np.ndarray, int]],
        ref_text: Optional[str] = None,
    ) -> str:
        """Register a reference voice for cloning.

        Args:
            name: Name to cache this voice under.
            audio_path_or_array: Path to reference audio file or tuple of (audio_array, sample_rate).
            ref_text: Transcript of reference audio (improves quality).

        Returns:
            The voice name for use in synthesize().
        """
        self._ensure_model()
        _, _, _, preprocess_ref_audio_text = _get_f5_infer()

        # Handle both file path and audio array input
        if isinstance(audio_path_or_array, str):
            path = Path(audio_path_or_array)
            if not path.exists():
                raise TTSError(
                    f"Reference audio not found: {audio_path_or_array}",
                    error_code="F5_CLONE_ERR",
                )
            audio_input = str(path)
        else:
            # Tuple of (audio_array, sample_rate)
            audio_input = audio_path_or_array

        try:
            processed_audio, processed_text = preprocess_ref_audio_text(
                audio_input, ref_text or "",
            )
            self._voice_cache[name] = (processed_audio, processed_text)
            logger.info("Voice '%s' cached", name)
            return name

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"Voice cloning failed: {e}",
                error_code="F5_CLONE_ERR",
                original_error=e,
            )

    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get cached voice references."""
        voices: Dict[str, Dict[str, Any]] = {}
        for name in self._voice_cache:
            voices[name] = {"type": "cloned", "engine": "f5-tts"}
        return voices

    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None,
    ) -> Path:
        """Save audio array to WAV file."""
        sf = _get_soundfile()
        out = Path(output_path)
        if not out.suffix:
            out = out.with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, sample_rate or self.sample_rate)
        return out

    def cleanup(self) -> None:
        """Release model, vocoder, and voice cache."""
        self._model = None
        self._vocoder = None
        self._voice_cache.clear()
        self.initialized = False
        gc.collect()
        try:
            torch = _get_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except TTSError:
            pass
        logger.info("F5-TTS cleaned up")

    def __enter__(self) -> "F5TTS":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.cleanup()
        return False
