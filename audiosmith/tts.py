"""Chatterbox multilingual TTS engine wrapper."""

import gc
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    'en': 'en', 'pl': 'pl', 'de': 'de', 'fr': 'fr', 'es': 'es',
    'it': 'it', 'pt': 'pt', 'ru': 'ru', 'ja': 'ja', 'ko': 'ko',
    'zh': 'zh', 'ar': 'ar', 'hi': 'hi', 'nl': 'nl', 'sv': 'sv',
    'da': 'da', 'fi': 'fi', 'el': 'el', 'he': 'he', 'ms': 'ms',
    'nb': 'nb', 'sw': 'sw', 'tr': 'tr',
}


class ChatterboxTTS:
    """Chatterbox multilingual TTS engine with zero-shot voice cloning."""

    def __init__(self, device: str = 'cuda'):
        self._device = device
        self._model = None

    def load_model(self) -> None:
        """Load the Chatterbox multilingual model."""
        import perth
        if perth.PerthImplicitWatermarker is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
        # Fix transformers 5.x: sdpa blocks output_attentions setter
        import transformers.configuration_utils as _cfg
        _cfg.PretrainedConfig.output_attentions = property(
            _cfg.PretrainedConfig.output_attentions.fget,
            lambda self, v: object.__setattr__(self, '_output_attentions', v),
        )
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=self._device)
        # Switch LLM layers from sdpa to eager so attention tensors are returned
        tfmr = self._model.t3.tfmr
        tfmr.config._attn_implementation = 'eager'
        for layer in tfmr.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.config._attn_implementation = 'eager'
        logger.info("Chatterbox multilingual model loaded on %s", self._device)

    def synthesize(
        self,
        text: str,
        language: str = 'en',
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> np.ndarray:
        """Synthesize speech from text. Returns audio waveform as numpy array."""
        if language not in LANGUAGE_MAP:
            raise ValueError(
                f"Unsupported language '{language}'. Supported: {sorted(LANGUAGE_MAP.keys())}"
            )
        wav = self._model.generate(
            text,
            language_id=LANGUAGE_MAP[language],
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        return wav.squeeze().cpu().numpy()

    @property
    def sample_rate(self) -> int:
        """Return the model's native sample rate."""
        return self._model.sr

    def cleanup(self) -> None:
        """Release model and GPU memory."""
        if self._model is not None:
            del self._model
        self._model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Chatterbox model cleaned up")
