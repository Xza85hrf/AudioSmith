"""TTS engine protocol and factory for AudioSmith.

All TTS engines must conform to the TTSEngine protocol:
- name: Engine identifier string
- sample_rate: Output sample rate in Hz
- load_model(): Load model into memory
- synthesize(text, **kwargs) -> (audio_array, sample_rate)
- cleanup(): Release resources

The factory function get_engine() creates engines by name with lazy imports.
"""

from __future__ import annotations

from typing import Any, Protocol, Tuple, runtime_checkable

import numpy as np



@runtime_checkable
class TTSEngine(Protocol):
    """Protocol that all TTS engines must satisfy.

    All engines return synthesize results as (audio_array, sample_rate) tuples,
    even if the underlying implementation returns just audio.
    """

    @property
    def name(self) -> str:
        """Engine identifier string (e.g. 'chatterbox', 'piper', 'f5')."""
        ...

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        ...

    def load_model(self) -> None:
        """Load the model into memory (lazy initialization OK)."""
        ...

    def synthesize(self, text: str, **kwargs: Any) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            **kwargs: Engine-specific parameters (voice, language, speed, etc.)

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        ...

    def cleanup(self) -> None:
        """Release model resources (CUDA memory, file handles, etc.)."""
        ...


def get_engine(engine_name: str, **kwargs: Any) -> TTSEngine:
    """Factory to create TTS engine by name with lazy imports.

    Args:
        engine_name: Engine name (chatterbox, piper, f5, elevenlabs, fish,
                     qwen3, cosyvoice, orpheus, indextts).
        **kwargs: Engine-specific initialization parameters.

    Returns:
        Engine instance conforming to TTSEngine protocol.

    Raises:
        ValueError: If engine_name is unknown.
        TTSError: If engine fails to initialize (missing deps, config, etc.)
    """
    engine_loaders = {
        'chatterbox': _load_chatterbox,
        'piper': _load_piper,
        'f5': _load_f5,
        'elevenlabs': _load_elevenlabs,
        'fish': _load_fish,
        'qwen3': _load_qwen3,
        'cosyvoice': _load_cosyvoice,
        'orpheus': _load_orpheus,
        'indextts': _load_indextts,
    }

    loader = engine_loaders.get(engine_name)
    if loader is None:
        available = sorted(engine_loaders.keys())
        raise ValueError(
            f"Unknown TTS engine: {engine_name}. Available: {available}"
        )

    return loader(**kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Engine Loaders (lazy imports)
# ─────────────────────────────────────────────────────────────────────────


def _load_chatterbox(**kwargs: Any) -> TTSEngine:
    """Load Chatterbox TTS engine."""
    from audiosmith.tts import ChatterboxTTS

    return ChatterboxTTS(**kwargs)  # type: ignore[return-value]


def _load_piper(**kwargs: Any) -> TTSEngine:
    """Load Piper TTS engine with return-type normalization."""
    from audiosmith.piper_tts import PiperTTS

    # Piper returns just ndarray, not (ndarray, sr) tuple.
    # Wrap it to conform to protocol.
    return PiperAdapter(PiperTTS(**kwargs))


def _load_f5(**kwargs: Any) -> TTSEngine:
    """Load F5-TTS engine."""
    from audiosmith.f5_tts import F5TTS

    return F5TTS(**kwargs)  # type: ignore[return-value]


def _load_elevenlabs(**kwargs: Any) -> TTSEngine:
    """Load ElevenLabs TTS engine."""
    from audiosmith.elevenlabs_tts import ElevenLabsTTS

    return ElevenLabsTTS(**kwargs)  # type: ignore[return-value]


def _load_fish(**kwargs: Any) -> TTSEngine:
    """Load Fish Speech TTS engine."""
    from audiosmith.fish_speech_tts import FishSpeechTTS

    return FishSpeechTTS(**kwargs)  # type: ignore[return-value]


def _load_qwen3(**kwargs: Any) -> TTSEngine:
    """Load Qwen3 TTS engine."""
    from audiosmith.qwen3_tts import Qwen3TTS

    return Qwen3TTS(**kwargs)  # type: ignore[return-value]


def _load_cosyvoice(**kwargs: Any) -> TTSEngine:
    """Load CosyVoice2 TTS engine."""
    from audiosmith.cosyvoice_tts import CosyVoice2TTS

    return CosyVoice2TTS(**kwargs)  # type: ignore[return-value]


def _load_orpheus(**kwargs: Any) -> TTSEngine:
    """Load Orpheus TTS engine."""
    from audiosmith.orpheus_tts import OrpheusTTS

    return OrpheusTTS(**kwargs)  # type: ignore[return-value]


def _load_indextts(**kwargs: Any) -> TTSEngine:
    """Load IndexTTS-2 engine."""
    from audiosmith.indextts_tts import IndexTTS2TTS

    return IndexTTS2TTS(**kwargs)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────
# Adapters (normalize return types for engines that don't return tuples)
# ─────────────────────────────────────────────────────────────────────────


class PiperAdapter:
    """Adapter to normalize Piper TTS return type to (audio, sr) tuple.

    Piper.synthesize() returns just np.ndarray, but the protocol requires
    (np.ndarray, int) tuples. This adapter wraps Piper to normalize.
    """

    def __init__(self, piper_engine: Any) -> None:
        self._engine = piper_engine

    @property
    def name(self) -> str:
        return 'piper'

    @property
    def sample_rate(self) -> int:
        return self._engine.sample_rate  # type: ignore[no-any-return]

    def load_model(self) -> None:
        """Piper loads model lazily in synthesize(), so this is a no-op."""
        # Piper's _load_model is private and called on first synthesize
        pass  # type: ignore[no-any-return]

    def synthesize(self, text: str, **kwargs: Any) -> Tuple[np.ndarray, int]:
        """Synthesize and normalize return type to (audio, sr)."""
        audio = self._engine.synthesize(text, **kwargs)
        return audio, self.sample_rate

    def cleanup(self) -> None:
        self._engine.cleanup()
