"""TTS Model Manager — hot-swap engines with automatic VRAM management.

Keeps at most one GPU-resident TTS engine loaded at a time.
When a different engine is requested, the current one is cleaned up
(freeing VRAM) before the new one loads.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from audiosmith.tts_protocol import TTSEngine, get_engine

logger = logging.getLogger("audiosmith.tts_manager")


class TTSModelManager:
    """Hot-swap TTS engine manager.

    Only one engine is loaded in VRAM at a time. Requesting a different
    engine triggers cleanup of the current one before loading the new one.

    Usage:
        mgr = TTSModelManager()
        audio, sr = mgr.synthesize("fish", "Hello world", language="en")
        audio, sr = mgr.synthesize("qwen3", "Bonjour", language="fr")  # swaps
        mgr.cleanup()  # final cleanup

    Or use as a context manager:
        with TTSModelManager() as mgr:
            audio, sr = mgr.synthesize("fish", "Hello")
            # cleanup() is called automatically on exit
    """

    def __init__(self) -> None:
        self._engines: Dict[str, TTSEngine] = {}  # name -> configured instance
        self._active: Optional[str] = None  # currently loaded engine name

    @property
    def active_engine_name(self) -> Optional[str]:
        """Name of the currently loaded engine, or None."""
        return self._active

    @property
    def active_engine(self) -> Optional[TTSEngine]:
        """The currently loaded engine instance, or None."""
        if self._active:
            return self._engines.get(self._active)
        return None

    def register(self, name: str, engine: TTSEngine) -> None:
        """Register a pre-configured engine instance.

        Use this when the engine needs special setup (voice cloning,
        custom config) that get_engine() alone can't provide.
        The engine is NOT loaded into VRAM yet — that happens on first synthesize.

        If re-registering the currently active engine, it is unloaded first.
        """
        if name in self._engines and name == self._active:
            # Replacing the active engine — unload first
            self._unload(name)
        self._engines[name] = engine
        logger.debug("Registered engine '%s' (%s)", name, type(engine).__name__)

    def get_or_create(self, name: str, **kwargs: Any) -> TTSEngine:
        """Get a registered engine or create one via the factory.

        Does NOT load the model — call ensure_loaded() or synthesize() for that.

        Args:
            name: Engine name.
            **kwargs: Passed to get_engine() factory if creating.

        Returns:
            Engine instance.
        """
        if name not in self._engines:
            engine = get_engine(name, **kwargs)
            self._engines[name] = engine
            logger.info("Created engine '%s' via factory", name)
        return self._engines[name]

    def ensure_loaded(self, name: str, **kwargs: Any) -> TTSEngine:
        """Ensure the named engine is loaded in VRAM, swapping if needed.

        If a different engine is currently active, it is cleaned up first.

        Args:
            name: Engine name.
            **kwargs: Passed to get_or_create() if engine doesn't exist.

        Returns:
            The loaded engine instance.
        """
        engine = self.get_or_create(name, **kwargs)

        if self._active == name:
            return engine

        # Swap: unload current, load new
        if self._active is not None:
            self._unload(self._active)

        self._load(name, engine)
        return engine

    def synthesize(self, name: str, text: str, **kwargs: Any) -> Tuple[np.ndarray, int]:
        """Synthesize text using the named engine, hot-swapping if needed.

        Args:
            name: Engine name (e.g. 'fish', 'qwen3', 'chatterbox').
            text: Text to synthesize.
            **kwargs: Passed to engine.synthesize().

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        engine = self.ensure_loaded(name)
        return engine.synthesize(text, **kwargs)

    def cleanup(self) -> None:
        """Clean up all engines and free VRAM.

        Safe to call multiple times.
        """
        for name in list(self._engines.keys()):
            self._unload(name)
        self._engines.clear()
        self._active = None
        self._gc_collect()
        logger.info("TTSModelManager fully cleaned up")

    def _load(self, name: str, engine: TTSEngine) -> None:
        """Load an engine's model into VRAM.

        Args:
            name: Engine name.
            engine: Engine instance.
        """
        logger.info("Loading TTS engine '%s' into VRAM", name)
        if hasattr(engine, 'load_model'):
            engine.load_model()
        self._active = name

    def _unload(self, name: str) -> None:
        """Unload an engine, freeing its VRAM.

        Args:
            name: Engine name.
        """
        engine = self._engines.get(name)
        if engine is None:
            return
        logger.info("Unloading TTS engine '%s' from VRAM", name)
        if hasattr(engine, 'cleanup'):
            engine.cleanup()
        if hasattr(engine, 'unload'):
            engine.unload()
        if self._active == name:
            self._active = None
        self._gc_collect()

    @staticmethod
    def _gc_collect() -> None:
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self) -> TTSModelManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — cleanup on exit."""
        self.cleanup()

    def __contains__(self, name: str) -> bool:
        """Check if an engine is registered."""
        return name in self._engines

    def __len__(self) -> int:
        """Return the number of registered engines."""
        return len(self._engines)
