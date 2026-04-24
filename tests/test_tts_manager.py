"""Tests for audiosmith.tts_manager module.

Tests the TTSModelManager for hot-swapping TTS engines with VRAM management.
"""

from __future__ import annotations

import gc
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from audiosmith.tts_manager import TTSModelManager


def _make_engine(name: str = "test") -> MagicMock:
    """Create a mock TTSEngine for testing."""
    engine = MagicMock()
    engine.name = name
    engine.sample_rate = 44100
    engine.synthesize.return_value = (np.zeros(1000, dtype=np.float32), 44100)
    return engine


class TestTTSModelManagerInit:
    """Test initialization."""

    def test_init_empty(self):
        """Manager starts with no engines loaded."""
        mgr = TTSModelManager()
        assert mgr.active_engine_name is None
        assert mgr.active_engine is None
        assert len(mgr) == 0


class TestTTSModelManagerRegister:
    """Test engine registration."""

    def test_register_single_engine(self):
        """Can register a single engine."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)
        assert "fish" in mgr
        assert len(mgr) == 1

    def test_register_multiple_engines(self):
        """Can register multiple engines without loading."""
        mgr = TTSModelManager()
        fish = _make_engine("fish")
        qwen = _make_engine("qwen3")
        mgr.register("fish", fish)
        mgr.register("qwen3", qwen)
        assert len(mgr) == 2
        assert "fish" in mgr
        assert "qwen3" in mgr
        assert mgr.active_engine_name is None

    def test_register_replaces_existing(self):
        """Re-registering an engine replaces the old one."""
        mgr = TTSModelManager()
        old_engine = _make_engine("fish")
        new_engine = _make_engine("fish")
        mgr.register("fish", old_engine)
        mgr.register("fish", new_engine)
        assert mgr._engines["fish"] is new_engine

    def test_register_active_engine_unloads_first(self):
        """Re-registering the active engine unloads it first."""
        mgr = TTSModelManager()
        engine1 = _make_engine("fish")
        engine2 = _make_engine("fish")
        mgr.register("fish", engine1)
        mgr.ensure_loaded("fish")
        assert mgr.active_engine_name == "fish"
        mgr.register("fish", engine2)
        # Old engine should be cleaned up
        engine1.cleanup.assert_called()
        assert mgr.active_engine_name is None


class TestTTSModelManagerGetOrCreate:
    """Test get_or_create factory method."""

    def test_get_or_create_registered_engine(self):
        """Returns registered engine without calling factory."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)

        with patch("audiosmith.tts_manager.get_engine") as mock_factory:
            result = mgr.get_or_create("fish")
            assert result is engine
            mock_factory.assert_not_called()

    def test_get_or_create_unregistered_engine(self):
        """Creates engine via factory when not registered."""
        mgr = TTSModelManager()
        factory_engine = _make_engine("qwen3")

        with patch("audiosmith.tts_manager.get_engine", return_value=factory_engine):
            result = mgr.get_or_create("qwen3")
            assert result is factory_engine
            assert "qwen3" in mgr

    def test_get_or_create_with_kwargs(self):
        """Passes kwargs to factory."""
        mgr = TTSModelManager()
        factory_engine = _make_engine("fish")

        with patch("audiosmith.tts_manager.get_engine", return_value=factory_engine) as mock_factory:
            mgr.get_or_create("fish", temperature=0.7, top_p=0.9)
            mock_factory.assert_called_once_with("fish", temperature=0.7, top_p=0.9)

    def test_get_or_create_does_not_load(self):
        """get_or_create does not call load_model()."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.get_or_create("fish")
        engine.load_model.assert_not_called()


class TestTTSModelManagerEnsureLoaded:
    """Test ensure_loaded method."""

    def test_ensure_loaded_first_time(self):
        """First load_model() is called."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)

        result = mgr.ensure_loaded("fish")
        assert result is engine
        assert mgr.active_engine_name == "fish"
        engine.load_model.assert_called_once()

    def test_ensure_loaded_already_active(self):
        """Does not reload if already active."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)
        mgr.ensure_loaded("fish")
        engine.load_model.reset_mock()

        result = mgr.ensure_loaded("fish")
        assert result is engine
        engine.load_model.assert_not_called()

    def test_ensure_loaded_swaps_engines(self):
        """Swaps to a different engine, cleaning up the old one."""
        mgr = TTSModelManager()
        fish = _make_engine("fish")
        qwen = _make_engine("qwen3")
        mgr.register("fish", fish)
        mgr.register("qwen3", qwen)

        mgr.ensure_loaded("fish")
        assert mgr.active_engine_name == "fish"
        fish.load_model.assert_called_once()

        mgr.ensure_loaded("qwen3")
        assert mgr.active_engine_name == "qwen3"
        fish.cleanup.assert_called_once()
        qwen.load_model.assert_called_once()

    def test_ensure_loaded_creates_via_factory(self):
        """Creates engine via factory if not registered."""
        mgr = TTSModelManager()
        factory_engine = _make_engine("piper")

        with patch("audiosmith.tts_manager.get_engine", return_value=factory_engine):
            result = mgr.ensure_loaded("piper")
            assert result is factory_engine
            assert mgr.active_engine_name == "piper"
            factory_engine.load_model.assert_called_once()


class TestTTSModelManagerSynthesize:
    """Test synthesize method."""

    def test_synthesize_basic(self):
        """Synthesize delegates to engine.synthesize()."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        expected_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        engine.synthesize.return_value = (expected_audio, 44100)
        mgr.register("fish", engine)

        audio, sr = mgr.synthesize("fish", "Hello world")

        np.testing.assert_array_equal(audio, expected_audio)
        assert sr == 44100
        engine.synthesize.assert_called_once_with("Hello world")

    def test_synthesize_with_kwargs(self):
        """Passes kwargs to engine.synthesize()."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)

        mgr.synthesize("fish", "Hello", language="en", voice="clone")

        engine.synthesize.assert_called_once_with(
            "Hello", language="en", voice="clone"
        )

    def test_synthesize_loads_if_needed(self):
        """Calls ensure_loaded before synthesizing."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)

        mgr.synthesize("fish", "Hello")

        assert mgr.active_engine_name == "fish"
        engine.load_model.assert_called_once()

    def test_synthesize_with_hot_swap(self):
        """Swaps engines when synthesizing with a different engine."""
        mgr = TTSModelManager()
        fish = _make_engine("fish")
        qwen = _make_engine("qwen3")
        mgr.register("fish", fish)
        mgr.register("qwen3", qwen)

        mgr.synthesize("fish", "Text A")
        assert mgr.active_engine_name == "fish"

        mgr.synthesize("qwen3", "Text B")
        assert mgr.active_engine_name == "qwen3"
        fish.cleanup.assert_called_once()
        qwen.load_model.assert_called_once()


class TestTTSModelManagerCleanup:
    """Test cleanup method."""

    def test_cleanup_single_engine(self):
        """Cleanup calls cleanup on all engines."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)
        mgr.ensure_loaded("fish")

        mgr.cleanup()

        engine.cleanup.assert_called_once()
        assert mgr.active_engine_name is None
        assert len(mgr) == 0

    def test_cleanup_multiple_engines(self):
        """Cleanup calls cleanup on all registered engines."""
        mgr = TTSModelManager()
        fish = _make_engine("fish")
        qwen = _make_engine("qwen3")
        mgr.register("fish", fish)
        mgr.register("qwen3", qwen)
        mgr.ensure_loaded("fish")

        mgr.cleanup()

        fish.cleanup.assert_called_once()
        qwen.cleanup.assert_called_once()
        assert len(mgr) == 0

    def test_cleanup_when_nothing_loaded(self):
        """Cleanup is safe even when nothing is loaded."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)
        # Don't load it

        mgr.cleanup()

        assert len(mgr) == 0
        assert mgr.active_engine_name is None


class TestTTSModelManagerGarbageCollection:
    """Test garbage collection behavior."""

    def test_unload_calls_gc_collect(self):
        """_unload calls gc.collect()."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)

        with patch("audiosmith.tts_manager.gc.collect") as mock_gc:
            mgr._unload("fish")
            mock_gc.assert_called()

    def test_unload_calls_torch_cuda_empty_cache(self):
        """_unload calls torch.cuda.empty_cache() if torch available."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)

        with patch("audiosmith.tts_manager.gc.collect"):
            with patch("torch.cuda.empty_cache") as mock_cuda:
                mgr._unload("fish")
                mock_cuda.assert_called()

    def test_cleanup_calls_gc_collect(self):
        """cleanup() calls gc.collect() at the end."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)

        with patch("audiosmith.tts_manager.gc.collect") as mock_gc:
            mgr.cleanup()
            assert mock_gc.call_count > 0


class TestTTSModelManagerContextManager:
    """Test context manager support."""

    def test_context_manager_enter(self):
        """__enter__ returns self."""
        mgr = TTSModelManager()
        assert mgr.__enter__() is mgr

    def test_context_manager_exit(self):
        """__exit__ calls cleanup()."""
        engine = _make_engine("fish")
        with TTSModelManager() as mgr:
            mgr.register("fish", engine)
            mgr.ensure_loaded("fish")

        engine.cleanup.assert_called()

    def test_context_manager_cleanup_on_exception(self):
        """__exit__ calls cleanup() even on exception."""
        engine = _make_engine("fish")
        try:
            with TTSModelManager() as mgr:
                mgr.register("fish", engine)
                mgr.ensure_loaded("fish")
                raise ValueError("Test error")
        except ValueError:
            pass

        engine.cleanup.assert_called()


class TestTTSModelManagerContains:
    """Test __contains__ magic method."""

    def test_contains_registered_engine(self):
        """__contains__ returns True for registered engines."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)
        assert "fish" in mgr

    def test_contains_unregistered_engine(self):
        """__contains__ returns False for unregistered engines."""
        mgr = TTSModelManager()
        assert "fish" not in mgr


class TestTTSModelManagerLen:
    """Test __len__ magic method."""

    def test_len_empty(self):
        """len() returns 0 for empty manager."""
        mgr = TTSModelManager()
        assert len(mgr) == 0

    def test_len_after_register(self):
        """len() counts registered engines."""
        mgr = TTSModelManager()
        mgr.register("fish", _make_engine("fish"))
        mgr.register("qwen3", _make_engine("qwen3"))
        assert len(mgr) == 2


class TestTTSModelManagerProperties:
    """Test active_engine_name and active_engine properties."""

    def test_active_engine_name_none_initially(self):
        """active_engine_name is None initially."""
        mgr = TTSModelManager()
        assert mgr.active_engine_name is None

    def test_active_engine_name_after_load(self):
        """active_engine_name reflects loaded engine."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)
        mgr.ensure_loaded("fish")
        assert mgr.active_engine_name == "fish"

    def test_active_engine_none_initially(self):
        """active_engine is None initially."""
        mgr = TTSModelManager()
        assert mgr.active_engine is None

    def test_active_engine_after_load(self):
        """active_engine returns the loaded engine instance."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)
        mgr.ensure_loaded("fish")
        assert mgr.active_engine is engine

    def test_active_engine_after_swap(self):
        """active_engine reflects the current engine after swap."""
        mgr = TTSModelManager()
        fish = _make_engine("fish")
        qwen = _make_engine("qwen3")
        mgr.register("fish", fish)
        mgr.register("qwen3", qwen)
        mgr.ensure_loaded("fish")
        mgr.ensure_loaded("qwen3")
        assert mgr.active_engine is qwen


class TestTTSModelManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_unload_nonexistent_engine(self):
        """_unload is safe when engine not registered."""
        mgr = TTSModelManager()
        # Should not raise
        mgr._unload("nonexistent")

    def test_double_cleanup(self):
        """Multiple cleanup() calls are safe."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)
        mgr.ensure_loaded("fish")

        mgr.cleanup()
        engine.cleanup.reset_mock()
        mgr.cleanup()

        # Should not crash, and not call cleanup again on missing engine
        assert len(mgr) == 0

    def test_engine_without_unload_method(self):
        """Manager handles engines without unload() method."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        # Remove unload method (not all engines have it)
        delattr(engine, "unload")
        mgr.register("fish", engine)

        # Should not raise when trying to call unload
        mgr.ensure_loaded("fish")
        mgr._unload("fish")
        engine.cleanup.assert_called()

    def test_swap_to_same_engine_twice(self):
        """Swapping to the same engine multiple times works."""
        mgr = TTSModelManager()
        engine = _make_engine("fish")
        mgr.register("fish", engine)

        mgr.ensure_loaded("fish")
        engine.load_model.reset_mock()

        mgr.ensure_loaded("fish")
        engine.load_model.assert_not_called()

        mgr.ensure_loaded("fish")
        engine.load_model.assert_not_called()


class TestTTSModelManagerIntegration:
    """Integration-style tests (no GPU, but realistic workflows)."""

    def test_realistic_workflow(self):
        """Realistic workflow: register, load, synthesize, swap, cleanup."""
        mgr = TTSModelManager()

        # Register two engines
        fish = _make_engine("fish")
        qwen = _make_engine("qwen3")
        mgr.register("fish", fish)
        mgr.register("qwen3", qwen)

        # Synthesize with fish
        audio1, sr1 = mgr.synthesize("fish", "Hello", language="en")
        assert mgr.active_engine_name == "fish"
        fish.synthesize.assert_called_once()

        # Swap to qwen and synthesize
        audio2, sr2 = mgr.synthesize("qwen3", "Bonjour", language="fr")
        assert mgr.active_engine_name == "qwen3"
        fish.cleanup.assert_called()
        qwen.synthesize.assert_called_once()

        # Cleanup
        mgr.cleanup()
        qwen.cleanup.assert_called()
        assert len(mgr) == 0
        assert mgr.active_engine_name is None

    def test_realistic_workflow_with_context_manager(self):
        """Realistic workflow using context manager."""
        fish = _make_engine("fish")
        qwen = _make_engine("qwen3")

        with TTSModelManager() as mgr:
            mgr.register("fish", fish)
            mgr.register("qwen3", qwen)

            mgr.synthesize("fish", "Hello")
            assert mgr.active_engine_name == "fish"

            mgr.synthesize("qwen3", "Bonjour")
            assert mgr.active_engine_name == "qwen3"

        # Context manager should clean up
        fish.cleanup.assert_called()
        qwen.cleanup.assert_called()

    def test_multiple_swaps_minimal_vram(self):
        """Multiple engine swaps keep only one loaded at a time."""
        mgr = TTSModelManager()
        engines = {}
        for name in ["fish", "qwen3", "chatterbox", "piper"]:
            engine = _make_engine(name)
            engines[name] = engine
            mgr.register(name, engine)

        # Swap through all engines
        for name in ["fish", "qwen3", "chatterbox", "piper"]:
            mgr.ensure_loaded(name)

        # Each previous engine should have been cleaned up
        for name in ["fish", "qwen3", "chatterbox"]:
            engines[name].cleanup.assert_called()

        # Only piper should be active
        assert mgr.active_engine_name == "piper"
        assert engines["piper"].cleanup.call_count == 0
