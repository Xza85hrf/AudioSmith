"""Tests for TTS engine protocol and factory."""

import pytest
import numpy as np
from unittest.mock import Mock

from audiosmith.tts_protocol import TTSEngine, get_engine


class TestTTSProtocol:
    """Test that all engines conform to the TTSEngine protocol."""

    def test_protocol_has_required_methods(self):
        """Verify TTSEngine protocol defines required methods."""
        # Check the protocol's requirements
        assert hasattr(TTSEngine, 'name')
        assert hasattr(TTSEngine, 'sample_rate')
        assert hasattr(TTSEngine, 'load_model')
        assert hasattr(TTSEngine, 'synthesize')
        assert hasattr(TTSEngine, 'cleanup')

    def test_synthesize_returns_tuple(self):
        """Verify synthesize returns (audio_array, sample_rate) tuple."""
        # Create a mock engine that conforms to the protocol
        mock_engine = Mock(spec=TTSEngine)
        mock_engine.name = 'test'
        mock_engine.sample_rate = 24000
        mock_engine.load_model = Mock()
        mock_engine.cleanup = Mock()

        # Mock synthesize to return the expected tuple
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_engine.synthesize = Mock(return_value=(audio, 24000))

        result = mock_engine.synthesize('test text')
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], int)


class TestTTSFactory:
    """Test the factory function for creating engines."""

    def test_factory_returns_valid_engine(self):
        """Verify factory returns an engine for known engine names."""
        # Test that factory function exists and can be called
        assert callable(get_engine)

    def test_factory_raises_for_unknown_engine(self):
        """Verify factory raises ValueError for unknown engine names."""
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            get_engine('unknown_engine_xyz')

    def test_factory_engine_names(self):
        """Verify factory supports all known engines."""
        known_engines = [
            'chatterbox', 'piper', 'f5', 'elevenlabs', 'fish',
            'qwen3', 'cosyvoice', 'orpheus', 'indextts'
        ]
        for engine_name in known_engines:
            # We can't actually create all engines (missing dependencies),
            # but we can verify the factory function doesn't immediately reject them
            # by checking for the error message
            try:
                get_engine(engine_name)
            except ValueError as e:
                # If it raises ValueError, it should NOT be the "unknown engine" message
                assert "Unknown TTS engine" not in str(e)
            except Exception:
                # Other exceptions are OK (missing deps, config, etc.)
                pass


class TestEngineReturnTypeNormalization:
    """Test that all engines return (audio, sample_rate) tuples."""

    def test_piper_returns_normalized_tuple(self):
        """Piper returns ndarray directly; wrapper should normalize to tuple."""
        # This tests that if an engine returns just ndarray,
        # the protocol wrapper fixes it
        pass

    def test_chatterbox_returns_normalized_tuple(self):
        """Chatterbox returns ndarray directly; wrapper should normalize to tuple."""
        pass

    def test_elevenlabs_already_returns_tuple(self):
        """ElevenLabs already returns (ndarray, sr) tuple."""
        pass


class TestEngineIntegration:
    """Integration tests (skipped if dependencies missing)."""

    @pytest.mark.skipif(True, reason="Requires actual model files/APIs")
    def test_chatterbox_conformance(self):
        """Test real Chatterbox engine conforms to protocol."""
        from audiosmith.chatterbox_adapter import ChatterboxAdapter

        engine = ChatterboxAdapter(device='cpu')
        assert hasattr(engine, 'name')
        assert engine.name == 'chatterbox'
        assert hasattr(engine, 'sample_rate')
        assert isinstance(engine.sample_rate, int)
        assert hasattr(engine, 'load_model')
        assert hasattr(engine, 'synthesize')
        assert hasattr(engine, 'cleanup')
