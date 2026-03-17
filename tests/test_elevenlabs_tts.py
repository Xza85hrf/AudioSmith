"""Tests for audiosmith.elevenlabs_tts module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.elevenlabs_tts import (ELEVENLABS_LANGS, ELEVENLABS_MODELS,
                                       VOICE_MAP, ElevenLabsTTS)
from audiosmith.exceptions import TTSError


class TestConstants:
    """Test module-level constants."""

    def test_voice_map_has_presets(self):
        assert len(VOICE_MAP) > 10
        assert "Rachel" in VOICE_MAP
        assert "Adam" in VOICE_MAP
        assert "Bella" in VOICE_MAP

    def test_voice_map_values_are_strings(self):
        for name, vid in VOICE_MAP.items():
            assert isinstance(vid, str) and len(vid) > 10, f"{name} has invalid ID"

    def test_models_dict_has_entries(self):
        assert "eleven_v3" in ELEVENLABS_MODELS
        assert "eleven_multilingual_v2" in ELEVENLABS_MODELS
        assert "eleven_flash_v2_5" in ELEVENLABS_MODELS

    def test_langs_set_not_empty(self):
        assert len(ELEVENLABS_LANGS) > 20
        assert "en" in ELEVENLABS_LANGS
        assert "es" in ELEVENLABS_LANGS
        assert "pl" in ELEVENLABS_LANGS


class TestElevenLabsTTSInit:
    """Test initialization."""

    def test_init_defaults(self):
        tts = ElevenLabsTTS()
        assert tts.model_id == "eleven_v3"
        assert tts.voice_id == VOICE_MAP["Rachel"]
        assert tts.sample_rate == 24000
        assert tts.initialized is False

    def test_init_custom_model(self):
        tts = ElevenLabsTTS(model_id="eleven_flash_v2_5")
        assert tts.model_id == "eleven_flash_v2_5"

    def test_init_invalid_model(self):
        with pytest.raises(TTSError, match="Unknown model"):
            ElevenLabsTTS(model_id="nonexistent_model")

    def test_init_voice_by_name(self):
        tts = ElevenLabsTTS(voice_name="Adam")
        assert tts.voice_id == VOICE_MAP["Adam"]

    def test_init_voice_by_id(self):
        tts = ElevenLabsTTS(voice_id="custom-uuid-123")
        assert tts.voice_id == "custom-uuid-123"

    def test_init_invalid_voice_name(self):
        with pytest.raises(TTSError, match="Unknown voice name"):
            ElevenLabsTTS(voice_name="FakeVoice")

    def test_init_voice_name_takes_precedence(self):
        tts = ElevenLabsTTS(voice_id="ignored-id", voice_name="Bella")
        assert tts.voice_id == VOICE_MAP["Bella"]

    def test_sample_rate_from_output_format(self):
        tts = ElevenLabsTTS(output_format="pcm_44100")
        assert tts.sample_rate == 44100


class TestEnsureClient:
    """Test client initialization and API key handling."""

    def test_missing_api_key(self):
        tts = ElevenLabsTTS()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(TTSError, match="ELEVENLABS_API_KEY"):
                tts._ensure_client()

    @patch("audiosmith.elevenlabs_tts._get_elevenlabs")
    def test_client_created_with_key(self, mock_get):
        mock_cls = MagicMock()
        mock_get.return_value = mock_cls
        tts = ElevenLabsTTS()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            tts._ensure_client()
        mock_cls.assert_called_once_with(api_key="test-key")
        assert tts.initialized is True

    @patch("audiosmith.elevenlabs_tts._get_elevenlabs")
    def test_client_cached(self, mock_get):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts._ensure_client()
        mock_get.assert_not_called()


class TestSynthesize:
    """Test text-to-speech synthesis."""

    def test_empty_text_raises(self):
        tts = ElevenLabsTTS()
        with pytest.raises(TTSError, match="Text cannot be empty"):
            tts.synthesize("")

    def test_whitespace_text_raises(self):
        tts = ElevenLabsTTS()
        with pytest.raises(TTSError, match="Text cannot be empty"):
            tts.synthesize("   ")

    def test_synthesize_success(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        # Simulate PCM bytes: 4 samples of signed 16-bit
        pcm_data = np.array([1000, -2000, 3000, -4000], dtype=np.int16).tobytes()
        tts._client.text_to_speech.convert.return_value = [pcm_data]

        audio, sr = tts.synthesize("Hello world")

        assert sr == 24000
        assert audio.dtype == np.float32
        assert len(audio) == 4
        np.testing.assert_allclose(audio[0], 1000.0 / 32768.0, atol=1e-5)

    def test_synthesize_uses_default_voice(self):
        tts = ElevenLabsTTS(voice_name="Adam")
        tts._client = MagicMock()
        tts._client.text_to_speech.convert.return_value = [b"\x00\x00"]

        tts.synthesize("Hi")

        call_kwargs = tts._client.text_to_speech.convert.call_args
        assert call_kwargs.kwargs["voice_id"] == VOICE_MAP["Adam"]

    def test_synthesize_voice_override_by_id(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts._client.text_to_speech.convert.return_value = [b"\x00\x00"]

        tts.synthesize("Hi", voice_id="override-id")

        call_kwargs = tts._client.text_to_speech.convert.call_args
        assert call_kwargs.kwargs["voice_id"] == "override-id"

    def test_synthesize_voice_override_by_name(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts._client.text_to_speech.convert.return_value = [b"\x00\x00"]

        tts.synthesize("Hi", voice_name="Bella")

        call_kwargs = tts._client.text_to_speech.convert.call_args
        assert call_kwargs.kwargs["voice_id"] == VOICE_MAP["Bella"]

    def test_synthesize_rate_limit_error(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts._client.text_to_speech.convert.side_effect = Exception("429 rate_limit")

        with pytest.raises(TTSError, match="rate limit"):
            tts.synthesize("Hello")

    def test_synthesize_auth_error(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts._client.text_to_speech.convert.side_effect = Exception("401 unauthorized")

        with pytest.raises(TTSError, match="invalid or expired"):
            tts.synthesize("Hello")

    def test_synthesize_generic_error(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts._client.text_to_speech.convert.side_effect = Exception("network timeout")

        with pytest.raises(TTSError, match="synthesis failed"):
            tts.synthesize("Hello")

    def test_synthesize_passes_model_and_format(self):
        tts = ElevenLabsTTS(model_id="eleven_flash_v2_5", output_format="pcm_44100")
        tts._client = MagicMock()
        tts._client.text_to_speech.convert.return_value = [b"\x00\x00"]

        tts.synthesize("Hi")

        call_kwargs = tts._client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["model_id"] == "eleven_flash_v2_5"
        assert call_kwargs["output_format"] == "pcm_44100"


class TestSynthesizeStreaming:
    """Test streaming synthesis."""

    def test_streaming_yields_chunks(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        # 8 bytes = 4 PCM samples, chunk_size=4 → 2 yields
        pcm = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
        tts._client.text_to_speech.convert.return_value = [pcm]

        chunks = list(tts.synthesize_streaming("Hello", chunk_size=4))
        assert len(chunks) == 2
        assert all(isinstance(c[0], np.ndarray) for c in chunks)

    def test_streaming_empty_text_raises(self):
        tts = ElevenLabsTTS()
        with pytest.raises(TTSError, match="Text cannot be empty"):
            list(tts.synthesize_streaming(""))


class TestVoiceClone:
    """Test voice cloning."""

    def test_clone_success(self, tmp_path):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        mock_voice = MagicMock()
        mock_voice.voice_id = "cloned-id-abc"
        tts._client.voices.ivc.create.return_value = mock_voice

        sample = tmp_path / "sample.wav"
        sample.write_bytes(b"fake audio data")

        result = tts.create_voice_clone("MyVoice", [str(sample)])

        assert result == "cloned-id-abc"
        assert tts._cloned_voices["MyVoice"] == "cloned-id-abc"

    def test_clone_no_files_raises(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        with pytest.raises(TTSError, match="At least one audio file"):
            tts.create_voice_clone("Test", [])

    def test_clone_missing_file_raises(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        with pytest.raises(TTSError, match="Audio file not found"):
            tts.create_voice_clone("Test", ["/nonexistent/file.wav"])

    def test_clone_api_error(self, tmp_path):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts._client.voices.ivc.create.side_effect = Exception("API error")

        sample = tmp_path / "sample.wav"
        sample.write_bytes(b"fake audio data")

        with pytest.raises(TTSError, match="Voice cloning failed"):
            tts.create_voice_clone("Test", [str(sample)])


class TestGetAvailableVoices:
    """Test voice listing."""

    def test_preset_voices_included(self):
        tts = ElevenLabsTTS()
        voices = tts.get_available_voices()
        assert "Rachel" in voices
        assert voices["Rachel"]["type"] == "preset"

    def test_cloned_voices_included(self):
        tts = ElevenLabsTTS()
        tts._cloned_voices["CustomVoice"] = "custom-id"
        voices = tts.get_available_voices()
        assert "CustomVoice" in voices
        assert voices["CustomVoice"]["type"] == "cloned"
        assert voices["CustomVoice"]["voice_id"] == "custom-id"


class TestResolveVoice:
    """Test voice resolution logic."""

    def test_resolve_default(self):
        tts = ElevenLabsTTS(voice_name="Adam")
        assert tts._resolve_voice() == VOICE_MAP["Adam"]

    def test_resolve_override_id(self):
        tts = ElevenLabsTTS()
        assert tts._resolve_voice(voice_id="override") == "override"

    def test_resolve_override_name(self):
        tts = ElevenLabsTTS()
        assert tts._resolve_voice(voice_name="Bella") == VOICE_MAP["Bella"]

    def test_resolve_cloned_voice_name(self):
        tts = ElevenLabsTTS()
        tts._cloned_voices["MyClone"] = "clone-id"
        assert tts._resolve_voice(voice_name="MyClone") == "clone-id"

    def test_resolve_unknown_name_raises(self):
        tts = ElevenLabsTTS()
        with pytest.raises(TTSError, match="Unknown voice"):
            tts._resolve_voice(voice_name="NonexistentVoice")


class TestPcmToAudio:
    """Test PCM bytes → numpy conversion."""

    def test_basic_conversion(self):
        tts = ElevenLabsTTS()
        pcm = np.array([16384, -16384], dtype=np.int16).tobytes()
        audio, sr = tts._pcm_to_audio(pcm)
        assert sr == 24000
        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio[0], 0.5, atol=1e-4)
        np.testing.assert_allclose(audio[1], -0.5, atol=1e-4)

    def test_empty_bytes(self):
        tts = ElevenLabsTTS()
        audio, sr = tts._pcm_to_audio(b"")
        assert len(audio) == 0
        assert sr == 24000


class TestCleanup:
    """Test cleanup and context manager."""

    def test_cleanup(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts._cloned_voices["test"] = "id"
        tts.initialized = True

        tts.cleanup()

        assert tts._client is None
        assert len(tts._cloned_voices) == 0
        assert tts.initialized is False

    def test_context_manager(self):
        tts = ElevenLabsTTS()
        tts._client = MagicMock()
        tts.initialized = True

        with tts as t:
            assert isinstance(t, ElevenLabsTTS)

        assert t._client is None
        assert t.initialized is False
