"""Tests for audiosmith.fish_speech_tts module."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.exceptions import TTSError
from audiosmith.fish_speech_tts import FISH_LANGS, FishSpeechTTS


def _make_wav_bytes(duration: float = 1.0, sr: int = 44100) -> bytes:
    """Create valid WAV bytes for testing."""
    import soundfile as sf

    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestConstants:
    def test_fish_langs_has_13_languages(self):
        assert len(FISH_LANGS) == 13

    def test_expected_language_codes(self):
        expected = {'en', 'zh', 'ja', 'ko', 'de', 'fr', 'es', 'pt', 'ru', 'nl', 'it', 'pl', 'ar'}
        assert FISH_LANGS == expected

    def test_all_iso_codes_two_chars(self):
        for code in FISH_LANGS:
            assert len(code) == 2


class TestFishSpeechTTS_Init:
    def test_defaults(self):
        tts = FishSpeechTTS()
        assert tts.backend == "speech-1.6"
        assert tts.temperature == 0.7
        assert tts.top_p == 0.7
        assert tts.default_reference_id is None
        assert tts.base_url is None
        assert tts.sample_rate == 44100
        assert tts.initialized is False
        assert tts._client is None

    def test_custom_params(self):
        tts = FishSpeechTTS(
            backend="custom-backend",
            temperature=0.5,
            top_p=0.8,
            reference_id="ref-123",
        )
        assert tts.backend == "custom-backend"
        assert tts.temperature == 0.5
        assert tts.top_p == 0.8
        assert tts.default_reference_id == "ref-123"

    def test_with_local_base_url(self):
        """Test initialization with local server URL."""
        tts = FishSpeechTTS(base_url="http://localhost:8080")
        assert tts.base_url == "http://localhost:8080"
        assert tts.backend == "speech-1.6"
        assert tts.initialized is False

    def test_ensure_client_no_api_key(self):
        tts = FishSpeechTTS()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(TTSError, match="FISH_API_KEY"):
                tts._ensure_client()

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_ensure_client_success(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_get.return_value = mock_sdk

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            tts._ensure_client()

        assert tts.initialized is True
        assert tts._client is mock_client
        mock_sdk.Session.assert_called_once_with(apikey="test-key")

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_ensure_client_local_mode(self, mock_get):
        """Test local server mode initialization (no API key required)."""
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_get.return_value = mock_sdk

        tts = FishSpeechTTS(base_url="http://localhost:8080")
        # No FISH_API_KEY needed for local mode
        with patch.dict("os.environ", {}, clear=True):
            tts._ensure_client()

        assert tts.initialized is True
        assert tts._client is mock_client
        # Should call Session with apikey="local" and base_url
        mock_sdk.Session.assert_called_once_with(apikey="local", base_url="http://localhost:8080")

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_ensure_client_local_mode_custom_url(self, mock_get):
        """Test local server mode with custom URL."""
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_get.return_value = mock_sdk

        custom_url = "http://192.168.1.100:9090"
        tts = FishSpeechTTS(base_url=custom_url)
        with patch.dict("os.environ", {}, clear=True):
            tts._ensure_client()

        assert tts.initialized is True
        mock_sdk.Session.assert_called_once_with(apikey="local", base_url=custom_url)

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_ensure_client_idempotent(self, mock_get):
        """Test that _ensure_client only initializes once."""
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_get.return_value = mock_sdk

        tts = FishSpeechTTS(base_url="http://localhost:8080")
        with patch.dict("os.environ", {}, clear=True):
            tts._ensure_client()
            tts._ensure_client()  # Call again

        # Session should only be called once
        assert mock_sdk.Session.call_count == 1


class TestFishSpeechTTS_Synthesize:
    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_synthesize_success(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_get.return_value = mock_sdk

        wav_bytes = _make_wav_bytes(0.5)
        # Simulate streaming response (generator of byte chunks)
        mock_client.tts.return_value = iter([wav_bytes])

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            audio, sr = tts.synthesize("Hello world")

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 44100
        assert len(audio) > 0

    def test_synthesize_empty_text(self):
        tts = FishSpeechTTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("")

    def test_synthesize_whitespace_text(self):
        tts = FishSpeechTTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("   \n  ")

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_synthesize_with_emotion(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.TTSRequest = MagicMock()
        mock_get.return_value = mock_sdk

        wav_bytes = _make_wav_bytes(0.5)
        mock_client.tts.return_value = iter([wav_bytes])

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            audio, sr = tts.synthesize("Hello", emotion="happy")

        # Verify the emotion tag was prepended to text
        call_args = mock_sdk.TTSRequest.call_args
        assert call_args is not None
        text_arg = call_args[1]['text']
        assert text_arg.startswith("[happy]")

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_synthesize_with_voice(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.TTSRequest = MagicMock()
        mock_get.return_value = mock_sdk

        wav_bytes = _make_wav_bytes(0.5)
        mock_client.tts.return_value = iter([wav_bytes])

        tts = FishSpeechTTS()
        tts._cloned_voices = {"my-voice": "ref-123"}

        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            audio, sr = tts.synthesize("Hello", voice="my-voice")

        # Verify reference_id was resolved from cloned voice
        call_args = mock_sdk.TTSRequest.call_args
        assert call_args is not None
        assert call_args[1]['reference_id'] == "ref-123"

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_synthesize_with_inline_reference(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_ref_audio = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.TTSRequest = MagicMock()
        mock_sdk.ReferenceAudio = MagicMock(return_value=mock_ref_audio)
        mock_get.return_value = mock_sdk

        wav_bytes = _make_wav_bytes(0.5)
        mock_client.tts.return_value = iter([wav_bytes])

        tts = FishSpeechTTS()
        tts._inline_references = {"inline-voice": mock_ref_audio}

        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            audio, sr = tts.synthesize("Hello", voice="inline-voice")

        # Verify inline reference was passed (reference_id should be None, references list populated)
        call_args = mock_sdk.TTSRequest.call_args
        assert call_args is not None
        assert call_args[1]['reference_id'] is None
        assert call_args[1]['references'] == [mock_ref_audio]

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_synthesize_api_error(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.TTSRequest = MagicMock()
        mock_get.return_value = mock_sdk
        mock_client.tts.side_effect = RuntimeError("API down")

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            with pytest.raises(TTSError, match="synthesis failed"):
                tts.synthesize("Hello")

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_synthesize_rate_limit(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.TTSRequest = MagicMock()
        mock_get.return_value = mock_sdk
        mock_client.tts.side_effect = RuntimeError("rate_limit exceeded 429")

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            with pytest.raises(TTSError, match="rate limit"):
                tts.synthesize("Hello")

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_synthesize_auth_error(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.TTSRequest = MagicMock()
        mock_get.return_value = mock_sdk
        mock_client.tts.side_effect = RuntimeError("401 unauthorized")

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            with pytest.raises(TTSError, match="invalid or expired"):
                tts.synthesize("Hello")


class TestFishSpeechTTS_VoiceClone:
    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_create_clone_success(self, mock_get, tmp_path):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_ref_audio = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.ReferenceAudio = MagicMock(return_value=mock_ref_audio)
        mock_get.return_value = mock_sdk

        # Mock the persistent model creation
        mock_model = MagicMock()
        mock_model.id = "model-123"
        mock_client.create_model.return_value = mock_model

        # Create a fake WAV file with sufficient duration
        import soundfile as sf
        audio_path = tmp_path / "ref.wav"
        audio = np.random.randn(44100 * 15).astype(np.float32) * 0.5  # 15s
        sf.write(str(audio_path), audio, 44100)

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            voice_id = tts.create_voice_clone("test-voice", str(audio_path))

        # Should return the model ID
        assert voice_id == "model-123"
        # Both inline and persistent should be stored
        assert "test-voice" in tts._cloned_voices
        assert "test-voice" in tts._inline_references
        assert tts._cloned_voices["test-voice"] == "model-123"

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_create_clone_model_creation_fails_fallback(self, mock_get, tmp_path):
        """When persistent model creation fails, inline reference should still work."""
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_ref_audio = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.ReferenceAudio = MagicMock(return_value=mock_ref_audio)
        mock_get.return_value = mock_sdk

        # Simulate model creation failure
        mock_client.create_model.side_effect = RuntimeError("Backend unavailable")

        import soundfile as sf
        audio_path = tmp_path / "ref.wav"
        audio = np.random.randn(44100 * 15).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio, 44100)

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            voice_id = tts.create_voice_clone("test-voice", str(audio_path))

        # Should return voice_name when model creation fails
        assert voice_id == "test-voice"
        # Inline reference should still be stored
        assert "test-voice" in tts._inline_references

    def test_create_clone_file_not_found(self):
        tts = FishSpeechTTS()
        tts._client = MagicMock()  # Skip API key check
        tts.initialized = True
        with pytest.raises(TTSError, match="not found"):
            tts.create_voice_clone("test", "/nonexistent/audio.wav")

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_create_clone_audio_too_short(self, mock_get, tmp_path):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_get.return_value = mock_sdk

        # Create 2-second audio (below 3s hard minimum)
        import soundfile as sf
        audio_path = tmp_path / "short.wav"
        audio = np.zeros(44100 * 2, dtype=np.float32)
        sf.write(str(audio_path), audio, 44100)

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            with pytest.raises(TTSError, match="too short"):
                tts.create_voice_clone("test", str(audio_path))

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_create_clone_from_numpy_array(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_ref_audio = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.ReferenceAudio = MagicMock(return_value=mock_ref_audio)
        mock_get.return_value = mock_sdk

        mock_model = MagicMock()
        mock_model.id = "model-456"
        mock_client.create_model.return_value = mock_model

        # Create numpy array audio (15 seconds at 44.1kHz)
        audio = np.random.randn(44100 * 15).astype(np.float32) * 0.5
        sr = 44100

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            voice_id = tts.create_voice_clone("np-voice", (audio, sr))

        assert voice_id == "model-456"
        assert "np-voice" in tts._cloned_voices
        assert "np-voice" in tts._inline_references

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_create_clone_accepts_path_object(self, mock_get, tmp_path):
        """Test that create_voice_clone accepts Path objects."""
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_ref_audio = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_sdk.ReferenceAudio = MagicMock(return_value=mock_ref_audio)
        mock_get.return_value = mock_sdk

        mock_model = MagicMock()
        mock_model.id = "model-789"
        mock_client.create_model.return_value = mock_model

        import soundfile as sf
        audio_path = tmp_path / "ref.wav"
        audio = np.random.randn(44100 * 15).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio, 44100)

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            # Pass as Path object, not string
            voice_id = tts.create_voice_clone("path-voice", audio_path)

        assert voice_id == "model-789"
        assert "path-voice" in tts._cloned_voices

    def test_get_available_voices_includes_cloned(self):
        tts = FishSpeechTTS()
        tts._cloned_voices = {"voice1": "id1", "voice2": "id2"}
        voices = tts.get_available_voices()
        assert len(voices) == 2
        assert voices["voice1"]["type"] == "cloned"
        assert voices["voice1"]["reference_id"] == "id1"


class TestFishSpeechTTS_VoiceResolution:
    def test_resolve_reference_id_none(self):
        tts = FishSpeechTTS()
        assert tts._resolve_reference_id(None) is None

    def test_resolve_reference_id_default(self):
        tts = FishSpeechTTS(reference_id="default-ref")
        assert tts._resolve_reference_id(None) == "default-ref"

    def test_resolve_reference_id_cloned_voice(self):
        tts = FishSpeechTTS()
        tts._cloned_voices = {"my-voice": "ref-123"}
        assert tts._resolve_reference_id("my-voice") == "ref-123"

    def test_resolve_reference_id_raw_id(self):
        tts = FishSpeechTTS()
        assert tts._resolve_reference_id("raw-reference-id") == "raw-reference-id"

    def test_resolve_reference_id_prefers_inline(self):
        """Inline references should take precedence over persistent models."""
        tts = FishSpeechTTS()
        tts._inline_references = {"voice": MagicMock()}
        tts._cloned_voices = {"voice": "persistent-id"}
        # Should return None to trigger inline path
        assert tts._resolve_reference_id("voice") is None

    def test_resolve_inline_references_none(self):
        tts = FishSpeechTTS()
        refs = tts._resolve_inline_references(None)
        assert refs == []

    def test_resolve_inline_references_found(self):
        tts = FishSpeechTTS()
        mock_ref = MagicMock()
        tts._inline_references = {"voice": mock_ref}
        refs = tts._resolve_inline_references("voice")
        assert refs == [mock_ref]

    def test_resolve_inline_references_not_found(self):
        tts = FishSpeechTTS()
        refs = tts._resolve_inline_references("nonexistent")
        assert refs == []


class TestFishSpeechTTS_Lifecycle:
    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_cleanup(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_get.return_value = mock_sdk

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            tts._ensure_client()

        tts._cloned_voices = {"v": "id"}
        tts._inline_references = {"v": MagicMock()}

        tts.cleanup()

        assert tts._client is None
        assert tts._cloned_voices == {}
        assert tts._inline_references == {}
        assert tts.initialized is False

    @patch("audiosmith.fish_speech_tts._get_sdk")
    def test_context_manager(self, mock_get):
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.Session.return_value = mock_client
        mock_get.return_value = mock_sdk

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            with tts:
                tts._ensure_client()
                assert tts.initialized is True

        assert tts._client is None
        assert tts.initialized is False

    def test_save_audio(self, tmp_path):
        tts = FishSpeechTTS()
        audio = np.zeros(4410, dtype=np.float32)
        out = tts.save_audio(audio, tmp_path / "test.wav", 44100)

        assert out.exists()
        assert out.suffix == ".wav"

    def test_save_audio_adds_extension(self, tmp_path):
        tts = FishSpeechTTS()
        audio = np.zeros(4410, dtype=np.float32)
        out = tts.save_audio(audio, tmp_path / "test", 44100)

        assert out.suffix == ".wav"

    def test_import_error(self):
        with patch("audiosmith.fish_speech_tts._get_sdk", side_effect=TTSError(
            "fish-audio-sdk not installed", error_code="FISH_IMPORT_ERR",
        )):
            tts = FishSpeechTTS()
            with pytest.raises(TTSError, match="not installed"):
                with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
                    tts._ensure_client()

    def test_name_property(self):
        tts = FishSpeechTTS()
        assert tts.name == 'fish'

    def test_sample_rate_property(self):
        tts = FishSpeechTTS()
        assert tts.sample_rate == 44100

    def test_load_model(self):
        tts = FishSpeechTTS()
        with patch.object(tts, '_ensure_client'):
            tts.load_model()
            tts._ensure_client.assert_called_once()
