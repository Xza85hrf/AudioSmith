"""Tests for audiosmith.fish_speech_tts module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.exceptions import TTSError
from audiosmith.fish_speech_tts import (
    FISH_LANGUAGE_MAP,
    FISH_LANGS,
    FishSpeechTTS,
)


def _make_wav_bytes(duration: float = 1.0, sr: int = 44100) -> bytes:
    """Create valid WAV bytes for testing."""
    import soundfile as sf

    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestConstants:
    def test_language_map_has_13_languages(self):
        assert len(FISH_LANGUAGE_MAP) == 13

    def test_fish_langs_set_matches_map(self):
        assert FISH_LANGS == set(FISH_LANGUAGE_MAP.keys())

    def test_all_iso_codes_two_chars(self):
        for code in FISH_LANGUAGE_MAP:
            assert len(code) == 2


class TestFishSpeechTTS_Init:
    def test_defaults(self):
        tts = FishSpeechTTS()
        assert tts.model_id == "s1"
        assert tts.sample_rate == 44100
        assert tts.initialized is False
        assert tts._client is None

    def test_custom_model_id(self):
        tts = FishSpeechTTS(model_id="s1-mini")
        assert tts.model_id == "s1-mini"

    def test_ensure_client_no_api_key(self):
        tts = FishSpeechTTS()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(TTSError, match="FISH_API_KEY"):
                tts._ensure_client()

    @patch("audiosmith.fish_speech_tts._get_fishaudio")
    def test_ensure_client_success(self, mock_get):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.FishAudio.return_value = mock_client
        mock_get.return_value = mock_module

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            tts._ensure_client()

        assert tts.initialized is True
        assert tts._client is mock_client


class TestFishSpeechTTS_Synthesize:
    @patch("audiosmith.fish_speech_tts._get_fishaudio")
    def test_synthesize_success(self, mock_get):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.FishAudio.return_value = mock_client
        mock_get.return_value = mock_module

        wav_bytes = _make_wav_bytes(0.5)
        mock_client.tts.convert.return_value = wav_bytes

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

    @patch("audiosmith.fish_speech_tts._get_fishaudio")
    def test_synthesize_with_voice(self, mock_get):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.FishAudio.return_value = mock_client
        mock_get.return_value = mock_module

        wav_bytes = _make_wav_bytes(0.5)
        mock_client.tts.convert.return_value = wav_bytes

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            tts.synthesize("Hello", voice="some-reference-id")

        mock_client.tts.convert.assert_called_once_with(
            text="Hello", reference_id="some-reference-id",
        )

    @patch("audiosmith.fish_speech_tts._get_fishaudio")
    def test_synthesize_api_error(self, mock_get):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.FishAudio.return_value = mock_client
        mock_get.return_value = mock_module
        mock_client.tts.convert.side_effect = RuntimeError("API down")

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            with pytest.raises(TTSError, match="synthesis failed"):
                tts.synthesize("Hello")

    @patch("audiosmith.fish_speech_tts._get_fishaudio")
    def test_synthesize_rate_limit(self, mock_get):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.FishAudio.return_value = mock_client
        mock_get.return_value = mock_module
        mock_client.tts.convert.side_effect = RuntimeError("rate_limit exceeded 429")

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            with pytest.raises(TTSError, match="rate limit"):
                tts.synthesize("Hello")


class TestFishSpeechTTS_VoiceClone:
    @patch("audiosmith.fish_speech_tts._get_fishaudio")
    def test_create_clone_success(self, mock_get, tmp_path):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.FishAudio.return_value = mock_client
        mock_get.return_value = mock_module

        mock_voice = MagicMock()
        mock_voice.id = "cloned-voice-123"
        mock_client.voices.create.return_value = mock_voice

        # Create a fake WAV file with sufficient duration
        import soundfile as sf
        audio_path = tmp_path / "ref.wav"
        audio = np.random.randn(44100 * 15).astype(np.float32) * 0.5  # 15s
        sf.write(str(audio_path), audio, 44100)

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            voice_id = tts.create_voice_clone("test-voice", str(audio_path))

        assert voice_id == "cloned-voice-123"
        assert "test-voice" in tts._cloned_voices
        assert tts._cloned_voices["test-voice"] == "cloned-voice-123"

    def test_create_clone_file_not_found(self):
        tts = FishSpeechTTS()
        tts._client = MagicMock()  # Skip API key check
        tts.initialized = True
        with pytest.raises(TTSError, match="not found"):
            tts.create_voice_clone("test", "/nonexistent/audio.wav")

    @patch("audiosmith.fish_speech_tts._get_fishaudio")
    def test_create_clone_audio_too_short(self, mock_get, tmp_path):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.FishAudio.return_value = mock_client
        mock_get.return_value = mock_module

        # Create 2-second audio (below 3s hard minimum)
        import soundfile as sf
        audio_path = tmp_path / "short.wav"
        audio = np.zeros(44100 * 2, dtype=np.float32)
        sf.write(str(audio_path), audio, 44100)

        tts = FishSpeechTTS()
        with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
            with pytest.raises(TTSError, match="too short"):
                tts.create_voice_clone("test", str(audio_path))

    def test_get_available_voices_includes_cloned(self):
        tts = FishSpeechTTS()
        tts._cloned_voices = {"voice1": "id1", "voice2": "id2"}
        voices = tts.get_available_voices()
        assert len(voices) == 2
        assert voices["voice1"]["type"] == "cloned"
        assert voices["voice1"]["reference_id"] == "id1"


class TestFishSpeechTTS_Lifecycle:
    def test_cleanup(self):
        tts = FishSpeechTTS()
        tts._client = MagicMock()
        tts._cloned_voices = {"v": "id"}
        tts.initialized = True

        tts.cleanup()

        assert tts._client is None
        assert tts._cloned_voices == {}
        assert tts.initialized is False

    def test_context_manager(self):
        tts = FishSpeechTTS()
        tts._client = MagicMock()
        tts.initialized = True

        with tts:
            pass

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
        with patch("audiosmith.fish_speech_tts._get_fishaudio", side_effect=TTSError(
            "fish-audio-sdk not installed", error_code="FISH_IMPORT_ERR",
        )):
            tts = FishSpeechTTS()
            with pytest.raises(TTSError, match="not installed"):
                with patch.dict("os.environ", {"FISH_API_KEY": "test-key"}):
                    tts._ensure_client()

    def test_resolve_voice_none(self):
        tts = FishSpeechTTS()
        assert tts._resolve_voice(None) is None

    def test_resolve_voice_cloned(self):
        tts = FishSpeechTTS()
        tts._cloned_voices = {"my-voice": "ref-123"}
        assert tts._resolve_voice("my-voice") == "ref-123"

    def test_resolve_voice_raw_id(self):
        tts = FishSpeechTTS()
        assert tts._resolve_voice("raw-reference-id") == "raw-reference-id"
