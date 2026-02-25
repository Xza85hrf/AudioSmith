"""Tests for audiosmith.piper_tts module."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path
from audiosmith.piper_tts import PiperTTS, POLISH_VOICES, ENGLISH_VOICES
from audiosmith.exceptions import TTSError


class TestPiperTTS:
    def test_init_defaults(self):
        tts = PiperTTS()
        assert tts.voice == "en_US-lessac-medium"
        assert tts.model_path is None
        assert tts._model is None

    def test_sample_rate(self):
        assert PiperTTS().sample_rate == 22050

    def test_synthesize_empty_text(self):
        tts = PiperTTS()
        with pytest.raises(TTSError):
            tts.synthesize("")

    def test_synthesize_whitespace_only(self):
        tts = PiperTTS()
        with pytest.raises(TTSError):
            tts.synthesize("   \n\t  ")

    def test_load_model_no_piper_or_path(self):
        tts = PiperTTS(model_path=None)
        with pytest.raises(TTSError):
            tts._load_model()

    def test_list_voices_default(self):
        voices = PiperTTS().list_voices()
        for v in POLISH_VOICES + ENGLISH_VOICES:
            assert v in voices

    def test_list_voices_with_data_path(self, tmp_path):
        (tmp_path / "voice1.onnx").touch()
        (tmp_path / "voice2.onnx").touch()
        (tmp_path / "readme.txt").touch()
        voices = PiperTTS(data_path=tmp_path).list_voices()
        assert "voice1" in voices
        assert "voice2" in voices
        assert "readme" not in voices

    def test_cleanup_no_model(self):
        PiperTTS().cleanup()  # should not raise

    def test_cleanup_with_model(self):
        tts = PiperTTS()
        tts._model = MagicMock()
        tts.cleanup()
        assert tts._model is None

    def test_voice_constants(self):
        assert len(POLISH_VOICES) == 2
        assert len(ENGLISH_VOICES) == 3
        assert all("pl_PL" in v for v in POLISH_VOICES)
        assert all("en_" in v for v in ENGLISH_VOICES)
