"""Tests for audiosmith.tts module."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from audiosmith.tts import ChatterboxTTS, LANGUAGE_MAP


class TestLanguageMap:
    def test_english(self):
        assert 'en' in LANGUAGE_MAP

    def test_polish(self):
        assert 'pl' in LANGUAGE_MAP

    def test_count(self):
        assert len(LANGUAGE_MAP) == 23


class TestChatterboxTTS:
    def test_init(self):
        tts = ChatterboxTTS()
        assert tts._device == 'cuda'
        assert tts._model is None

    def test_init_custom_device(self):
        tts = ChatterboxTTS(device='cpu')
        assert tts._device == 'cpu'

    def test_synthesize_unsupported_lang(self):
        tts = ChatterboxTTS()
        tts._model = MagicMock()
        with pytest.raises(ValueError, match='Unsupported language'):
            tts.synthesize('test', language='xx')

    def test_cleanup(self):
        tts = ChatterboxTTS()
        tts._model = MagicMock()
        tts.cleanup()
        assert tts._model is None

    def test_sample_rate(self):
        tts = ChatterboxTTS()
        tts._model = MagicMock()
        tts._model.sr = 24000
        assert tts.sample_rate == 24000
