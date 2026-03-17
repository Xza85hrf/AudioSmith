"""Tests for audiosmith.orpheus_tts module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.exceptions import TTSError
from audiosmith.orpheus_tts import (ORPHEUS_EMOTION_TAGS, ORPHEUS_LANGS,
                                    ORPHEUS_VOICES, OrpheusTTS)


class TestConstants:
    def test_language_set_has_13_languages(self):
        assert len(ORPHEUS_LANGS) == 13

    def test_languages_include_key_langs(self):
        for lang in ('en', 'zh', 'es', 'fr', 'de', 'it', 'pt', 'hi', 'ko'):
            assert lang in ORPHEUS_LANGS

    def test_all_iso_codes_two_chars(self):
        for code in ORPHEUS_LANGS:
            assert len(code) == 2

    def test_voices_list_has_8(self):
        assert len(ORPHEUS_VOICES) == 8

    def test_voices_include_tara(self):
        assert 'tara' in ORPHEUS_VOICES

    def test_emotion_tags_mapping(self):
        assert ORPHEUS_EMOTION_TAGS['happy'] == '<laugh>'
        assert ORPHEUS_EMOTION_TAGS['sad'] == '<sigh>'
        assert ORPHEUS_EMOTION_TAGS['fearful'] == '<gasp>'


class TestOrpheusTTS_Init:
    def test_defaults(self):
        tts = OrpheusTTS()
        assert tts.default_voice == 'tara'
        assert tts.temperature == 0.7
        assert tts.initialized is False
        assert tts._model is None

    def test_custom_voice(self):
        tts = OrpheusTTS(voice='leo')
        assert tts.default_voice == 'leo'

    def test_invalid_voice(self):
        with pytest.raises(TTSError, match="Unknown voice"):
            OrpheusTTS(voice='invalid')

    def test_custom_temperature(self):
        tts = OrpheusTTS(temperature=0.9)
        assert tts.temperature == 0.9

    def test_sample_rate_default(self):
        tts = OrpheusTTS()
        assert tts.sample_rate == 24000


class TestEnsureModel:
    @patch("audiosmith.orpheus_tts._get_orpheus")
    def test_load_success(self, mock_get):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        tts = OrpheusTTS()
        tts._ensure_model()

        assert tts._model is mock_model
        assert tts.initialized is True

    @patch("audiosmith.orpheus_tts._get_orpheus")
    def test_already_loaded_skips(self, mock_get):
        tts = OrpheusTTS()
        tts._model = MagicMock()
        tts._ensure_model()
        mock_get.assert_not_called()

    @patch("audiosmith.orpheus_tts._get_orpheus")
    def test_load_failure(self, mock_get):
        mock_cls = MagicMock()
        mock_cls.side_effect = RuntimeError("Out of VRAM")
        mock_get.return_value = mock_cls

        tts = OrpheusTTS()
        with pytest.raises(TTSError, match="Failed to load"):
            tts._ensure_model()


class TestSynthesize:
    @patch("audiosmith.orpheus_tts._get_orpheus")
    def test_success(self, mock_get):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        # Return numpy chunks
        chunk1 = np.random.randn(12000).astype(np.float32)
        chunk2 = np.random.randn(12000).astype(np.float32)
        mock_model.generate_speech.return_value = [chunk1, chunk2]

        tts = OrpheusTTS()
        audio, sr = tts.synthesize("Hello world", voice='tara')

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 24000
        assert len(audio) == 24000

    def test_empty_text(self):
        tts = OrpheusTTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("")

    def test_whitespace_text(self):
        tts = OrpheusTTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("   \n  ")

    def test_unsupported_language(self):
        tts = OrpheusTTS()
        tts._model = MagicMock()
        with pytest.raises(TTSError, match="not supported"):
            tts.synthesize("Hej", language='pl')

    def test_invalid_voice(self):
        tts = OrpheusTTS()
        tts._model = MagicMock()
        with pytest.raises(TTSError, match="not found"):
            tts.synthesize("Hello", voice='nonexistent')

    @patch("audiosmith.orpheus_tts._get_orpheus")
    def test_emotion_tag_injection(self, mock_get):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        mock_model.generate_speech.return_value = [
            np.zeros(24000, dtype=np.float32),
        ]

        tts = OrpheusTTS()
        tts.synthesize("Hello", voice='tara', emotion='happy')

        call_kwargs = mock_model.generate_speech.call_args[1]
        assert '<laugh>' in call_kwargs['prompt']

    @patch("audiosmith.orpheus_tts._get_orpheus")
    def test_unknown_emotion_no_tag(self, mock_get):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        mock_model.generate_speech.return_value = [
            np.zeros(24000, dtype=np.float32),
        ]

        tts = OrpheusTTS()
        tts.synthesize("Hello", voice='tara', emotion='neutral')

        call_kwargs = mock_model.generate_speech.call_args[1]
        # 'neutral' has no tag mapping, so text should be unchanged
        assert call_kwargs['prompt'] == 'Hello'

    def test_collect_chunks_bytes(self):
        tts = OrpheusTTS()
        # Simulate bytes chunks (int16 PCM)
        pcm = np.array([1000, -1000, 500], dtype=np.int16).tobytes()
        result = tts._collect_chunks([pcm])
        assert result.dtype == np.float32
        assert len(result) == 3

    def test_collect_chunks_empty(self):
        tts = OrpheusTTS()
        result = tts._collect_chunks([])
        assert len(result) == 0


class TestVoiceCloning:
    def test_create_clone_success(self, tmp_path):
        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)

        tts = OrpheusTTS()
        name = tts.create_voice_clone('my-voice', ref)

        assert name == 'my-voice'
        assert 'my-voice' in tts._cloned_voices
        assert tts._cloned_voices['my-voice'] == ref

    def test_create_clone_file_not_found(self):
        tts = OrpheusTTS()
        with pytest.raises(TTSError, match="not found"):
            tts.create_voice_clone("test", "/nonexistent/audio.wav")

    def test_get_available_voices(self):
        tts = OrpheusTTS()
        tts._cloned_voices = {'custom': Path('/tmp/c.wav')}
        voices = tts.get_available_voices()
        # 8 presets + 1 cloned
        assert len(voices) == 9
        assert voices['tara']['type'] == 'preset'
        assert voices['custom']['type'] == 'cloned'


class TestCleanup:
    def test_cleanup(self):
        tts = OrpheusTTS()
        tts._model = MagicMock()
        tts._cloned_voices = {'v': Path('/tmp/v.wav')}
        tts.initialized = True

        tts.cleanup()

        assert tts._model is None
        assert tts._cloned_voices == {}
        assert tts.initialized is False

    def test_context_manager(self):
        tts = OrpheusTTS()
        tts._model = MagicMock()
        tts.initialized = True

        with tts:
            pass

        assert tts._model is None
        assert tts.initialized is False

    def test_context_manager_returns_self(self):
        tts = OrpheusTTS()
        with tts as engine:
            assert engine is tts
