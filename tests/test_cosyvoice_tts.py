"""Tests for audiosmith.cosyvoice_tts module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.cosyvoice_tts import COSYVOICE_LANGS, CosyVoice2TTS
from audiosmith.exceptions import TTSError


class TestConstants:
    def test_language_set_has_9_languages(self):
        assert len(COSYVOICE_LANGS) == 9

    def test_languages_include_key_langs(self):
        for lang in ('en', 'zh', 'ja', 'ko', 'de', 'es', 'fr', 'it', 'ru'):
            assert lang in COSYVOICE_LANGS

    def test_all_iso_codes_two_chars(self):
        for code in COSYVOICE_LANGS:
            assert len(code) == 2


class TestCosyVoice2TTS_Init:
    def test_defaults(self):
        tts = CosyVoice2TTS()
        assert tts.model_dir is None
        assert tts.device == 'cuda'
        assert tts.initialized is False
        assert tts._model is None

    def test_custom_model_dir(self):
        tts = CosyVoice2TTS(model_dir='/some/path')
        assert tts.model_dir == '/some/path'

    def test_custom_device(self):
        tts = CosyVoice2TTS(device='cpu')
        assert tts.device == 'cpu'

    def test_sample_rate_default(self):
        tts = CosyVoice2TTS()
        assert tts.sample_rate == 22050


class TestEnsureModel:
    @patch("audiosmith.cosyvoice_tts._get_cosyvoice")
    def test_load_success(self, mock_get, tmp_path):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        tts = CosyVoice2TTS(model_dir=str(tmp_path))
        tts._ensure_model()

        assert tts._model is mock_model
        assert tts.initialized is True

    @patch("audiosmith.cosyvoice_tts._get_cosyvoice")
    def test_model_dir_not_set(self, mock_get):
        mock_get.return_value = MagicMock()
        tts = CosyVoice2TTS()
        with patch.dict("os.environ", {}, clear=True):
            # Remove COSYVOICE_MODEL_DIR if present
            import os
            os.environ.pop('COSYVOICE_MODEL_DIR', None)
            with pytest.raises(TTSError, match="COSYVOICE_MODEL_DIR"):
                tts._ensure_model()

    @patch("audiosmith.cosyvoice_tts._get_cosyvoice")
    def test_model_dir_missing(self, mock_get):
        mock_get.return_value = MagicMock()
        tts = CosyVoice2TTS(model_dir='/nonexistent/path')
        with pytest.raises(TTSError, match="not found"):
            tts._ensure_model()

    @patch("audiosmith.cosyvoice_tts._get_cosyvoice")
    def test_already_loaded_skips(self, mock_get):
        tts = CosyVoice2TTS()
        tts._model = MagicMock()
        tts._ensure_model()
        mock_get.assert_not_called()


class TestSynthesize:
    def test_empty_text(self):
        tts = CosyVoice2TTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("")

    def test_whitespace_text(self):
        tts = CosyVoice2TTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("   \n  ")

    def test_unsupported_language(self):
        tts = CosyVoice2TTS()
        tts._model = MagicMock()
        with pytest.raises(TTSError, match="not supported"):
            tts.synthesize("Hello", language='pl')

    def test_no_voice_raises(self):
        tts = CosyVoice2TTS()
        tts._model = MagicMock()
        with pytest.raises(TTSError, match="requires a voice reference"):
            tts.synthesize("Hello", voice=None)

    @patch("audiosmith.cosyvoice_tts._get_cosyvoice")
    def test_zero_shot_success(self, mock_get, tmp_path):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        # Mock torch tensor output
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(22050).astype(np.float32)
        mock_model.inference_zero_shot.return_value = [{'tts_speech': mock_tensor}]

        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)

        tts = CosyVoice2TTS(model_dir=str(tmp_path))
        tts.create_voice_clone('clone', ref_audio=ref, ref_text='hello world')
        audio, sr = tts.synthesize("Test text", voice='clone')

        assert isinstance(audio, np.ndarray)
        assert sr == 22050
        mock_model.inference_zero_shot.assert_called_once()

    @patch("audiosmith.cosyvoice_tts._get_cosyvoice")
    def test_cross_lingual_when_no_ref_text(self, mock_get, tmp_path):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.zeros(22050, dtype=np.float32)
        mock_model.inference_cross_lingual.return_value = [{'tts_speech': mock_tensor}]

        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)

        tts = CosyVoice2TTS(model_dir=str(tmp_path))
        tts.create_voice_clone('clone', ref_audio=ref)  # No ref_text
        audio, sr = tts.synthesize("Test", voice='clone')

        mock_model.inference_cross_lingual.assert_called_once()

    @patch("audiosmith.cosyvoice_tts._get_cosyvoice")
    def test_instruct_mode(self, mock_get, tmp_path):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.zeros(22050, dtype=np.float32)
        mock_model.inference_instruct2.return_value = [{'tts_speech': mock_tensor}]

        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)

        tts = CosyVoice2TTS(model_dir=str(tmp_path))
        tts.create_voice_clone('clone', ref_audio=ref, ref_text='hi')
        audio, sr = tts.synthesize("Test", voice='clone', instruct="Speak happily")

        mock_model.inference_instruct2.assert_called_once()

    def test_collect_output_multiple_chunks(self):
        tts = CosyVoice2TTS()
        mock_t1 = MagicMock()
        mock_t1.cpu.return_value.numpy.return_value = np.ones(100, dtype=np.float32)
        mock_t2 = MagicMock()
        mock_t2.cpu.return_value.numpy.return_value = np.ones(200, dtype=np.float32)

        result = tts._collect_output([
            {'tts_speech': mock_t1},
            {'tts_speech': mock_t2},
        ])
        assert len(result) == 300

    def test_collect_output_empty(self):
        tts = CosyVoice2TTS()
        result = tts._collect_output([])
        assert len(result) == 0


class TestVoiceCloning:
    def test_create_clone_success(self, tmp_path):
        import soundfile as sf
        ref = tmp_path / 'ref.wav'
        audio = np.random.randn(22050 * 15).astype(np.float32)
        sf.write(str(ref), audio, 22050)

        tts = CosyVoice2TTS()
        name = tts.create_voice_clone('my-voice', ref, ref_text='test transcript')

        assert name == 'my-voice'
        assert 'my-voice' in tts._cloned_voices
        assert tts._cloned_voices['my-voice']['ref_text'] == 'test transcript'

    def test_create_clone_without_ref_text(self, tmp_path):
        import soundfile as sf
        ref = tmp_path / 'ref.wav'
        audio = np.random.randn(22050 * 15).astype(np.float32)
        sf.write(str(ref), audio, 22050)

        tts = CosyVoice2TTS()
        tts.create_voice_clone('v1', ref)
        assert tts._cloned_voices['v1']['ref_text'] is None

    def test_create_clone_file_not_found(self):
        tts = CosyVoice2TTS()
        with pytest.raises(TTSError, match="not found"):
            tts.create_voice_clone("test", "/nonexistent/audio.wav")

    def test_create_clone_too_short(self, tmp_path):
        import soundfile as sf
        ref = tmp_path / 'short.wav'
        audio = np.zeros(22050 * 2, dtype=np.float32)  # 2s
        sf.write(str(ref), audio, 22050)

        tts = CosyVoice2TTS()
        with pytest.raises(TTSError, match="too short"):
            tts.create_voice_clone("test", ref)

    def test_get_available_voices(self):
        tts = CosyVoice2TTS()
        tts._cloned_voices = {
            'v1': {'ref_audio': '/tmp/v1.wav', 'ref_text': 'hello'},
            'v2': {'ref_audio': '/tmp/v2.wav', 'ref_text': None},
        }
        voices = tts.get_available_voices()
        assert len(voices) == 2
        assert voices['v1']['mode'] == 'zero-shot'
        assert voices['v2']['mode'] == 'cross-lingual'

    def test_resolve_voice_none(self):
        tts = CosyVoice2TTS()
        assert tts._resolve_voice(None) is None

    def test_resolve_voice_cloned(self):
        tts = CosyVoice2TTS()
        tts._cloned_voices = {'my-voice': {'ref_audio': '/tmp/r.wav', 'ref_text': 'hi'}}
        result = tts._resolve_voice('my-voice')
        assert result['ref_audio'] == '/tmp/r.wav'

    def test_resolve_voice_path(self, tmp_path):
        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)
        tts = CosyVoice2TTS()
        result = tts._resolve_voice(str(ref))
        assert result['ref_audio'] == str(ref)

    def test_resolve_voice_not_found(self):
        tts = CosyVoice2TTS()
        with pytest.raises(TTSError, match="not found"):
            tts._resolve_voice("nonexistent-voice")


class TestCleanup:
    def test_cleanup(self):
        tts = CosyVoice2TTS()
        tts._model = MagicMock()
        tts._cloned_voices = {'v': {'ref_audio': '/tmp/v.wav'}}
        tts.initialized = True

        tts.cleanup()

        assert tts._model is None
        assert tts._cloned_voices == {}
        assert tts.initialized is False

    def test_context_manager(self):
        tts = CosyVoice2TTS()
        tts._model = MagicMock()
        tts.initialized = True

        with tts:
            pass

        assert tts._model is None
        assert tts.initialized is False

    def test_context_manager_returns_self(self):
        tts = CosyVoice2TTS()
        with tts as engine:
            assert engine is tts
