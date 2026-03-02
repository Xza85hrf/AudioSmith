"""Tests for audiosmith.indextts_tts module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.exceptions import TTSError
from audiosmith.indextts_tts import (
    INDEXTTS_LANGS,
    IndexTTS2TTS,
)


class TestConstants:
    def test_language_set_has_2_languages(self):
        assert len(INDEXTTS_LANGS) == 2

    def test_languages_are_en_zh(self):
        assert INDEXTTS_LANGS == {'en', 'zh'}

    def test_all_iso_codes_two_chars(self):
        for code in INDEXTTS_LANGS:
            assert len(code) == 2


class TestIndexTTS2TTS_Init:
    def test_defaults(self):
        tts = IndexTTS2TTS()
        assert tts.model_variant == 'base'
        assert tts.device == 'cuda'
        assert tts.emo_alpha == 0.5
        assert tts.initialized is False
        assert tts._model is None

    def test_custom_variant(self):
        tts = IndexTTS2TTS(model_variant='design')
        assert tts.model_variant == 'design'

    def test_invalid_variant(self):
        with pytest.raises(TTSError, match="Unknown model variant"):
            IndexTTS2TTS(model_variant='invalid')

    def test_custom_emo_alpha(self):
        tts = IndexTTS2TTS(emo_alpha=0.8)
        assert tts.emo_alpha == 0.8

    def test_sample_rate_default(self):
        tts = IndexTTS2TTS()
        assert tts.sample_rate == 24000


class TestEnsureModel:
    @patch("audiosmith.indextts_tts._get_indextts")
    def test_load_success(self, mock_get, tmp_path):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get.return_value = mock_cls

        # Create fake config file
        cfg = tmp_path / 'config.yaml'
        cfg.write_text('dummy: true')

        tts = IndexTTS2TTS()
        with patch.dict("os.environ", {"INDEXTTS_MODEL_DIR": str(tmp_path)}):
            tts._ensure_model()

        assert tts._model is mock_model
        assert tts.initialized is True

    @patch("audiosmith.indextts_tts._get_indextts")
    def test_model_dir_missing(self, mock_get):
        mock_get.return_value = MagicMock()
        tts = IndexTTS2TTS()
        with patch.dict("os.environ", {"INDEXTTS_MODEL_DIR": "/nonexistent/path"}):
            with pytest.raises(TTSError, match="model directory not found"):
                tts._ensure_model()

    @patch("audiosmith.indextts_tts._get_indextts")
    def test_config_missing(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock()
        tts = IndexTTS2TTS()
        with patch.dict("os.environ", {"INDEXTTS_MODEL_DIR": str(tmp_path)}):
            with pytest.raises(TTSError, match="config not found"):
                tts._ensure_model()

    @patch("audiosmith.indextts_tts._get_indextts")
    def test_already_loaded_skips(self, mock_get):
        tts = IndexTTS2TTS()
        tts._model = MagicMock()
        tts._ensure_model()
        mock_get.assert_not_called()


class TestSynthesize:
    @patch("audiosmith.indextts_tts._get_soundfile")
    @patch("audiosmith.indextts_tts._get_indextts")
    def test_success(self, mock_get_idx, mock_get_sf, tmp_path):
        # Set up model mock
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get_idx.return_value = mock_cls

        # Set up soundfile mock
        mock_sf = MagicMock()
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_sf.read.return_value = (audio_data, 24000)
        mock_get_sf.return_value = mock_sf

        # Create config and ref audio
        cfg = tmp_path / 'config.yaml'
        cfg.write_text('dummy: true')
        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)

        tts = IndexTTS2TTS()
        with patch.dict("os.environ", {"INDEXTTS_MODEL_DIR": str(tmp_path)}):
            audio, sr = tts.synthesize("Hello world", voice=str(ref))

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 24000
        mock_model.infer.assert_called_once()

    def test_empty_text(self):
        tts = IndexTTS2TTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("")

    def test_whitespace_text(self):
        tts = IndexTTS2TTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("   \n  ")

    def test_unsupported_language(self):
        tts = IndexTTS2TTS()
        tts._model = MagicMock()  # Skip model loading
        with pytest.raises(TTSError, match="not supported"):
            tts.synthesize("Hola", language='es')

    def test_no_voice_raises(self):
        tts = IndexTTS2TTS()
        tts._model = MagicMock()  # Skip model loading
        with pytest.raises(TTSError, match="requires voice reference"):
            tts.synthesize("Hello", voice=None)

    @patch("audiosmith.indextts_tts._get_soundfile")
    @patch("audiosmith.indextts_tts._get_indextts")
    def test_stereo_to_mono(self, mock_get_idx, mock_get_sf, tmp_path):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get_idx.return_value = mock_cls

        # Return stereo audio
        mock_sf = MagicMock()
        stereo = np.random.randn(24000, 2).astype(np.float32)
        mock_sf.read.return_value = (stereo, 24000)
        mock_get_sf.return_value = mock_sf

        cfg = tmp_path / 'config.yaml'
        cfg.write_text('dummy: true')
        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)

        tts = IndexTTS2TTS()
        with patch.dict("os.environ", {"INDEXTTS_MODEL_DIR": str(tmp_path)}):
            audio, sr = tts.synthesize("Hello", voice=str(ref))

        assert audio.ndim == 1

    @patch("audiosmith.indextts_tts._get_soundfile")
    @patch("audiosmith.indextts_tts._get_indextts")
    def test_with_emotion_prompt(self, mock_get_idx, mock_get_sf, tmp_path):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get_idx.return_value = mock_cls

        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros(24000, dtype=np.float32), 24000)
        mock_get_sf.return_value = mock_sf

        cfg = tmp_path / 'config.yaml'
        cfg.write_text('dummy: true')
        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)
        emo = tmp_path / 'emotion.wav'
        emo.write_bytes(b'RIFF' + b'\x00' * 100)

        tts = IndexTTS2TTS()
        with patch.dict("os.environ", {"INDEXTTS_MODEL_DIR": str(tmp_path)}):
            tts.synthesize("Hello", voice=str(ref), emotion_prompt=str(emo))

        call_kwargs = mock_model.infer.call_args[1]
        assert 'emo_audio_prompt' in call_kwargs

    def test_emotion_prompt_not_found(self):
        tts = IndexTTS2TTS()
        tts._model = MagicMock()
        # Need a valid voice to get past voice check
        tts._cloned_voices = {'test': Path('/tmp/fake.wav')}
        with pytest.raises(TTSError, match="Emotion prompt not found"):
            tts.synthesize("Hello", voice='test', emotion_prompt='/nonexistent/emo.wav')

    @patch("audiosmith.indextts_tts._get_soundfile")
    @patch("audiosmith.indextts_tts._get_indextts")
    def test_with_target_duration(self, mock_get_idx, mock_get_sf, tmp_path):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        mock_get_idx.return_value = mock_cls

        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros(24000, dtype=np.float32), 24000)
        mock_get_sf.return_value = mock_sf

        cfg = tmp_path / 'config.yaml'
        cfg.write_text('dummy: true')
        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)

        tts = IndexTTS2TTS()
        with patch.dict("os.environ", {"INDEXTTS_MODEL_DIR": str(tmp_path)}):
            tts.synthesize("Hello", voice=str(ref), target_duration_ms=3000)

        call_kwargs = mock_model.infer.call_args[1]
        assert call_kwargs['target_dur'] == 3000


class TestVoiceCloning:
    def test_create_clone_success(self, tmp_path):
        import soundfile as sf
        ref = tmp_path / 'ref.wav'
        audio = np.random.randn(24000 * 15).astype(np.float32)
        sf.write(str(ref), audio, 24000)

        tts = IndexTTS2TTS()
        name = tts.create_voice_clone('my-voice', ref)

        assert name == 'my-voice'
        assert 'my-voice' in tts._cloned_voices
        assert tts._cloned_voices['my-voice'] == ref

    def test_create_clone_file_not_found(self):
        tts = IndexTTS2TTS()
        with pytest.raises(TTSError, match="not found"):
            tts.create_voice_clone("test", "/nonexistent/audio.wav")

    def test_create_clone_too_short(self, tmp_path):
        import soundfile as sf
        ref = tmp_path / 'short.wav'
        audio = np.zeros(24000 * 2, dtype=np.float32)  # 2s
        sf.write(str(ref), audio, 24000)

        tts = IndexTTS2TTS()
        with pytest.raises(TTSError, match="too short"):
            tts.create_voice_clone("test", ref)

    def test_get_available_voices(self):
        tts = IndexTTS2TTS()
        tts._cloned_voices = {
            'v1': Path('/tmp/v1.wav'),
            'v2': Path('/tmp/v2.wav'),
        }
        voices = tts.get_available_voices()
        assert len(voices) == 2
        assert voices['v1']['type'] == 'cloned'

    def test_resolve_voice_none(self):
        tts = IndexTTS2TTS()
        assert tts._resolve_voice(None) is None

    def test_resolve_voice_cloned(self):
        ref = Path('/tmp/ref.wav')
        tts = IndexTTS2TTS()
        tts._cloned_voices = {'my-voice': ref}
        assert tts._resolve_voice('my-voice') == ref

    def test_resolve_voice_path(self, tmp_path):
        ref = tmp_path / 'ref.wav'
        ref.write_bytes(b'RIFF' + b'\x00' * 100)
        tts = IndexTTS2TTS()
        assert tts._resolve_voice(str(ref)) == ref

    def test_resolve_voice_not_found(self):
        tts = IndexTTS2TTS()
        with pytest.raises(TTSError, match="Voice not found"):
            tts._resolve_voice("nonexistent-voice")


class TestCleanup:
    def test_cleanup(self):
        tts = IndexTTS2TTS()
        tts._model = MagicMock()
        tts._cloned_voices = {'v': Path('/tmp/v.wav')}
        tts.initialized = True

        tts.cleanup()

        assert tts._model is None
        assert tts._cloned_voices == {}
        assert tts.initialized is False

    def test_context_manager(self):
        tts = IndexTTS2TTS()
        tts._model = MagicMock()
        tts.initialized = True

        with tts:
            pass

        assert tts._model is None
        assert tts.initialized is False

    def test_context_manager_returns_self(self):
        tts = IndexTTS2TTS()
        with tts as engine:
            assert engine is tts
