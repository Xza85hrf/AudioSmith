"""Tests for audiosmith.multi_voice_tts module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.multi_voice_tts import MultiVoiceTTS


class TestInit:
    def test_defaults(self):
        tts = MultiVoiceTTS()
        assert tts.device == 'cuda'
        assert tts.language == 'pl'
        assert tts.default_exaggeration == 0.5
        assert tts.default_cfg_weight == 0.5
        assert tts._engine is None
        assert tts.voice_count == 0

    def test_custom(self):
        tts = MultiVoiceTTS(device='cpu', language='en', default_exaggeration=0.8)
        assert tts.device == 'cpu'
        assert tts.language == 'en'
        assert tts.default_exaggeration == 0.8


class TestVoiceAssignment:
    def test_assign_voice(self, tmp_path):
        wav = tmp_path / 'speaker1.wav'
        wav.touch()
        tts = MultiVoiceTTS()
        tts.assign_voice('spk1', str(wav))
        assert tts.voice_count == 1
        assert tts._voice_map['spk1'] == str(wav)

    def test_assign_voice_missing_file(self):
        tts = MultiVoiceTTS()
        with pytest.raises(ValueError, match='not found'):
            tts.assign_voice('spk1', '/nonexistent/path.wav')

    def test_set_default_voice(self, tmp_path):
        wav = tmp_path / 'default.wav'
        wav.touch()
        tts = MultiVoiceTTS()
        tts.set_default_voice(str(wav))
        assert tts._default_prompt == str(wav)

    def test_set_default_voice_missing(self):
        tts = MultiVoiceTTS()
        with pytest.raises(ValueError, match='not found'):
            tts.set_default_voice('/nonexistent.wav')

    def test_auto_assign_voices(self, tmp_path):
        (tmp_path / 'SPEAKER_00.wav').touch()
        (tmp_path / 'SPEAKER_01.wav').touch()
        # SPEAKER_02 intentionally missing

        tts = MultiVoiceTTS()
        tts.auto_assign_voices(['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02'], tmp_path)
        assert tts.voice_count == 2


class TestSynthesize:
    @pytest.fixture
    def loaded_tts(self, tmp_path):
        """Return a MultiVoiceTTS with mocked engine."""
        tts = MultiVoiceTTS(device='cpu', language='en')
        mock_engine = MagicMock()
        mock_engine.synthesize.return_value = np.zeros(16000, dtype=np.float32)
        mock_engine.sample_rate = 24000
        tts._engine = mock_engine

        wav = tmp_path / 'spk.wav'
        wav.touch()
        tts.assign_voice('spk', str(wav))

        default_wav = tmp_path / 'default.wav'
        default_wav.touch()
        tts.set_default_voice(str(default_wav))

        return tts, mock_engine

    def test_with_mapped_speaker(self, loaded_tts, tmp_path):
        tts, mock_engine = loaded_tts
        tts.synthesize('Hello', speaker_id='spk')
        call_kw = mock_engine.synthesize.call_args.kwargs
        assert call_kw['audio_prompt_path'] == str(tmp_path / 'spk.wav')

    def test_with_default_prompt(self, loaded_tts, tmp_path):
        tts, mock_engine = loaded_tts
        tts.synthesize('Hello', speaker_id='unknown')
        call_kw = mock_engine.synthesize.call_args.kwargs
        assert call_kw['audio_prompt_path'] == str(tmp_path / 'default.wav')

    def test_no_prompt_passes_none(self, loaded_tts):
        tts, mock_engine = loaded_tts
        tts._default_prompt = None
        tts.synthesize('Hello', speaker_id='unknown')
        call_kw = mock_engine.synthesize.call_args.kwargs
        assert call_kw['audio_prompt_path'] is None

    def test_emotion_params(self, loaded_tts):
        tts, mock_engine = loaded_tts
        tts.synthesize('Hello', speaker_id='spk', emotion_params={
            'exaggeration': 0.9, 'cfg_weight': 0.3,
        })
        call_kw = mock_engine.synthesize.call_args.kwargs
        assert call_kw['exaggeration'] == 0.9
        assert call_kw['cfg_weight'] == 0.3

    def test_default_params_without_emotion(self, loaded_tts):
        tts, mock_engine = loaded_tts
        tts.synthesize('Hello', speaker_id='spk')
        call_kw = mock_engine.synthesize.call_args.kwargs
        assert call_kw['exaggeration'] == 0.5
        assert call_kw['cfg_weight'] == 0.5

    def test_auto_loads_model(self):
        tts = MultiVoiceTTS()
        with patch.object(tts, 'load_model') as mock_load:
            mock_engine = MagicMock()
            mock_engine.synthesize.return_value = np.zeros(100)

            def set_engine():
                tts._engine = mock_engine

            mock_load.side_effect = set_engine
            tts.synthesize('Test')
            mock_load.assert_called_once()


class TestProperties:
    def test_voice_count(self, tmp_path):
        tts = MultiVoiceTTS()
        assert tts.voice_count == 0
        wav = tmp_path / 's.wav'
        wav.touch()
        tts.assign_voice('s', str(wav))
        assert tts.voice_count == 1

    def test_sample_rate_without_engine(self):
        tts = MultiVoiceTTS()
        assert tts.sample_rate == 24000

    def test_sample_rate_with_engine(self):
        tts = MultiVoiceTTS()
        mock_engine = MagicMock()
        mock_engine.sample_rate = 22050
        tts._engine = mock_engine
        assert tts.sample_rate == 22050


class TestUnload:
    def test_unload_clears_all(self, tmp_path):
        tts = MultiVoiceTTS()
        tts._engine = MagicMock()
        wav = tmp_path / 's.wav'
        wav.touch()
        tts.assign_voice('s', str(wav))

        tts.unload()
        assert tts._engine is None
        assert tts.voice_count == 0

    def test_unload_when_not_loaded(self):
        tts = MultiVoiceTTS()
        tts.unload()  # Should not crash
        assert tts._engine is None
