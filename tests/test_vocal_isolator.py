"""Tests for audiosmith.vocal_isolator module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audiosmith.exceptions import VocalIsolationError
from audiosmith.vocal_isolator import VocalIsolator


class TestVocalIsolatorInit:
    def test_defaults(self):
        v = VocalIsolator()
        assert v.model_name == 'htdemucs'
        assert v.device == 'cuda'
        assert v._model is None

    def test_custom(self):
        v = VocalIsolator(model_name='htdemucs_ft', device='cpu')
        assert v.model_name == 'htdemucs_ft'
        assert v.device == 'cpu'


class TestGracefulFallback:
    def test_missing_demucs(self, monkeypatch):
        # Block demucs imports
        monkeypatch.setitem(sys.modules, 'demucs', None)
        monkeypatch.setitem(sys.modules, 'demucs.pretrained', None)
        monkeypatch.setitem(sys.modules, 'demucs.apply', None)

        v = VocalIsolator()
        with pytest.raises(VocalIsolationError, match='audiosmith\\[quality\\]'):
            v.load_model()


class TestIsolate:
    @pytest.fixture
    def mock_torch(self):
        """Set up mocked torch and torchaudio for isolation tests."""
        import torch

        # Mock model
        mock_model = MagicMock()
        mock_model.samplerate = 44100
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_model.to = MagicMock(return_value=mock_model)

        # Sources tensor: [batch=1, n_sources=4, channels=2, time=1000]
        mock_sources = torch.randn(1, 4, 2, 1000)

        return mock_model, mock_sources

    def _run_isolate(self, v, mock_apply, wav, sr, audio_file, output_dir=None):
        """Helper: run isolate with mocked torchaudio."""
        with patch('torchaudio.load', return_value=(wav, sr)), \
             patch('torchaudio.save') as mock_save, \
             patch('torchaudio.transforms.Resample', return_value=lambda x: x):
            result = v.isolate(audio_file, output_dir=output_dir)
        return result, mock_save

    def test_returns_paths(self, tmp_path, mock_torch):
        import torch

        mock_model, mock_sources = mock_torch
        mock_apply = MagicMock(return_value=mock_sources)

        v = VocalIsolator(device='cpu')
        v._model = mock_model
        v._apply_model = mock_apply

        audio_file = tmp_path / 'test.wav'
        audio_file.touch()

        result, mock_save = self._run_isolate(
            v, mock_apply, torch.randn(2, 44100), 44100, audio_file, tmp_path,
        )

        assert result['vocals_path'] == tmp_path / 'test_vocals.wav'
        assert result['background_path'] == tmp_path / 'test_background.wav'
        assert mock_save.call_count == 2

    def test_mono_to_stereo(self, tmp_path, mock_torch):
        import torch

        mock_model, mock_sources = mock_torch
        mock_apply = MagicMock(return_value=mock_sources)

        v = VocalIsolator(device='cpu')
        v._model = mock_model
        v._apply_model = mock_apply

        audio_file = tmp_path / 'mono.wav'
        audio_file.touch()

        self._run_isolate(v, mock_apply, torch.randn(1, 44100), 44100, audio_file, tmp_path)

        # apply_model should receive stereo (2 channels)
        input_tensor = mock_apply.call_args[0][1]
        assert input_tensor.shape[1] == 2

    def test_default_output_dir(self, tmp_path, mock_torch):
        import torch

        mock_model, mock_sources = mock_torch
        mock_apply = MagicMock(return_value=mock_sources)

        v = VocalIsolator(device='cpu')
        v._model = mock_model
        v._apply_model = mock_apply

        audio_file = tmp_path / 'song.wav'
        audio_file.touch()

        result, _ = self._run_isolate(v, mock_apply, torch.randn(2, 44100), 44100, audio_file)

        assert result['vocals_path'].parent == tmp_path

    def test_auto_loads_model(self, tmp_path):
        """isolate() calls load_model() if model not loaded."""
        v = VocalIsolator()
        assert v._model is None

        with patch.object(v, 'load_model') as mock_load:
            mock_load.side_effect = VocalIsolationError("test")
            with pytest.raises(VocalIsolationError):
                v.isolate(tmp_path / 'audio.wav')
            mock_load.assert_called_once()


class TestUnload:
    def test_unload_clears_model(self):
        v = VocalIsolator()
        v._model = MagicMock()
        v._apply_model = MagicMock()

        v.unload()
        assert v._model is None
        assert v._apply_model is None

    def test_unload_when_not_loaded(self):
        v = VocalIsolator()
        v.unload()  # Should not crash
        assert v._model is None
