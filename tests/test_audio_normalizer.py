"""Tests for audiosmith.audio_normalizer module."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from audiosmith.audio_normalizer import AudioNormalizer


class TestAudioNormalizer:
    @pytest.fixture
    def normalizer(self):
        return AudioNormalizer()

    @patch('audiosmith.audio_normalizer.subprocess.run')
    def test_analyze_parses_lufs(self, mock_run, normalizer):
        mock_run.return_value = MagicMock(
            stderr="I: -20.5 LUFS\nPeak: -3.2 dBFS"
        )
        result = normalizer.analyze(Path("test.wav"))
        assert result["lufs"] == -20.5
        assert result["peak_db"] == -3.2

    @patch('audiosmith.audio_normalizer.subprocess.run')
    def test_analyze_fallback_on_failure(self, mock_run, normalizer):
        mock_run.side_effect = Exception("ffmpeg crashed")
        result = normalizer.analyze(Path("test.wav"))
        assert result == {"lufs": -23.0, "peak_db": -1.0}

    @patch('audiosmith.audio_normalizer.subprocess.run')
    def test_analyze_unparseable_output(self, mock_run, normalizer):
        mock_run.return_value = MagicMock(stderr="no match here")
        result = normalizer.analyze(Path("test.wav"))
        assert result == {"lufs": -23.0, "peak_db": -1.0}

    @patch('audiosmith.audio_normalizer.subprocess.run')
    def test_normalize_computes_gain(self, mock_run, normalizer):
        normalizer.analyze = MagicMock(return_value={"lufs": -30.0, "peak_db": -5.0})
        normalizer.normalize(Path("in.wav"), Path("out.wav"))
        cmd = " ".join(str(x) for x in mock_run.call_args[0][0])
        assert "7.0" in cmd

    @patch('audiosmith.audio_normalizer.subprocess.run')
    def test_normalize_returns_output_path(self, mock_run, normalizer):
        normalizer.analyze = MagicMock(return_value={"lufs": -23.0, "peak_db": -1.0})
        out = Path("out.wav")
        assert normalizer.normalize(Path("in.wav"), out) == out

    def test_db_to_linear(self, normalizer):
        assert normalizer._db_to_linear(0) == 1.0
        assert abs(normalizer._db_to_linear(-6) - 0.501) < 0.01

    @patch('audiosmith.audio_normalizer.subprocess.run')
    def test_custom_target_lufs(self, mock_run):
        n = AudioNormalizer(target_lufs=-16.0)
        n.analyze = MagicMock(return_value={"lufs": -20.0, "peak_db": -1.0})
        n.normalize(Path("in.wav"), Path("out.wav"))
        cmd = " ".join(str(x) for x in mock_run.call_args[0][0])
        assert "4.0" in cmd
