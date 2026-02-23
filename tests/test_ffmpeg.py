"""Tests for audiosmith.ffmpeg module."""

import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from audiosmith.ffmpeg import probe_duration, extract_audio, encode_video
from audiosmith.exceptions import DubbingError


class TestProbeDuration:
    @patch('audiosmith.ffmpeg.subprocess.run')
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout='120.5\n', stderr='', returncode=0
        )
        dur = probe_duration(Path('video.mp4'))
        assert dur == 120.5

    @patch('audiosmith.ffmpeg.subprocess.run')
    def test_error_raises(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ffprobe')
        with pytest.raises(DubbingError):
            probe_duration(Path('video.mp4'))


class TestExtractAudio:
    @patch('audiosmith.ffmpeg.subprocess.run')
    def test_success(self, mock_run, tmp_path):
        out = tmp_path / 'audio.wav'
        out.write_bytes(b'\x00' * 100)
        mock_run.return_value = MagicMock(returncode=0)
        extract_audio(Path('video.mp4'), out)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert 'ffmpeg' in cmd[0]

    @patch('audiosmith.ffmpeg.subprocess.run')
    def test_failure_raises(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ffmpeg', stderr=b'error')
        with pytest.raises(DubbingError):
            extract_audio(Path('video.mp4'), tmp_path / 'out.wav')


class TestEncodeVideo:
    @patch('audiosmith.ffmpeg.subprocess.run')
    def test_without_subtitles(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        encode_video(
            Path('video.mp4'), Path('audio.wav'),
            tmp_path / 'out.mp4', subtitle_path=None,
        )
        mock_run.assert_called_once()
