"""Tests for audiosmith.transcribe module."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from audiosmith.transcribe import Transcriber


class TestTranscriber:
    def test_defaults(self):
        t = Transcriber()
        assert t.model == 'large-v3'
        assert t.device == 'cuda'
        assert t.batch_size == 16

    def test_custom_init(self):
        t = Transcriber(model='base', device='cpu', batch_size=8)
        assert t.model == 'base'
        assert t.device == 'cpu'
        assert t.batch_size == 8

    @patch('audiosmith.transcribe.Transcriber.load_model')
    def test_transcribe_returns_segments(self, mock_load):
        t = Transcriber()

        # Mock the batched pipeline
        mock_seg = MagicMock()
        mock_seg.text = '  Hello world  '
        mock_seg.start = 0.0
        mock_seg.end = 2.5
        mock_seg.words = []

        mock_info = MagicMock()
        mock_info.language = 'en'

        t._batched = MagicMock()
        t._batched.transcribe.return_value = ([mock_seg], mock_info)

        result = t.transcribe(Path('test.wav'))
        assert len(result) == 1
        assert result[0]['text'] == 'Hello world'
        assert result[0]['start'] == 0.0
        assert result[0]['language'] == 'en'

    def test_unload(self):
        t = Transcriber()
        t._model = MagicMock()
        t._batched = MagicMock()
        t.unload()
        assert t._model is None
        assert t._batched is None
