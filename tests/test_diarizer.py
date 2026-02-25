"""Tests for audiosmith.diarizer module (no GPU required)."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from audiosmith.diarizer import Diarizer
from audiosmith.exceptions import DiarizationError


class TestDiarizerInit:
    def test_defaults(self):
        d = Diarizer()
        assert d.device == 'auto'
        assert d.min_speakers == 1
        assert d.max_speakers == 10
        assert d.merge_threshold == 0.5
        assert d._pipeline is None

    def test_custom_init(self):
        d = Diarizer(device='cpu', hf_token='tok', min_speakers=2, max_speakers=5)
        assert d.device == 'cpu'
        assert d.hf_token == 'tok'
        assert d.min_speakers == 2
        assert d.max_speakers == 5

    def test_hf_token_from_env(self, monkeypatch):
        monkeypatch.setenv('HF_TOKEN', 'env-token')
        d = Diarizer()
        assert d.hf_token == 'env-token'


class TestGracefulFallback:
    def test_load_model_missing_pyannote(self):
        d = Diarizer(hf_token='tok')
        with patch.dict('sys.modules', {'pyannote': None, 'pyannote.audio': None}):
            with pytest.raises(DiarizationError, match='pyannote-audio not installed'):
                d.load_model()

    def test_load_model_missing_token(self):
        d = Diarizer(hf_token=None)
        mock_pyannote = MagicMock()
        with patch.dict('sys.modules', {'pyannote': mock_pyannote, 'pyannote.audio': mock_pyannote}):
            with patch.dict('os.environ', {}, clear=True):
                d.hf_token = None
                with pytest.raises(DiarizationError, match='HuggingFace token required'):
                    d.load_model()


class TestMergeSegments:
    def test_empty_segments(self):
        assert Diarizer._merge_segments([], 0.5) == []

    def test_single_segment(self):
        segs = [{'speaker': 'A', 'start': 0.0, 'end': 1.0, 'duration': 1.0, 'confidence': 1.0}]
        result = Diarizer._merge_segments(segs, 0.5)
        assert len(result) == 1
        assert result[0]['speaker'] == 'A'

    def test_merge_same_speaker_within_threshold(self):
        segs = [
            {'speaker': 'A', 'start': 0.0, 'end': 1.0, 'duration': 1.0, 'confidence': 0.9},
            {'speaker': 'A', 'start': 1.3, 'end': 2.5, 'duration': 1.2, 'confidence': 0.8},
        ]
        result = Diarizer._merge_segments(segs, 0.5)
        assert len(result) == 1
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == 2.5
        assert result[0]['duration'] == pytest.approx(2.5)
        assert result[0]['confidence'] == pytest.approx(0.85)

    def test_no_merge_different_speakers(self):
        segs = [
            {'speaker': 'A', 'start': 0.0, 'end': 1.0, 'duration': 1.0, 'confidence': 1.0},
            {'speaker': 'B', 'start': 1.2, 'end': 2.5, 'duration': 1.3, 'confidence': 1.0},
        ]
        result = Diarizer._merge_segments(segs, 0.5)
        assert len(result) == 2

    def test_no_merge_gap_exceeds_threshold(self):
        segs = [
            {'speaker': 'A', 'start': 0.0, 'end': 1.0, 'duration': 1.0, 'confidence': 1.0},
            {'speaker': 'A', 'start': 2.0, 'end': 3.0, 'duration': 1.0, 'confidence': 1.0},
        ]
        result = Diarizer._merge_segments(segs, 0.5)
        assert len(result) == 2

    def test_merge_chain(self):
        """Three consecutive same-speaker segments should merge into one."""
        segs = [
            {'speaker': 'A', 'start': 0.0, 'end': 1.0, 'duration': 1.0, 'confidence': 1.0},
            {'speaker': 'A', 'start': 1.2, 'end': 2.0, 'duration': 0.8, 'confidence': 0.9},
            {'speaker': 'A', 'start': 2.3, 'end': 3.0, 'duration': 0.7, 'confidence': 0.8},
        ]
        result = Diarizer._merge_segments(segs, 0.5)
        assert len(result) == 1
        assert result[0]['end'] == 3.0


class TestApplyToTranscription:
    def test_assigns_speaker_by_overlap(self):
        trans = [
            {'text': 'Hello world', 'start': 0.0, 'end': 2.0},
            {'text': 'How are you', 'start': 2.5, 'end': 4.5},
        ]
        diar = [
            {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 2.2},
            {'speaker': 'SPEAKER_01', 'start': 2.3, 'end': 5.0},
        ]
        result = Diarizer.apply_to_transcription(trans, diar)
        assert result[0]['speaker'] == 'SPEAKER_00'
        assert result[1]['speaker'] == 'SPEAKER_01'

    def test_no_overlap_gives_none(self):
        trans = [{'text': 'Silence', 'start': 10.0, 'end': 12.0}]
        diar = [{'speaker': 'A', 'start': 0.0, 'end': 5.0}]
        result = Diarizer.apply_to_transcription(trans, diar)
        assert result[0]['speaker'] is None

    def test_partial_overlap_picks_best(self):
        trans = [{'text': 'Overlap test', 'start': 1.5, 'end': 3.5}]
        diar = [
            {'speaker': 'A', 'start': 0.0, 'end': 2.0},  # overlap: 0.5s
            {'speaker': 'B', 'start': 2.0, 'end': 5.0},  # overlap: 1.5s
        ]
        result = Diarizer.apply_to_transcription(trans, diar)
        assert result[0]['speaker'] == 'B'

    def test_empty_inputs(self):
        assert Diarizer.apply_to_transcription([], []) == []
        assert Diarizer.apply_to_transcription([], [{'speaker': 'A', 'start': 0, 'end': 1}]) == []


class TestUnload:
    def test_unload_clears_pipeline(self):
        d = Diarizer()
        d._pipeline = MagicMock()
        d.unload()
        assert d._pipeline is None

    def test_unload_when_not_loaded(self):
        d = Diarizer()
        d.unload()  # Should not raise
        assert d._pipeline is None
