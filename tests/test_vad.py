"""Tests for audiosmith.vad module."""

import pytest
from unittest.mock import MagicMock
from audiosmith.vad import SpeechSegment, VADProcessor


class TestSpeechSegment:
    def test_duration_ms(self):
        seg = SpeechSegment(start=1.0, end=3.5, confidence=0.9)
        assert seg.duration_ms == 2500

    def test_zero_duration(self):
        seg = SpeechSegment(start=1.0, end=1.0)
        assert seg.duration_ms == 0

    def test_default_confidence(self):
        seg = SpeechSegment(start=0.0, end=1.0)
        assert seg.confidence == 1.0


class TestVADProcessor:
    @pytest.fixture
    def processor(self):
        return VADProcessor()

    def test_init_defaults(self, processor):
        assert processor.threshold == 0.5
        assert processor.sample_rate == 16000
        assert processor.device == "cpu"
        assert processor._model is None

    def test_filter_silence_marks_nonspeech(self, processor):
        seg1 = MagicMock(start_time=0.0, end_time=1.0)
        seg2 = MagicMock(start_time=5.0, end_time=6.0)
        speech = [SpeechSegment(start=0.5, end=0.8)]

        processor.filter_silence([seg1, seg2], speech)
        assert seg1.is_speech is True
        assert seg2.is_speech is False

    def test_filter_silence_all_speech(self, processor):
        seg1 = MagicMock(start_time=0.0, end_time=2.0)
        speech = [SpeechSegment(start=0.0, end=3.0)]

        processor.filter_silence([seg1], speech)
        assert seg1.is_speech is True

    def test_filter_silence_empty_regions(self, processor):
        seg1 = MagicMock(start_time=0.0, end_time=1.0)
        processor.filter_silence([seg1], [])
        assert seg1.is_speech is False

    def test_unload_clears_model(self, processor):
        processor._model = MagicMock()
        processor._get_speech_ts = MagicMock()
        processor.unload()
        assert processor._model is None
        assert processor._get_speech_ts is None

    def test_unload_when_no_model(self, processor):
        processor.unload()  # should not raise
