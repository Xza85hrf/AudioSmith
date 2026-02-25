"""Tests for audiosmith.transcription_post_processor module."""

import pytest
from audiosmith.transcription_post_processor import TranscriptionPostProcessor
from audiosmith.models import DubbingSegment


class TestTranscriptionPostProcessor:
    @pytest.fixture
    def processor(self):
        return TranscriptionPostProcessor()

    def test_process_single_segment(self, processor):
        segments = [DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='hello world')]
        result = processor.process(segments)
        assert len(result) > 0
        assert isinstance(result[0], DubbingSegment)

    def test_process_empty_list(self, processor):
        assert processor.process([]) == []

    def test_stage1_strings(self, processor):
        cleaned = processor.stage1_hallucination_filter(['hello uh world', 'um goodbye'])
        assert all('uh' not in s for s in cleaned)
        assert all('um' not in s for s in cleaned)

    def test_stage1_segments(self, processor):
        segments = [
            DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='hello uh world'),
        ]
        result = processor.stage1_hallucination_filter(segments)
        assert 'uh' not in result[0].original_text

    def test_stage2_short_segment(self, processor):
        segments = [DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='Short text')]
        result = processor.stage2_segment_splitter(segments)
        assert len(result) == 1

    def test_stage2_long_segment(self, processor):
        long_text = 'This is a sentence. ' * 50
        segments = [DubbingSegment(index=0, start_time=0.0, end_time=60.0, original_text=long_text)]
        result = processor.stage2_segment_splitter(segments)
        assert len(result) > 1

    def test_stage3_adds_punctuation(self, processor):
        segments = [DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='hello world')]
        result = processor.stage3_punctuation_restorer(segments)
        assert result[0].original_text.endswith('.')

    def test_stage4_labels_music(self, processor):
        segments = [DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='[MUSIC]')]
        result = processor.stage4_non_speech_labeler(segments)
        assert result[0].is_speech is False

    def test_stage4_labels_speech(self, processor):
        segments = [DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='Hello world')]
        result = processor.stage4_non_speech_labeler(segments)
        assert result[0].is_speech is True

    def test_enable_disable_stage(self, processor):
        processor.enable_stage('hallucination_filter', False)
        segments = [DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='hello uh world')]
        result = processor.process(segments)
        assert any('uh' in seg.original_text for seg in result)

    def test_full_pipeline(self, processor):
        segments = [
            DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='hello world'),
            DubbingSegment(index=1, start_time=2.0, end_time=4.0, original_text='[MUSIC]'),
        ]
        result = processor.process(segments)
        speech = [s for s in result if s.is_speech]
        non_speech = [s for s in result if not s.is_speech]
        assert len(speech) >= 1
        assert len(non_speech) >= 1
