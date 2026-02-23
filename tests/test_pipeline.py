"""Tests for audiosmith.pipeline module."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from audiosmith.pipeline import DubbingPipeline, CHECKPOINT_FILE
from audiosmith.models import DubbingConfig, DubbingSegment, PipelineState
from audiosmith.exceptions import DubbingError


@pytest.fixture
def config(tmp_path):
    return DubbingConfig(video_path=Path('video.mp4'), output_dir=tmp_path)


class TestDubbingPipelineInit:
    def test_fresh_start(self, config):
        p = DubbingPipeline(config)
        assert p.state.completed_steps == []

    def test_resume_loads_checkpoint(self, tmp_path):
        state = PipelineState()
        state.mark_step_done('extract_audio')
        state.audio_path = '/tmp/audio.wav'
        state.save(tmp_path / CHECKPOINT_FILE)

        cfg = DubbingConfig(
            video_path=Path('v.mp4'), output_dir=tmp_path, resume=True,
        )
        p = DubbingPipeline(cfg)
        assert p.state.is_step_done('extract_audio')
        assert p.state.audio_path == '/tmp/audio.wav'

    def test_resume_no_checkpoint(self, tmp_path):
        cfg = DubbingConfig(
            video_path=Path('v.mp4'), output_dir=tmp_path, resume=True,
        )
        p = DubbingPipeline(cfg)
        assert p.state.completed_steps == []


class TestHelpers:
    def test_segments_roundtrip(self, config):
        p = DubbingPipeline(config)
        segments = [
            DubbingSegment(index=0, start_time=0.0, end_time=1.5, original_text='hi',
                          translated_text='cześć'),
            DubbingSegment(index=1, start_time=2.0, end_time=3.0, original_text='bye',
                          tts_audio_path=Path('/tmp/seg.wav'), tts_duration_ms=800),
        ]
        dicts = p._segments_to_dicts(segments)
        assert len(dicts) == 2
        assert dicts[0]['translated_text'] == 'cześć'
        assert dicts[1]['tts_audio_path'] == '/tmp/seg.wav'

        restored = p._dicts_to_segments(dicts)
        assert restored[0].translated_text == 'cześć'
        assert restored[1].tts_audio_path == Path('/tmp/seg.wav')
        assert restored[1].tts_duration_ms == 800

    def test_write_srt(self, config, tmp_path):
        p = DubbingPipeline(config)
        segments = [
            DubbingSegment(index=0, start_time=0.0, end_time=1.0,
                          original_text='hello', translated_text='cześć'),
        ]
        srt_path = tmp_path / 'test.srt'
        p._write_srt(segments, srt_path)
        content = srt_path.read_text()
        assert 'cześć' in content
        assert '-->' in content
