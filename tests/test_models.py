"""Tests for audiosmith.models module."""

import json
import pytest
from pathlib import Path

from audiosmith.models import (
    DubbingConfig, DubbingSegment, DubbingStep, DubbingResult,
    PipelineState, ScheduledSegment,
)


class TestDubbingConfig:
    def test_defaults(self, tmp_path):
        cfg = DubbingConfig(video_path=Path('v.mp4'), output_dir=tmp_path)
        assert cfg.source_language == 'en'
        assert cfg.target_language == 'pl'
        assert cfg.resume is False
        assert cfg.whisper_model == 'large-v3'

    def test_custom_values(self, tmp_path):
        cfg = DubbingConfig(
            video_path=Path('v.mp4'), output_dir=tmp_path,
            target_language='es', resume=True, max_speedup=2.0,
        )
        assert cfg.target_language == 'es'
        assert cfg.resume is True
        assert cfg.max_speedup == 2.0


class TestDubbingSegment:
    def test_duration_ms(self):
        seg = DubbingSegment(index=0, start_time=1.0, end_time=3.5, original_text='hi')
        assert seg.duration_ms == 2500

    def test_defaults(self):
        seg = DubbingSegment(index=0, start_time=0, end_time=1, original_text='a')
        assert seg.translated_text == ''
        assert seg.tts_audio_path is None
        assert seg.is_speech is True


class TestDubbingStep:
    def test_all_steps(self):
        steps = [s.value for s in DubbingStep]
        assert 'extract_audio' in steps
        assert 'transcribe' in steps
        assert 'encode_video' in steps
        assert 'isolate_vocals' in steps
        assert 'diarize' in steps
        assert 'detect_emotion' in steps
        assert 'post_process' in steps
        assert len(steps) == 10


class TestPipelineState:
    def test_mark_and_check(self):
        state = PipelineState()
        assert not state.is_step_done('transcribe')
        state.mark_step_done('transcribe')
        assert state.is_step_done('transcribe')

    def test_no_duplicates(self):
        state = PipelineState()
        state.mark_step_done('x')
        state.mark_step_done('x')
        assert state.completed_steps.count('x') == 1

    def test_save_load_roundtrip(self, tmp_path):
        state = PipelineState()
        state.mark_step_done('extract_audio')
        state.mark_step_done('transcribe')
        state.audio_path = '/tmp/audio.wav'
        state.duration = 120.5

        path = tmp_path / 'checkpoint.json'
        state.save(path)

        loaded = PipelineState.load(path)
        assert loaded.completed_steps == ['extract_audio', 'transcribe']
        assert loaded.audio_path == '/tmp/audio.wav'
        assert loaded.duration == 120.5


class TestScheduledSegment:
    def test_basic(self):
        seg = DubbingSegment(index=0, start_time=0, end_time=1, original_text='hi')
        ss = ScheduledSegment(segment=seg, place_at_ms=500, speed_factor=1.2, actual_duration_ms=800)
        assert ss.place_at_ms == 500
        assert ss.speed_factor == 1.2
