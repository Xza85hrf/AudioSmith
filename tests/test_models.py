"""Tests for audiosmith.models module."""

from pathlib import Path

from audiosmith.models import (DubbingConfig, DubbingSegment, DubbingStep,
                               PipelineState, ScheduledSegment)


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

    def test_fish_config_defaults(self, tmp_path):
        """Test Fish Speech config defaults."""
        cfg = DubbingConfig(video_path=Path('v.mp4'), output_dir=tmp_path)
        assert cfg.fish_reference_id is None
        assert cfg.fish_base_url is None
        assert cfg.fish_backend == 'speech-1.6'
        assert cfg.fish_temperature == 0.7
        assert cfg.fish_top_p == 0.7

    def test_fish_config_cloud_mode(self, tmp_path):
        """Test Fish Speech cloud mode (no base_url)."""
        cfg = DubbingConfig(
            video_path=Path('v.mp4'), output_dir=tmp_path,
            fish_reference_id='ref-123',
            fish_backend='speech-1.6',
            fish_temperature=0.5,
            fish_top_p=0.8,
        )
        assert cfg.fish_reference_id == 'ref-123'
        assert cfg.fish_base_url is None  # Cloud mode
        assert cfg.fish_backend == 'speech-1.6'
        assert cfg.fish_temperature == 0.5
        assert cfg.fish_top_p == 0.8

    def test_fish_config_local_mode(self, tmp_path):
        """Test Fish Speech local server mode."""
        cfg = DubbingConfig(
            video_path=Path('v.mp4'), output_dir=tmp_path,
            fish_base_url='http://localhost:8080',
            fish_backend='speech-1.6',
        )
        assert cfg.fish_base_url == 'http://localhost:8080'
        assert cfg.fish_reference_id is None
        assert cfg.fish_backend == 'speech-1.6'

    def test_fish_config_custom_local_url(self, tmp_path):
        """Test Fish Speech with custom local server URL."""
        cfg = DubbingConfig(
            video_path=Path('v.mp4'), output_dir=tmp_path,
            fish_base_url='http://192.168.1.100:9090',
        )
        assert cfg.fish_base_url == 'http://192.168.1.100:9090'


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
        assert 'merge_segments' in steps
        assert len(steps) == 11


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
