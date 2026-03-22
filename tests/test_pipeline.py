"""Tests for audiosmith.pipeline module."""

from pathlib import Path

import pytest

from audiosmith.emotion_config import EMOTION_STYLE_MAP
from audiosmith.models import DubbingConfig, DubbingSegment, PipelineState
from audiosmith.pipeline import CHECKPOINT_FILE, DubbingPipeline, _emotion_to_tts_params


@pytest.fixture
def config(tmp_path):
    return DubbingConfig(video_path=Path('video.mp4'), output_dir=tmp_path)


class TestEmotionToTtsParams:
    def test_known_emotion_returns_scaled_params(self):
        result = _emotion_to_tts_params('happy', intensity=1.0)
        assert result['exaggeration'] == pytest.approx(0.7)
        assert result['cfg_weight'] == pytest.approx(0.5)

    def test_unknown_emotion_returns_defaults(self):
        result = _emotion_to_tts_params('nonexistent', intensity=1.0)
        assert result['exaggeration'] == pytest.approx(0.5)
        assert result['cfg_weight'] == pytest.approx(0.5)

    def test_zero_intensity_collapses_to_midpoint(self):
        result = _emotion_to_tts_params('angry', intensity=0.0)
        assert result['exaggeration'] == pytest.approx(0.5)
        assert result['cfg_weight'] == pytest.approx(0.5)


class TestBuildSynthesisKwargs:
    """Regression tests for EMOTION_STYLE_MAP usage in _build_synthesis_kwargs."""

    def test_elevenlabs_style_uses_emotion_style_map(self, config):
        p = DubbingPipeline(config)
        seg = DubbingSegment(
            index=0, start_time=0.0, end_time=1.0,
            original_text='hello', translated_text='cześć',
            metadata={'emotion': {'primary': 'happy'}},
        )
        kwargs = p._build_synthesis_kwargs('elevenlabs', seg, None, False, 24000)
        assert kwargs['style'] == EMOTION_STYLE_MAP['happy']

    def test_elevenlabs_style_defaults_for_unknown_emotion(self, config):
        p = DubbingPipeline(config)
        seg = DubbingSegment(
            index=0, start_time=0.0, end_time=1.0,
            original_text='hello', translated_text='cześć',
            metadata={'emotion': {'primary': 'unknown_emotion'}},
        )
        kwargs = p._build_synthesis_kwargs('elevenlabs', seg, None, False, 24000)
        assert kwargs['style'] == 0.0  # default fallback

    def test_no_emotion_data_omits_style(self, config):
        p = DubbingPipeline(config)
        seg = DubbingSegment(
            index=0, start_time=0.0, end_time=1.0,
            original_text='hello', translated_text='cześć',
        )
        kwargs = p._build_synthesis_kwargs('elevenlabs', seg, None, False, 24000)
        assert 'style' not in kwargs


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
