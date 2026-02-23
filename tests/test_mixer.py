"""Tests for audiosmith.mixer module."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from audiosmith.mixer import AudioMixer
from audiosmith.models import DubbingConfig, DubbingSegment, ScheduledSegment


@pytest.fixture
def config(tmp_path):
    return DubbingConfig(video_path=Path('v.mp4'), output_dir=tmp_path)


@pytest.fixture
def mixer(config):
    return AudioMixer(config)


class TestSchedule:
    def test_empty(self, mixer):
        assert mixer.schedule([]) == []

    def test_skip_no_audio(self, mixer):
        seg = DubbingSegment(index=0, start_time=0, end_time=1, original_text='hi')
        # No tts_audio_path or tts_duration_ms
        result = mixer.schedule([seg])
        assert len(result) == 0

    def test_single_segment(self, mixer):
        seg = DubbingSegment(
            index=0, start_time=1.0, end_time=3.0, original_text='hi',
            tts_audio_path=Path('seg.wav'), tts_duration_ms=1500,
        )
        result = mixer.schedule([seg])
        assert len(result) == 1
        assert result[0].segment is seg
        assert result[0].speed_factor >= 1.0

    def test_speedup_applied(self, mixer):
        seg = DubbingSegment(
            index=0, start_time=1.0, end_time=2.0, original_text='hi',
            tts_audio_path=Path('seg.wav'), tts_duration_ms=3000,
        )
        result = mixer.schedule([seg])
        assert result[0].speed_factor > 1.0
        assert result[0].speed_factor <= mixer.max_speedup


class TestRender:
    def test_empty_schedule(self, mixer):
        buf = mixer.render([], 5.0)
        assert buf.shape == (5 * 48000, 2)
        assert np.allclose(buf, 0.0)
