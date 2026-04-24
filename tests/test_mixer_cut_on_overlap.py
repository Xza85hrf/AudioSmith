"""Tests for the cut-on-overlap scheduling mode in AudioMixer."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from audiosmith.mixer import AudioMixer
from audiosmith.models import DubbingConfig, DubbingSegment, ScheduledSegment


@pytest.fixture
def config_with_cut_on_overlap() -> DubbingConfig:
    """Create a config with cut_on_overlap enabled."""
    config = DubbingConfig(
        video_path=Path('/tmp/test.mp4'),
        output_dir=Path('/tmp/output'),
        cut_on_overlap=True,
    )
    return config


@pytest.fixture
def config_without_cut_on_overlap() -> DubbingConfig:
    """Create a config with cut_on_overlap disabled (default)."""
    config = DubbingConfig(
        video_path=Path('/tmp/test.mp4'),
        output_dir=Path('/tmp/output'),
        cut_on_overlap=False,
    )
    return config


def create_test_segment(
    index: int,
    start_ms: int,
    end_ms: int,
    tts_duration_ms: int,
) -> DubbingSegment:
    """Helper to create a test segment with TTS audio path."""
    seg = DubbingSegment(
        index=index,
        start_time=start_ms / 1000.0,
        end_time=end_ms / 1000.0,
        original_text=f'segment {index}',
    )
    # Fake path (won't be read in scheduling tests)
    seg.tts_audio_path = Path(f'/fake/seg_{index}.wav')
    seg.tts_duration_ms = tts_duration_ms
    return seg


class TestScheduleCutOnOverlapBasic:
    """Basic scheduling: segments placed at original times, speed=1.0."""

    def test_single_segment_placed_at_start(self, config_with_cut_on_overlap):
        """Single segment placed at its original start time with speed 1.0."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [create_test_segment(0, 0, 1000, 800)]  # 1s window, 800ms TTS

        scheduled = mixer.schedule_cut_on_overlap(segments)

        assert len(scheduled) == 1
        sch = scheduled[0]
        assert sch.place_at_ms == 0
        assert sch.speed_factor == 1.0
        assert sch.actual_duration_ms == 800

    def test_two_non_overlapping_segments(self, config_with_cut_on_overlap):
        """Two segments with gaps: each placed at original time, unchanged duration."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 1000, 800),  # 0-1s: 800ms TTS
            create_test_segment(1, 2000, 3000, 900),  # 2-3s: 900ms TTS
        ]

        scheduled = mixer.schedule_cut_on_overlap(segments)

        assert len(scheduled) == 2
        assert scheduled[0].place_at_ms == 0
        assert scheduled[0].actual_duration_ms == 800
        assert scheduled[1].place_at_ms == 2000
        assert scheduled[1].actual_duration_ms == 900

    def test_all_segments_have_speed_1_0(self, config_with_cut_on_overlap):
        """All segments maintain 1.0x speed (no speedup)."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 1000, 800),
            create_test_segment(1, 2000, 3000, 900),
            create_test_segment(2, 4000, 5000, 700),
        ]

        scheduled = mixer.schedule_cut_on_overlap(segments)

        for sch in scheduled:
            assert sch.speed_factor == 1.0


class TestScheduleCutOnOverlapTrimming:
    """Test trimming: earlier segment cut where next segment starts."""

    def test_overlap_trimmed_to_next_start(self, config_with_cut_on_overlap):
        """Seg A (0-1s, 1000ms TTS) overlaps with B (800-2s).
        A should be trimmed to 800ms (until B starts).
        """
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 1000, 1000),  # 0-1s: 1000ms TTS (would play 0-1s)
            create_test_segment(1, 800, 2000, 900),  # 0.8-2s: 900ms TTS
        ]

        scheduled = mixer.schedule_cut_on_overlap(segments)

        assert len(scheduled) == 2
        # Seg A placed at 0, should be trimmed to 800ms (next seg starts at 800)
        assert scheduled[0].place_at_ms == 0
        assert scheduled[0].actual_duration_ms == 800
        assert scheduled[1].place_at_ms == 800
        assert scheduled[1].actual_duration_ms == 900

    def test_three_segments_middle_trimmed(self, config_with_cut_on_overlap):
        """Three overlapping segments: only middle is trimmed."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 1000, 1000),   # 0-1s: 1000ms TTS
            create_test_segment(1, 500, 2000, 1500),  # 0.5-2s: 1500ms TTS (overlaps 0)
            create_test_segment(2, 1500, 3000, 800),  # 1.5-3s: 800ms TTS
        ]

        scheduled = mixer.schedule_cut_on_overlap(segments)

        assert len(scheduled) == 3
        # Seg 0: placed at 0, trimmed to 500ms (seg 1 starts at 500)
        assert scheduled[0].place_at_ms == 0
        assert scheduled[0].actual_duration_ms == 500
        # Seg 1: placed at 500, trimmed to 1000ms (seg 2 starts at 1500)
        assert scheduled[1].place_at_ms == 500
        assert scheduled[1].actual_duration_ms == 1000
        # Seg 2: placed at 1500, no trim (last segment)
        assert scheduled[2].place_at_ms == 1500
        assert scheduled[2].actual_duration_ms == 800

    def test_no_trim_when_no_overlap(self, config_with_cut_on_overlap):
        """No trimming needed when segments don't overlap."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 500, 400),  # 0-0.5s: 400ms TTS
            create_test_segment(1, 1000, 2000, 900),  # 1-2s: 900ms TTS (gap after 0)
        ]

        scheduled = mixer.schedule_cut_on_overlap(segments)

        assert len(scheduled) == 2
        assert scheduled[0].actual_duration_ms == 400  # No trim
        assert scheduled[1].actual_duration_ms == 900  # No trim


class TestScheduleCutOnOverlapEdgeCases:
    """Edge cases: empty segments, zero durations, invalid data."""

    def test_skip_segments_without_tts_path(self, config_with_cut_on_overlap):
        """Segments without tts_audio_path are skipped."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        seg_with_path = create_test_segment(0, 0, 1000, 800)
        seg_no_path = DubbingSegment(
            index=1,
            start_time=2.0,
            end_time=3.0,
            original_text='no tts',
        )
        # seg_no_path has no tts_audio_path

        scheduled = mixer.schedule_cut_on_overlap([seg_with_path, seg_no_path])

        assert len(scheduled) == 1
        assert scheduled[0].segment.index == 0

    def test_skip_segments_with_zero_duration(self, config_with_cut_on_overlap):
        """Segments with tts_duration_ms <= 0 are skipped."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        seg_valid = create_test_segment(0, 0, 1000, 800)
        seg_zero = create_test_segment(1, 2000, 3000, 0)

        scheduled = mixer.schedule_cut_on_overlap([seg_valid, seg_zero])

        assert len(scheduled) == 1
        assert scheduled[0].segment.index == 0

    def test_segment_trimmed_to_zero(self, config_with_cut_on_overlap):
        """If next segment starts before current placement, trim to 0."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 1000, 2000),  # 0-1s: 2000ms TTS (overflows)
            create_test_segment(1, 1000, 2000, 900),  # 1-2s (placed at same time)
        ]

        scheduled = mixer.schedule_cut_on_overlap(segments)

        # Seg 0: placed at 0, ends at 0+2000=2000, seg 1 starts at 1000
        # Seg 0 should be trimmed to 1000ms
        assert scheduled[0].actual_duration_ms == 1000
        # Ensure trimming never goes negative
        assert scheduled[0].actual_duration_ms >= 0

    def test_empty_segment_list(self, config_with_cut_on_overlap):
        """Empty segment list returns empty schedule."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        scheduled = mixer.schedule_cut_on_overlap([])
        assert len(scheduled) == 0

    def test_all_segments_skipped(self, config_with_cut_on_overlap):
        """If all segments are invalid, return empty schedule."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            DubbingSegment(
                index=0,
                start_time=0.0,
                end_time=1.0,
                original_text='no tts',
            ),
            DubbingSegment(
                index=1,
                start_time=2.0,
                end_time=3.0,
                original_text='also no tts',
            ),
        ]
        scheduled = mixer.schedule_cut_on_overlap(segments)
        assert len(scheduled) == 0


class TestScheduleCutOnOverlapIntegration:
    """Integration tests: full workflow with mixer config."""

    def test_mixer_config_attribute(self, config_with_cut_on_overlap):
        """Mixer correctly reads cut_on_overlap from config."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        assert mixer.cut_on_overlap is True

    def test_mixer_config_default_false(self, config_without_cut_on_overlap):
        """Mixer defaults cut_on_overlap to False when not set."""
        mixer = AudioMixer(config_without_cut_on_overlap)
        assert mixer.cut_on_overlap is False

    def test_many_segments_complex_overlap(self, config_with_cut_on_overlap):
        """Complex case: 5 segments with varied overlap patterns."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 1000, 1200),     # 0-1s: 1200ms TTS (ends at 1200)
            create_test_segment(1, 800, 2000, 1500),   # 0.8-2s: 1500ms TTS (overlaps 0)
            create_test_segment(2, 2000, 3000, 1000),  # 2-3s: 1000ms TTS (overlaps 1)
            create_test_segment(3, 2500, 4000, 1500),  # 2.5-4s: 1500ms TTS (overlaps 2)
            create_test_segment(4, 5000, 6000, 900),   # 5-6s: 900ms TTS (gap)
        ]

        scheduled = mixer.schedule_cut_on_overlap(segments)

        assert len(scheduled) == 5
        # Seg 0: placed at 0, ends at 1200, seg 1 starts at 800, trimmed to 800
        assert scheduled[0].place_at_ms == 0
        assert scheduled[0].actual_duration_ms == 800
        # Seg 1: placed at 800, ends at 2300, seg 2 starts at 2000, trimmed to 1200
        assert scheduled[1].place_at_ms == 800
        assert scheduled[1].actual_duration_ms == 1200
        # Seg 2: placed at 2000, ends at 3000, seg 3 starts at 2500, trimmed to 500
        assert scheduled[2].place_at_ms == 2000
        assert scheduled[2].actual_duration_ms == 500
        # Seg 3: placed at 2500, ends at 4000, seg 4 starts at 5000, no trim
        assert scheduled[3].place_at_ms == 2500
        assert scheduled[3].actual_duration_ms == 1500
        # Seg 4: placed at 5000, no trim (last)
        assert scheduled[4].place_at_ms == 5000
        assert scheduled[4].actual_duration_ms == 900


class TestScheduleCutOnOverlapVsDefault:
    """Verify cut-on-overlap produces different results than default scheduler."""

    def test_cut_on_overlap_no_speedup(self, config_with_cut_on_overlap):
        """Cut-on-overlap never speeds up (speed=1.0)."""
        mixer = AudioMixer(config_with_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 500, 600),  # 500ms window, 600ms TTS
        ]

        scheduled = mixer.schedule_cut_on_overlap(segments)

        # Should NOT be sped up; cut-on-overlap keeps speed at 1.0
        assert scheduled[0].speed_factor == 1.0
        assert scheduled[0].actual_duration_ms == 600  # Full duration preserved

    def test_default_scheduler_would_speedup(self, config_without_cut_on_overlap):
        """Default scheduler speeds up to fit tight window."""
        mixer = AudioMixer(config_without_cut_on_overlap)
        segments = [
            create_test_segment(0, 0, 500, 600),  # 500ms window, 600ms TTS
        ]

        scheduled = mixer.schedule(segments)

        # Default scheduler would speed up
        assert scheduled[0].speed_factor > 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
