"""Tests for audiosmith.metrics module."""

import time
import pytest
from audiosmith.metrics import MetricsCollector


class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_timer_records_duration(self, collector):
        with collector.timer("step1"):
            time.sleep(0.01)
        assert "step1" in collector.steps
        assert collector.steps["step1"] > 0

    def test_record_step(self, collector):
        collector.record_step("x", 2.5)
        assert collector.steps["x"] == 2.5

    def test_set_segment_count(self, collector):
        collector.set_segment_count(50)
        assert collector.segment_count == 50

    def test_set_word_count(self, collector):
        collector.set_word_count(1000)
        assert collector.word_count == 1000

    def test_get_summary_wps(self, collector):
        collector.set_word_count(100)
        collector.record_step("a", 10.0)
        assert collector.get_summary()["wps"] == pytest.approx(10.0)

    def test_get_summary_total_time(self, collector):
        collector.record_step("a", 5.0)
        collector.record_step("b", 3.0)
        assert collector.get_summary()["total_time"] == 8.0

    def test_get_summary_keys(self, collector):
        summary = collector.get_summary()
        expected = {"steps", "peak_memory_gb", "total_segments", "word_count", "wps", "total_time"}
        assert set(summary.keys()) == expected

    def test_get_summary_empty(self, collector):
        summary = collector.get_summary()
        assert summary["total_time"] == 0.0
        assert summary["wps"] == 0.0

    def test_reset_clears_all(self, collector):
        collector.set_word_count(100)
        collector.record_step("x", 1.0)
        collector.set_segment_count(10)
        collector.peak_memory_gb = 2.0
        collector.reset()
        assert collector.word_count == 0
        assert collector.segment_count == 0
        assert len(collector.steps) == 0
        assert collector.peak_memory_gb == 0.0

    def test_record_memory_peak(self, collector):
        """record_memory_peak should not raise regardless of psutil availability."""
        collector.record_memory_peak()
        assert collector.peak_memory_gb >= 0.0

    def test_multiple_timers(self, collector):
        with collector.timer("a"):
            pass
        with collector.timer("b"):
            pass
        assert "a" in collector.steps
        assert "b" in collector.steps
