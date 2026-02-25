"""Tests for audiosmith.memory_manager module."""

import gc
import pytest
from unittest.mock import patch, MagicMock
from audiosmith.memory_manager import MemoryManager


class TestMemoryManager:
    @pytest.fixture
    def mgr(self):
        return MemoryManager()

    def test_get_memory_usage_keys(self, mgr):
        usage = mgr.get_memory_usage()
        assert set(usage.keys()) == {"rss_gb", "vms_gb", "available_gb", "gpu_gb"}

    def test_get_memory_usage_values_non_negative(self, mgr):
        usage = mgr.get_memory_usage()
        for v in usage.values():
            assert v >= 0.0

    def test_check_available_reasonable(self, mgr):
        # With a tiny threshold, should always pass
        assert mgr.check_available(min_gb=0.001) is True

    def test_check_available_unreasonable(self, mgr):
        # 999 TB should always fail (if psutil available) or return True (no psutil)
        result = mgr.check_available(min_gb=999999)
        assert isinstance(result, bool)

    def test_cleanup_runs(self, mgr):
        mgr.cleanup()  # should not raise

    @patch('audiosmith.memory_manager.gc.collect')
    def test_cleanup_calls_gc(self, mock_gc, mgr):
        mgr.cleanup()
        mock_gc.assert_called_once()

    def test_tracking_context(self, mgr):
        with mgr.tracking_context():
            pass
        assert mgr.peak_rss_gb >= 0.0

    def test_warn_if_high_returns_none_or_string(self, mgr):
        result = mgr.warn_if_high()
        assert result is None or isinstance(result, str)

    def test_warn_thresholds(self):
        mgr = MemoryManager(warn_threshold_gb=0.0001, critical_threshold_gb=0.0002)
        result = mgr.warn_if_high()
        # With thresholds near zero, should trigger warning if psutil available
        if result is not None:
            assert "WARNING" in result or "CRITICAL" in result

    def test_peak_rss_starts_zero(self, mgr):
        assert mgr.peak_rss_gb == 0.0

    def test_custom_thresholds(self):
        mgr = MemoryManager(warn_threshold_gb=8.0, critical_threshold_gb=12.0)
        assert mgr.warn_threshold_gb == 8.0
        assert mgr.critical_threshold_gb == 12.0
