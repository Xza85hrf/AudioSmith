"""Tests for audiosmith.system_check module."""

import pytest
from audiosmith.system_check import SystemChecker


class TestSystemChecker:
    @pytest.fixture
    def checker(self):
        return SystemChecker()

    def test_check_ffmpeg(self, checker):
        assert isinstance(checker.check_ffmpeg(), bool)

    def test_check_torch(self, checker):
        assert isinstance(checker.check_torch(), bool)

    def test_check_cuda(self, checker):
        assert isinstance(checker.check_cuda(), bool)

    def test_check_faster_whisper(self, checker):
        assert isinstance(checker.check_faster_whisper(), bool)

    def test_check_disk_space_passes(self, checker):
        assert checker.check_disk_space(min_gb=0.001) is True

    def test_check_disk_space_fails(self, checker):
        assert checker.check_disk_space(min_gb=999999) is False

    def test_run_all_checks(self, checker):
        results = checker.run_all_checks()
        assert isinstance(results, dict)
        for key in ('ffmpeg', 'torch', 'cuda', 'faster_whisper', 'disk_space'):
            assert key in results
            assert isinstance(results[key], bool)

    def test_get_summary(self, checker):
        results = checker.run_all_checks()
        summary = checker.get_summary(results)
        assert isinstance(summary, str)
        assert 'Pre-Flight' in summary
        assert 'PASS' in summary or 'FAIL' in summary
