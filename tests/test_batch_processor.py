"""Tests for audiosmith.batch_processor module."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from audiosmith.batch_processor import BatchProcessor, BatchResult
from audiosmith.models import DubbingConfig, DubbingResult
from audiosmith.exceptions import BatchProcessingError


class TestBatchResult:
    def test_defaults(self):
        r = BatchResult(file_path=Path("x.mp4"), success=True)
        assert r.result is None
        assert r.error is None
        assert r.duration_seconds == 0.0


class TestBatchProcessor:
    @pytest.fixture
    def config(self):
        return DubbingConfig(video_path=Path("input.mp4"), output_dir=Path("/tmp/out"))

    @pytest.fixture
    def processor(self):
        return BatchProcessor()

    @patch("audiosmith.pipeline.DubbingPipeline")
    def test_process_single_file(self, MockPipeline, processor, config):
        MockPipeline.return_value.run.return_value = DubbingResult(success=True)
        results = processor.process([Path("f1.mp4")], config)
        assert len(results) == 1
        assert results[0].success is True

    @patch("audiosmith.pipeline.DubbingPipeline")
    def test_process_multiple_files(self, MockPipeline, processor, config):
        MockPipeline.return_value.run.return_value = DubbingResult(success=True)
        results = processor.process([Path("a.mp4"), Path("b.mp4"), Path("c.mp4")], config)
        assert len(results) == 3
        assert all(r.success for r in results)

    @patch("audiosmith.pipeline.DubbingPipeline")
    def test_process_with_failure(self, MockPipeline, processor, config):
        call_count = [0]

        def side_effect(fpath):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("bad file")
            return DubbingResult(success=True)

        MockPipeline.return_value.run.side_effect = side_effect
        results = processor.process(
            [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")], config, continue_on_error=True,
        )
        assert results[0].success is True
        assert results[1].success is False
        assert "bad file" in results[1].error
        assert results[2].success is True

    @patch("audiosmith.pipeline.DubbingPipeline")
    def test_stop_on_error(self, MockPipeline, processor, config):
        MockPipeline.return_value.run.side_effect = [
            DubbingResult(success=True),
            RuntimeError("fail"),
        ]
        with pytest.raises(BatchProcessingError):
            processor.process(
                [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")], config, continue_on_error=False,
            )

    @patch("audiosmith.pipeline.DubbingPipeline")
    def test_progress_callback(self, MockPipeline, processor, config):
        MockPipeline.return_value.run.return_value = DubbingResult(success=True)
        cb = MagicMock()
        processor.process([Path("a.mp4"), Path("b.mp4")], config, progress_callback=cb)
        assert cb.call_count >= 2

    def test_get_summary_mixed(self):
        results = [
            BatchResult(Path("a"), True, duration_seconds=5.0),
            BatchResult(Path("b"), False, error="err", duration_seconds=2.0),
            BatchResult(Path("c"), True, duration_seconds=3.0),
        ]
        s = BatchProcessor.get_summary(results)
        assert s["total"] == 3
        assert s["succeeded"] == 2
        assert s["failed"] == 1
        assert s["total_duration_seconds"] == 10.0
        assert str(Path("b")) in s["failed_files"]

    def test_get_summary_all_success(self):
        results = [BatchResult(Path("a"), True), BatchResult(Path("b"), True)]
        assert BatchProcessor.get_summary(results)["failed"] == 0

    def test_get_summary_empty(self):
        s = BatchProcessor.get_summary([])
        assert s["total"] == 0
        assert s["avg_duration_seconds"] == 0.0

    def test_adjust_config(self):
        config = DubbingConfig(video_path=Path("x.mp4"), output_dir=Path("/tmp/out"))
        adjusted = BatchProcessor._adjust_config(config, Path("my_video.mp4"))
        assert "my_video" in str(adjusted.output_dir)
