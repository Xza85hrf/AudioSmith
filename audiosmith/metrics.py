"""Metrics collection for pipeline performance monitoring."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects performance metrics for audio processing pipelines."""

    def __init__(self) -> None:
        self.steps: Dict[str, float] = {}
        self.segment_count: int = 0
        self.word_count: int = 0
        self.peak_memory_gb: float = 0.0

    @contextmanager
    def timer(self, name: str):
        """Context manager to time a named operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.steps[name] = time.perf_counter() - start

    def record_step(self, name: str, duration: float) -> None:
        """Manually record a step duration in seconds."""
        self.steps[name] = duration

    def set_segment_count(self, count: int) -> None:
        self.segment_count = count

    def set_word_count(self, count: int) -> None:
        self.word_count = count

    def record_memory_peak(self) -> None:
        """Capture current RSS and update peak if higher."""
        try:
            import psutil
            rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
            if rss_gb > self.peak_memory_gb:
                self.peak_memory_gb = rss_gb
        except ImportError:
            pass

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated metrics summary."""
        total_time = sum(self.steps.values())
        return {
            "steps": dict(self.steps),
            "peak_memory_gb": self.peak_memory_gb,
            "total_segments": self.segment_count,
            "word_count": self.word_count,
            "wps": self.word_count / total_time if total_time > 0 else 0.0,
            "total_time": total_time,
        }

    def reset(self) -> None:
        """Clear all collected metrics."""
        self.steps.clear()
        self.segment_count = 0
        self.word_count = 0
        self.peak_memory_gb = 0.0
