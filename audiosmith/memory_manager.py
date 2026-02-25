"""Memory monitoring and cleanup utilities."""

import gc
import logging
from contextlib import contextmanager
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """Monitors memory usage and provides cleanup utilities."""

    def __init__(
        self,
        warn_threshold_gb: float = 4.0,
        critical_threshold_gb: float = 6.0,
    ) -> None:
        self.warn_threshold_gb = warn_threshold_gb
        self.critical_threshold_gb = critical_threshold_gb
        self.peak_rss_gb: float = 0.0

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB."""
        result = {"rss_gb": 0.0, "vms_gb": 0.0, "available_gb": 0.0, "gpu_gb": 0.0}
        try:
            import psutil
            mem = psutil.Process().memory_info()
            result["rss_gb"] = mem.rss / (1024 ** 3)
            result["vms_gb"] = mem.vms / (1024 ** 3)
            result["available_gb"] = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        except ImportError:
            pass
        return result

    def check_available(self, min_gb: float = 1.0) -> bool:
        """Check if at least min_gb of system memory is available."""
        try:
            import psutil
            return psutil.virtual_memory().available >= min_gb * (1024 ** 3)
        except ImportError:
            return True

    def cleanup(self) -> None:
        """Force garbage collection and clear GPU cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Memory cleanup completed")

    @contextmanager
    def tracking_context(self):
        """Track peak RSS memory during a block."""
        try:
            yield
        finally:
            try:
                import psutil
                rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
                if rss_gb > self.peak_rss_gb:
                    self.peak_rss_gb = rss_gb
            except ImportError:
                pass

    def warn_if_high(self) -> Optional[str]:
        """Return warning string if memory exceeds thresholds, else None."""
        try:
            import psutil
            rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        except ImportError:
            return None

        if rss_gb >= self.critical_threshold_gb:
            return f"CRITICAL: Memory {rss_gb:.2f} GB exceeds {self.critical_threshold_gb:.2f} GB"
        if rss_gb >= self.warn_threshold_gb:
            return f"WARNING: Memory {rss_gb:.2f} GB exceeds {self.warn_threshold_gb:.2f} GB"
        return None
