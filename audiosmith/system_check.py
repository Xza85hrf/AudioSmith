"""Pre-flight system validation — checks FFmpeg, GPU, disk space, dependencies."""

import logging
import shutil
from typing import Dict

logger = logging.getLogger(__name__)


class SystemChecker:
    """Verify system requirements before processing."""

    def check_ffmpeg(self) -> bool:
        return shutil.which('ffmpeg') is not None

    def check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def check_torch(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def check_faster_whisper(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def check_disk_space(self, min_gb: float = 10.0) -> bool:
        stat = shutil.disk_usage('/')
        return stat.free / (1024 ** 3) >= min_gb

    def run_all_checks(self) -> Dict[str, bool]:
        return {
            'ffmpeg': self.check_ffmpeg(),
            'torch': self.check_torch(),
            'cuda': self.check_cuda(),
            'faster_whisper': self.check_faster_whisper(),
            'disk_space': self.check_disk_space(),
        }

    def get_summary(self, results: Dict[str, bool]) -> str:
        lines = ['System Pre-Flight Checks:']
        for key, passed in results.items():
            status = 'PASS' if passed else 'FAIL'
            lines.append(f'  {key:20s} {status}')
        return '\n'.join(lines)
