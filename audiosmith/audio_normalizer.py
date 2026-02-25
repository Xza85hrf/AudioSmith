"""Audio normalization using LUFS measurement via ffmpeg."""

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class AudioNormalizer:
    """LUFS-based audio normalizer using ffmpeg ebur128 filter."""

    def __init__(self, target_lufs: float = -23.0, max_peak_db: float = -1.0) -> None:
        self.target_lufs = target_lufs
        self.max_peak_db = max_peak_db

    def analyze(self, audio_path: Path) -> Dict[str, float]:
        """Analyze audio loudness using ffmpeg ebur128 filter.

        Returns dict with 'lufs' and 'peak_db' keys.
        """
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-af", "ebur128", "-f", "null", "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            stderr = result.stderr

            lufs_match = re.search(r"I:\s+([-\d.]+)\s+LUFS", stderr)
            peak_match = re.search(r"Peak:\s+([-\d.]+)\s+dBFS", stderr)

            lufs = float(lufs_match.group(1)) if lufs_match else self.target_lufs
            peak_db = float(peak_match.group(1)) if peak_match else self.max_peak_db
            return {"lufs": lufs, "peak_db": peak_db}
        except Exception as e:
            logger.warning("Analysis failed for %s: %s", audio_path, e)
            return {"lufs": self.target_lufs, "peak_db": self.max_peak_db}

    def normalize(self, input_path: Path, output_path: Path) -> Path:
        """Normalize audio to target LUFS with peak limiting."""
        analysis = self.analyze(input_path)
        gain_db = self.target_lufs - analysis["lufs"]
        peak_linear = self._db_to_linear(self.max_peak_db)

        logger.info(
            "Normalizing %s: current=%.1f LUFS, gain=%.2f dB",
            input_path.name, analysis["lufs"], gain_db,
        )

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"volume={gain_db}dB,alimiter=limit={peak_linear}",
            "-ar", "48000", str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path

    @staticmethod
    def _db_to_linear(db: float) -> float:
        """Convert decibels to linear gain factor."""
        return 10 ** (db / 20)
