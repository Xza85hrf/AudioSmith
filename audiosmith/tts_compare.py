"""A/B comparison tool for TTS engine selection.

Synthesizes the same text with multiple engines and computes quality metrics,
helping users pick the best engine for their use case.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import soundfile as sf

from audiosmith.tts_protocol import get_engine

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """Result from a single engine's synthesis."""

    engine_name: str
    audio: np.ndarray
    sample_rate: int
    synthesis_time_ms: float
    rtf: float  # Real-time factor: synthesis_time / audio_duration
    duration_s: float
    rms_db: float  # RMS level in dB
    peak_db: float  # Peak level in dB
    spectral_centroid_hz: float  # Brightness measure
    error: Optional[str] = None


@dataclass
class ComparisonReport:
    """Report comparing multiple engines on the same text."""

    text: str
    results: List[EngineResult] = field(default_factory=list)

    @property
    def fastest(self) -> Optional[EngineResult]:
        """Engine with lowest synthesis time."""
        valid = [r for r in self.results if r.error is None]
        return min(valid, key=lambda r: r.synthesis_time_ms) if valid else None

    @property
    def most_natural(self) -> Optional[EngineResult]:
        """Engine closest to natural speech spectral centroid (~2000-3000 Hz)."""
        valid = [r for r in self.results if r.error is None]
        if not valid:
            return None
        target = 2500.0
        return min(valid, key=lambda r: abs(r.spectral_centroid_hz - target))

    def summary(self) -> str:
        """Return a human-readable comparison summary."""
        lines = [
            f"TTS Comparison: '{self.text[:50]}...'"
            if len(self.text) > 50
            else f"TTS Comparison: '{self.text}'"
        ]
        lines.append(
            f"{'Engine':<15} {'Time':>8} {'RTF':>6} {'Duration':>8} {'RMS dB':>8} {'Centroid':>10}"
        )
        lines.append("-" * 60)
        for r in sorted(self.results, key=lambda x: x.synthesis_time_ms):
            if r.error:
                lines.append(f"{r.engine_name:<15} ERROR: {r.error}")
            else:
                lines.append(
                    f"{r.engine_name:<15} {r.synthesis_time_ms:>7.0f}ms {r.rtf:>5.2f}x "
                    f"{r.duration_s:>7.2f}s {r.rms_db:>7.1f} {r.spectral_centroid_hz:>9.0f}Hz"
                )
        return "\n".join(lines)


def _compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute spectral centroid (brightness) of audio in Hz."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    total = np.sum(spectrum)
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def _compute_rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in dB (relative to 1.0)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -100.0
    return float(20 * np.log10(rms))


def _compute_peak_db(audio: np.ndarray) -> float:
    """Compute peak level in dB."""
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return -100.0
    return float(20 * np.log10(peak))


def compare_engines(
    text: str,
    engine_names: List[str],
    output_dir: Optional[Path] = None,
    **synth_kwargs: Any,
) -> ComparisonReport:
    """Compare multiple TTS engines on the same text.

    Args:
        text: Text to synthesize.
        engine_names: List of engine names to compare.
        output_dir: Optional directory to save WAV files.
        **synth_kwargs: Extra kwargs passed to each engine's synthesize().

    Returns:
        ComparisonReport with per-engine results.
    """
    report = ComparisonReport(text=text)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for name in engine_names:
        try:
            engine = get_engine(name)
            engine.load_model()

            t0 = time.perf_counter()
            audio, sr = engine.synthesize(text, **synth_kwargs)
            synthesis_time = (time.perf_counter() - t0) * 1000  # ms

            duration = len(audio) / sr if sr > 0 else 0.0
            rtf = (synthesis_time / 1000) / duration if duration > 0 else float("inf")

            result = EngineResult(
                engine_name=name,
                audio=audio,
                sample_rate=sr,
                synthesis_time_ms=synthesis_time,
                rtf=rtf,
                duration_s=duration,
                rms_db=_compute_rms_db(audio),
                peak_db=_compute_peak_db(audio),
                spectral_centroid_hz=_compute_spectral_centroid(audio, sr),
            )

            if output_dir:
                sf.write(str(output_dir / f"{name}.wav"), audio, sr)

            engine.cleanup()

        except Exception as e:
            logger.warning("Engine %s failed: %s", name, e)
            result = EngineResult(
                engine_name=name,
                audio=np.array([], dtype=np.float32),
                sample_rate=0,
                synthesis_time_ms=0,
                rtf=0,
                duration_s=0,
                rms_db=-100,
                peak_db=-100,
                spectral_centroid_hz=0,
                error=str(e),
            )

        report.results.append(result)

    return report
