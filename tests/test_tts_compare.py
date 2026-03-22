"""Tests for TTS engine comparison module."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from audiosmith.tts_compare import (
    EngineResult,
    ComparisonReport,
    compare_engines,
    _compute_spectral_centroid,
    _compute_rms_db,
    _compute_peak_db,
)


class TestEngineResult:
    """Test EngineResult dataclass."""

    def test_engine_result_creation(self):
        """Create an EngineResult with all fields."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = EngineResult(
            engine_name="test_engine",
            audio=audio,
            sample_rate=24000,
            synthesis_time_ms=150.0,
            rtf=0.5,
            duration_s=0.3,
            rms_db=-20.0,
            peak_db=-10.0,
            spectral_centroid_hz=2500.0,
        )
        assert result.engine_name == "test_engine"
        assert np.array_equal(result.audio, audio)
        assert result.sample_rate == 24000
        assert result.synthesis_time_ms == 150.0
        assert result.rtf == 0.5
        assert result.duration_s == 0.3
        assert result.rms_db == -20.0
        assert result.peak_db == -10.0
        assert result.spectral_centroid_hz == 2500.0
        assert result.error is None

    def test_engine_result_with_error(self):
        """Create an EngineResult with error field."""
        result = EngineResult(
            engine_name="failed_engine",
            audio=np.array([], dtype=np.float32),
            sample_rate=0,
            synthesis_time_ms=0,
            rtf=0,
            duration_s=0,
            rms_db=-100,
            peak_db=-100,
            spectral_centroid_hz=0,
            error="Model loading failed",
        )
        assert result.error == "Model loading failed"


class TestComparisonReport:
    """Test ComparisonReport dataclass and properties."""

    def test_report_creation(self):
        """Create a ComparisonReport with results."""
        text = "Hello world"
        result1 = EngineResult(
            engine_name="engine1",
            audio=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=100.0,
            rtf=0.4,
            duration_s=0.25,
            rms_db=-20.0,
            peak_db=-10.0,
            spectral_centroid_hz=2000.0,
        )
        result2 = EngineResult(
            engine_name="engine2",
            audio=np.array([0.2], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=150.0,
            rtf=0.6,
            duration_s=0.25,
            rms_db=-18.0,
            peak_db=-8.0,
            spectral_centroid_hz=2600.0,
        )
        report = ComparisonReport(text=text, results=[result1, result2])
        assert report.text == text
        assert len(report.results) == 2

    def test_fastest_property(self):
        """Verify fastest property returns engine with lowest synthesis_time_ms."""
        result1 = EngineResult(
            engine_name="slow_engine",
            audio=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=200.0,
            rtf=0.8,
            duration_s=0.25,
            rms_db=-20.0,
            peak_db=-10.0,
            spectral_centroid_hz=2000.0,
        )
        result2 = EngineResult(
            engine_name="fast_engine",
            audio=np.array([0.2], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=50.0,
            rtf=0.2,
            duration_s=0.25,
            rms_db=-18.0,
            peak_db=-8.0,
            spectral_centroid_hz=2600.0,
        )
        report = ComparisonReport(text="test", results=[result1, result2])
        assert report.fastest is not None
        assert report.fastest.engine_name == "fast_engine"
        assert report.fastest.synthesis_time_ms == 50.0

    def test_fastest_with_errors(self):
        """Fastest should skip engines with errors."""
        result_ok = EngineResult(
            engine_name="ok_engine",
            audio=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=100.0,
            rtf=0.4,
            duration_s=0.25,
            rms_db=-20.0,
            peak_db=-10.0,
            spectral_centroid_hz=2000.0,
        )
        result_err = EngineResult(
            engine_name="failed_engine",
            audio=np.array([], dtype=np.float32),
            sample_rate=0,
            synthesis_time_ms=0,
            rtf=0,
            duration_s=0,
            rms_db=-100,
            peak_db=-100,
            spectral_centroid_hz=0,
            error="Failed",
        )
        report = ComparisonReport(text="test", results=[result_err, result_ok])
        assert report.fastest is not None
        assert report.fastest.engine_name == "ok_engine"

    def test_fastest_with_all_errors(self):
        """Fastest returns None if all results have errors."""
        result1 = EngineResult(
            engine_name="engine1",
            audio=np.array([], dtype=np.float32),
            sample_rate=0,
            synthesis_time_ms=0,
            rtf=0,
            duration_s=0,
            rms_db=-100,
            peak_db=-100,
            spectral_centroid_hz=0,
            error="Error 1",
        )
        result2 = EngineResult(
            engine_name="engine2",
            audio=np.array([], dtype=np.float32),
            sample_rate=0,
            synthesis_time_ms=0,
            rtf=0,
            duration_s=0,
            rms_db=-100,
            peak_db=-100,
            spectral_centroid_hz=0,
            error="Error 2",
        )
        report = ComparisonReport(text="test", results=[result1, result2])
        assert report.fastest is None

    def test_most_natural_property(self):
        """Most natural should return engine closest to 2500 Hz centroid."""
        result_low = EngineResult(
            engine_name="low_freq",
            audio=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=100.0,
            rtf=0.4,
            duration_s=0.25,
            rms_db=-20.0,
            peak_db=-10.0,
            spectral_centroid_hz=1000.0,  # Far from 2500
        )
        result_target = EngineResult(
            engine_name="natural",
            audio=np.array([0.2], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=150.0,
            rtf=0.6,
            duration_s=0.25,
            rms_db=-18.0,
            peak_db=-8.0,
            spectral_centroid_hz=2550.0,  # Close to 2500
        )
        result_high = EngineResult(
            engine_name="high_freq",
            audio=np.array([0.3], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=120.0,
            rtf=0.5,
            duration_s=0.25,
            rms_db=-19.0,
            peak_db=-9.0,
            spectral_centroid_hz=4000.0,  # Far from 2500
        )
        report = ComparisonReport(
            text="test", results=[result_low, result_target, result_high]
        )
        assert report.most_natural is not None
        assert report.most_natural.engine_name == "natural"

    def test_most_natural_with_errors(self):
        """Most natural should skip engines with errors."""
        result_err = EngineResult(
            engine_name="failed",
            audio=np.array([], dtype=np.float32),
            sample_rate=0,
            synthesis_time_ms=0,
            rtf=0,
            duration_s=0,
            rms_db=-100,
            peak_db=-100,
            spectral_centroid_hz=0,
            error="Failed",
        )
        result_ok = EngineResult(
            engine_name="ok",
            audio=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=100.0,
            rtf=0.4,
            duration_s=0.25,
            rms_db=-20.0,
            peak_db=-10.0,
            spectral_centroid_hz=2500.0,
        )
        report = ComparisonReport(text="test", results=[result_err, result_ok])
        assert report.most_natural is not None
        assert report.most_natural.engine_name == "ok"

    def test_summary_output(self):
        """Summary should return formatted string with all results."""
        result1 = EngineResult(
            engine_name="engine1",
            audio=np.array([0.1], dtype=np.float32),
            sample_rate=24000,
            synthesis_time_ms=100.0,
            rtf=0.4,
            duration_s=0.25,
            rms_db=-20.0,
            peak_db=-10.0,
            spectral_centroid_hz=2000.0,
        )
        result2 = EngineResult(
            engine_name="failed_engine",
            audio=np.array([], dtype=np.float32),
            sample_rate=0,
            synthesis_time_ms=0,
            rtf=0,
            duration_s=0,
            rms_db=-100,
            peak_db=-100,
            spectral_centroid_hz=0,
            error="Connection timeout",
        )
        report = ComparisonReport(text="Hello world", results=[result1, result2])
        summary = report.summary()
        assert "TTS Comparison" in summary
        assert "Hello world" in summary
        assert "engine1" in summary
        assert "failed_engine" in summary
        assert "ERROR" in summary
        assert "Connection timeout" in summary


class TestSpectralCentroid:
    """Test _compute_spectral_centroid function."""

    def test_spectral_centroid_sine_wave(self):
        """Spectral centroid of pure sine should be at or near sine frequency."""
        sr = 24000
        freq = 1000.0  # 1000 Hz sine wave
        duration = 0.5  # 500ms for better FFT resolution
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        centroid = _compute_spectral_centroid(audio, sr)
        # Should be reasonably close to 1000 Hz (spectral leakage causes variance)
        # Allow wider tolerance for FFT resolution and window effects
        assert 800 < centroid < 2000

    def test_spectral_centroid_silence(self):
        """Spectral centroid of silence should be 0."""
        audio = np.zeros(1000, dtype=np.float32)
        centroid = _compute_spectral_centroid(audio, 24000)
        assert centroid == 0.0

    def test_spectral_centroid_different_frequencies(self):
        """Higher frequency signals should have higher centroid."""
        sr = 24000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        audio_low = np.sin(2 * np.pi * 500 * t).astype(np.float32)
        audio_high = np.sin(2 * np.pi * 2000 * t).astype(np.float32)

        centroid_low = _compute_spectral_centroid(audio_low, sr)
        centroid_high = _compute_spectral_centroid(audio_high, sr)

        assert centroid_high > centroid_low


class TestRMSDB:
    """Test _compute_rms_db function."""

    def test_rms_db_silence(self):
        """RMS of silence should be -100 dB."""
        audio = np.zeros(1000, dtype=np.float32)
        rms_db = _compute_rms_db(audio)
        assert rms_db == -100.0

    def test_rms_db_unit_amplitude(self):
        """RMS of unit sine wave is ~0.707 (-3 dB)."""
        sr = 24000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = np.sin(2 * np.pi * t).astype(np.float32)
        rms_db = _compute_rms_db(audio)
        # RMS of unit sine is sqrt(2)/2 ≈ 0.707, which is ~-3 dB
        assert -4 < rms_db < -2

    def test_rms_db_higher_amplitude(self):
        """Higher amplitude signals should have higher RMS dB."""
        sr = 24000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio_low = (0.1 * np.sin(2 * np.pi * t)).astype(np.float32)
        audio_high = (0.5 * np.sin(2 * np.pi * t)).astype(np.float32)
        rms_low = _compute_rms_db(audio_low)
        rms_high = _compute_rms_db(audio_high)
        assert rms_high > rms_low


class TestPeakDB:
    """Test _compute_peak_db function."""

    def test_peak_db_silence(self):
        """Peak of silence should be -100 dB."""
        audio = np.zeros(1000, dtype=np.float32)
        peak_db = _compute_peak_db(audio)
        assert peak_db == -100.0

    def test_peak_db_unit_amplitude(self):
        """Peak of unit amplitude signal should be 0 dB."""
        audio = np.array([1.0, 0.5, -1.0, 0.0], dtype=np.float32)
        peak_db = _compute_peak_db(audio)
        # 20 * log10(1.0) = 0 dB
        assert -0.1 < peak_db < 0.1

    def test_peak_db_half_amplitude(self):
        """Peak of 0.5 amplitude should be -6 dB."""
        audio = np.array([0.5, 0.25, -0.5, 0.0], dtype=np.float32)
        peak_db = _compute_peak_db(audio)
        # 20 * log10(0.5) ≈ -6.02 dB
        assert -6.5 < peak_db < -5.5


class TestCompareEngines:
    """Test compare_engines function."""

    def test_compare_engines_basic(self):
        """Compare multiple engines with mocked get_engine."""
        text = "Test synthesis"
        engine_names = ["engine1", "engine2"]

        # Create mock engines
        mock_engine1 = Mock()
        mock_engine1.name = "engine1"
        mock_engine1.sample_rate = 24000
        mock_engine1.load_model = Mock()
        mock_engine1.cleanup = Mock()
        audio1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_engine1.synthesize = Mock(return_value=(audio1, 24000))

        mock_engine2 = Mock()
        mock_engine2.name = "engine2"
        mock_engine2.sample_rate = 24000
        mock_engine2.load_model = Mock()
        mock_engine2.cleanup = Mock()
        audio2 = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        mock_engine2.synthesize = Mock(return_value=(audio2, 24000))

        with patch(
            "audiosmith.tts_compare.get_engine",
            side_effect=lambda name: mock_engine1 if name == "engine1" else mock_engine2,
        ):
            report = compare_engines(text, engine_names)

        assert report.text == text
        assert len(report.results) == 2
        assert all(r.error is None for r in report.results)
        assert report.results[0].engine_name == "engine1"
        assert report.results[1].engine_name == "engine2"
        assert report.results[0].sample_rate == 24000
        assert report.results[1].sample_rate == 24000

    def test_compare_engines_with_exception(self):
        """Engine that raises exception should have error field."""
        text = "Test text"
        engine_names = ["working", "broken"]

        mock_working = Mock()
        mock_working.name = "working"
        mock_working.load_model = Mock()
        mock_working.cleanup = Mock()
        mock_working.synthesize = Mock(return_value=(np.array([0.1], dtype=np.float32), 24000))

        mock_broken = Mock()
        mock_broken.name = "broken"
        mock_broken.load_model = Mock(side_effect=RuntimeError("Model not found"))

        def get_engine_side_effect(name):
            if name == "working":
                return mock_working
            return mock_broken

        with patch("audiosmith.tts_compare.get_engine", side_effect=get_engine_side_effect):
            report = compare_engines(text, engine_names)

        assert len(report.results) == 2
        working_result = [r for r in report.results if r.engine_name == "working"][0]
        broken_result = [r for r in report.results if r.engine_name == "broken"][0]

        assert working_result.error is None
        assert broken_result.error is not None
        assert "Model not found" in broken_result.error

    def test_compare_engines_with_output_dir(self):
        """Compare with output_dir should call sf.write for each engine."""
        text = "Test"
        engine_names = ["engine1"]
        output_dir = Path("/tmp/test_output")

        mock_engine = Mock()
        mock_engine.name = "engine1"
        mock_engine.load_model = Mock()
        mock_engine.cleanup = Mock()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_engine.synthesize = Mock(return_value=(audio, 24000))

        with patch("audiosmith.tts_compare.get_engine", return_value=mock_engine):
            with patch("audiosmith.tts_compare.sf.write") as mock_write:
                with patch("audiosmith.tts_compare.Path.mkdir"):
                    report = compare_engines(text, engine_names, output_dir=output_dir)

        assert report.results[0].error is None
        mock_write.assert_called_once()
        call_args = mock_write.call_args
        assert "engine1.wav" in call_args[0][0]
        assert np.array_equal(call_args[0][1], audio)
        assert call_args[0][2] == 24000

    def test_compare_engines_with_synth_kwargs(self):
        """Synthesize kwargs should be passed to each engine."""
        text = "Test"
        engine_names = ["engine1"]

        mock_engine = Mock()
        mock_engine.name = "engine1"
        mock_engine.load_model = Mock()
        mock_engine.cleanup = Mock()
        mock_engine.synthesize = Mock(return_value=(np.array([0.1], dtype=np.float32), 24000))

        with patch("audiosmith.tts_compare.get_engine", return_value=mock_engine):
            report = compare_engines(
                text,
                engine_names,
                voice="woman",
                speed=1.2,
                language="en",
            )

        mock_engine.synthesize.assert_called_once_with(
            text,
            voice="woman",
            speed=1.2,
            language="en",
        )
        assert report.results[0].error is None

    def test_compare_engines_rtf_calculation(self):
        """RTF (real-time factor) should be computed correctly."""
        text = "Test"
        engine_names = ["engine1"]

        mock_engine = Mock()
        mock_engine.name = "engine1"
        mock_engine.load_model = Mock()
        mock_engine.cleanup = Mock()

        # 0.3 seconds of audio at 24000 Hz
        audio = np.zeros(7200, dtype=np.float32)  # 7200 samples / 24000 = 0.3 s
        mock_engine.synthesize = Mock(return_value=(audio, 24000))

        # Patch time.perf_counter to simulate 300ms synthesis time (0.3s)
        with patch("audiosmith.tts_compare.get_engine", return_value=mock_engine):
            with patch("audiosmith.tts_compare.time.perf_counter") as mock_time:
                # First call returns 0, second call returns 0.3 (300ms later)
                mock_time.side_effect = [0, 0.3]
                report = compare_engines(text, engine_names)

        result = report.results[0]
        assert result.duration_s == pytest.approx(0.3)
        assert result.synthesis_time_ms == pytest.approx(300.0)
        assert result.rtf == pytest.approx(1.0)  # 0.3s synthesis / 0.3s audio = 1.0x
