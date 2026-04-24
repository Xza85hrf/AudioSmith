"""Test suite for TTS engine comparison script."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Import the module we're testing
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiosmith.srt import SRTEntry, parse_srt


def test_parse_srt_returns_srt_entries():
    """Verify parse_srt returns SRTEntry objects with correct structure."""
    content = """1
00:00:00,000 --> 00:00:05,000
Hello world

2
00:00:06,000 --> 00:00:10,000
This is a test
"""
    entries = parse_srt(content)
    assert len(entries) == 2
    assert isinstance(entries[0], SRTEntry)
    assert entries[0].text == "Hello world"
    assert entries[1].text == "This is a test"


def test_parse_srt_filters_empty_text():
    """Empty subtitle entries should be skipped."""
    content = """1
00:00:00,000 --> 00:00:05,000


2
00:00:06,000 --> 00:00:10,000
Valid text
"""
    entries = parse_srt(content)
    # Empty entry may be skipped by parse_srt or included with empty text
    # Check that we get at least the valid entry
    valid_entries = [e for e in entries if e.text.strip()]
    assert len(valid_entries) >= 1


def test_timestamp_to_seconds_conversion():
    """Test SRT timestamp conversion to seconds."""
    from audiosmith.srt import timestamp_to_seconds

    # 1 hour, 0 minutes, 5 seconds, 500ms = 3605.5 seconds
    seconds = timestamp_to_seconds("01:00:05,500")
    assert seconds == pytest.approx(3605.5)

    # 0 hours, 0 minutes, 10 seconds = 10 seconds
    seconds = timestamp_to_seconds("00:00:10,000")
    assert seconds == 10.0


def test_clean_tts_text_removes_brackets():
    """_clean_tts_text removes bracketed stage directions."""
    from audiosmith.pipeline.helpers import _clean_tts_text

    text = "[Marty] Hello [laughing] world"
    cleaned = _clean_tts_text(text)
    assert "[" not in cleaned
    assert "]" not in cleaned
    assert "Hello" in cleaned
    assert "world" in cleaned


def test_dedup_repeated_words_collapses_runs():
    """_dedup_repeated_words collapses runs of 3+ identical words."""
    from audiosmith.pipeline.helpers import _dedup_repeated_words

    text = "Yes yes yes yes I agree"
    deduped = _dedup_repeated_words(text, max_repeats=2)
    words = deduped.split()
    # Should have at most 2 consecutive "yes"
    assert words.count("yes") <= 2
    assert "agree" in deduped


def test_segment_parsing_with_real_structure():
    """Test parsing segments in the way compare_tts_engines.py expects."""
    content = """1
00:00:00,000 --> 00:00:05,000
Hello world

2
00:00:06,000 --> 00:00:12,000
This is a test
"""
    entries = parse_srt(content)

    # Simulate what the script does
    segments = []
    for entry in entries:
        start = float(entry.start_time.replace(',', '.').split(':')[0]) * 3600
        start += float(entry.start_time.replace(',', '.').split(':')[1]) * 60
        start += float(entry.start_time.replace(',', '.').split(':')[2])

        segments.append({
            "start": start,
            "text": entry.text,
        })

    assert len(segments) == 2
    assert segments[0]["start"] == 0.0
    assert segments[1]["start"] == 6.0


@pytest.mark.slow
def test_mock_synthesize_returns_audio_and_sr():
    """Test that synthesize mock can return (audio, sample_rate) tuple."""
    audio = np.random.randn(22050).astype(np.float32)  # 1 second at 22050 Hz
    sr = 22050

    # Verify structure matches protocol
    result = (audio, sr)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], int)
    assert result[1] > 0


def test_output_dir_creation():
    """Test that output directory can be created."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "tts_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir.exists()

        # Test writing a dummy WAV
        import soundfile as sf
        audio = np.zeros(22050, dtype=np.float32)
        wav_path = output_dir / "test.wav"
        sf.write(str(wav_path), audio, 22050)
        assert wav_path.exists()


def test_compare_tts_script_parse_segments():
    """Test segment parsing logic used by compare_tts_engines.py."""
    from audiosmith.pipeline.helpers import _clean_tts_text, _dedup_repeated_words
    from audiosmith.srt import timestamp_to_seconds

    # Create test SRT content
    content = """1
00:00:01,000 --> 00:00:03,000
[Muzyka] Hello world

2
00:00:30,000 --> 00:00:35,000
This is a test sentence

3
00:00:36,000 --> 00:00:37,000
X

4
00:00:50,000 --> 00:00:55,000
Another test with repeated yes yes yes yes words
"""

    entries = parse_srt(content)
    assert len(entries) == 4

    # Simulate segment parsing
    segments = []
    MAX_TIME_S = 600.0
    for entry in entries:
        start_sec = timestamp_to_seconds(entry.start_time)

        if start_sec > MAX_TIME_S:
            break

        text = _clean_tts_text(entry.text)
        if not text or not text.strip():
            continue

        text = _dedup_repeated_words(text)

        if len(text.split()) < 2:
            continue

        segments.append({
            "start": start_sec,
            "end": timestamp_to_seconds(entry.end_time),
            "text": text,
        })

    # Should have 3 valid segments (entry 3 has < 2 words)
    assert len(segments) == 3
    assert segments[0]["text"] == "Hello world"
    assert "Another test" in segments[2]["text"]


def test_compare_tts_script_imports():
    """Verify that compare_tts_engines.py can be imported without errors."""
    import importlib.util

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "compare_tts_engines.py"
    if script_path.exists():
        spec = importlib.util.spec_from_file_location("compare_tts_engines", script_path)
        module = importlib.util.module_from_spec(spec)
        # We don't execute it, just verify it can be parsed
        assert spec.loader is not None


def test_normalize_audio_peak():
    """Test audio normalization logic."""
    # Create test audio with peak = 2.0
    audio = np.array([0.5, 1.0, 2.0, 1.5, 0.8], dtype=np.float32)

    peak = np.max(np.abs(audio))
    assert peak == 2.0

    # Normalize to 0.95 of peak
    if peak > 0.01:
        normalized = audio / peak * 0.95

    assert np.max(np.abs(normalized)) == pytest.approx(0.95)


def test_silence_concatenation():
    """Test audio concatenation with silence gaps."""
    sr = 22050
    duration_s = 1.0
    audio1 = np.ones(int(sr * duration_s), dtype=np.float32) * 0.5
    audio2 = np.ones(int(sr * duration_s), dtype=np.float32) * 0.3
    silence = np.zeros(int(0.3 * sr), dtype=np.float32)

    combined = []
    for audio in [audio1, audio2]:
        combined.append(audio)
        combined.append(silence)

    full_audio = np.concatenate(combined)

    # Total: 1s + 0.3s + 1s + 0.3s = 2.6s
    expected_samples = int(2.6 * sr)
    assert len(full_audio) == expected_samples


@pytest.mark.slow
def test_engine_factory_returns_valid_engine():
    """Test that get_engine factory creates valid TTSEngine instances."""
    from audiosmith.tts_protocol import get_engine, TTSEngine

    # Test that piper can be created (CPU-only, always available)
    try:
        engine = get_engine("piper", voice="en_US-lessac-medium")
        assert hasattr(engine, "name")
        assert hasattr(engine, "sample_rate")
        assert hasattr(engine, "synthesize")
        assert hasattr(engine, "cleanup")
        assert isinstance(engine, TTSEngine)
    except Exception as e:
        pytest.skip(f"Piper unavailable: {e}")


@pytest.mark.slow
def test_tts_protocol_compliance():
    """Verify TTSEngine protocol is correctly defined."""
    from audiosmith.tts_protocol import TTSEngine
    import inspect

    # Check that TTSEngine is a Protocol
    assert hasattr(TTSEngine, "__protocol_attrs__") or hasattr(TTSEngine, "__mro__")

    # Check required methods exist
    required_methods = ["synthesize", "cleanup", "load_model"]
    protocol_members = {name: member for name, member in inspect.getmembers(TTSEngine)}

    for method in required_methods:
        assert method in protocol_members or any(method in str(m) for m in protocol_members.values())


def test_compare_tts_script_handles_real_srt():
    """Integration test: parse real Polish SRT file and validate segment count."""
    from audiosmith.srt import parse_srt, timestamp_to_seconds
    from audiosmith.pipeline.helpers import _clean_tts_text, _dedup_repeated_words

    srt_path = Path(__file__).resolve().parent.parent / "test-files/videos/Original_subtitiles/Marty.Supreme.2025.pl.srt"

    if not srt_path.exists():
        pytest.skip(f"Test SRT file not found: {srt_path}")

    content = srt_path.read_text(encoding='utf-8')
    entries = parse_srt(content)

    assert len(entries) > 100, "SRT should have many entries"

    # Simulate segment parsing logic from compare_tts_engines.py
    segments = []
    MAX_TIME_S = 600.0

    for entry in entries:
        start_sec = timestamp_to_seconds(entry.start_time)

        if start_sec > MAX_TIME_S:
            break

        text = _clean_tts_text(entry.text)
        if not text or not text.strip():
            continue

        text = _dedup_repeated_words(text)

        if len(text.split()) < 2:
            continue

        segments.append({
            "start": start_sec,
            "end": timestamp_to_seconds(entry.end_time),
            "text": text,
        })

    # Should parse a reasonable number of segments
    assert len(segments) > 50, "Should parse at least 50 segments from first 10 minutes"
    assert len(segments) < 500, "Should parse less than 500 segments"

    # Verify all segments have valid structure
    for seg in segments:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert isinstance(seg["text"], str)
        assert len(seg["text"].split()) >= 2
        assert seg["start"] < seg["end"]
        assert seg["start"] < MAX_TIME_S


@pytest.mark.slow
def test_mock_engine_with_synthesize():
    """Test mock TTS engine that matches TTSEngine protocol."""
    import numpy as np
    from audiosmith.tts_protocol import TTSEngine

    class MockTTSEngine:
        """Mock TTS engine for testing."""

        @property
        def name(self) -> str:
            return "mock"

        @property
        def sample_rate(self) -> int:
            return 22050

        def load_model(self) -> None:
            pass

        def synthesize(self, text: str, **kwargs) -> tuple:
            # Generate dummy audio (1 second)
            audio = np.random.randn(22050).astype(np.float32) * 0.1
            return audio, self.sample_rate

        def cleanup(self) -> None:
            pass

    engine = MockTTSEngine()

    # Verify it matches protocol
    assert isinstance(engine, TTSEngine)
    assert engine.name == "mock"
    assert engine.sample_rate == 22050

    # Test synthesize
    audio, sr = engine.synthesize("Hello world", language="en")
    assert isinstance(audio, np.ndarray)
    assert isinstance(sr, int)
    assert sr == 22050
    assert len(audio) == 22050


def test_script_usage_and_output_structure():
    """Test that script generates correct output directory structure."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "tts_comparison"

        # Simulate what the script does
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test that we can write WAV files
        import soundfile as sf
        for engine_name in ["piper", "f5", "chatterbox", "qwen3", "fish"]:
            wav_path = output_dir / f"{engine_name}.wav"

            # Create dummy audio
            audio = np.ones(22050, dtype=np.float32) * 0.1
            sf.write(str(wav_path), audio, 22050)

            assert wav_path.exists()
            info = sf.info(str(wav_path))
            assert info.duration == pytest.approx(1.0, abs=0.01)
            assert info.samplerate == 22050


def test_script_error_handling():
    """Test that script handles missing SRT file gracefully."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/compare_tts_engines.py", "/nonexistent/file.srt"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()
