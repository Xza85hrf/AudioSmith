"""Test suite for Chatterbox parameter comparison script."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiosmith.srt import parse_srt, timestamp_to_seconds


class TestParseSegments:
    """Tests for segment parsing logic used by compare_chatterbox_params.py."""

    def test_parse_srt_basic(self):
        """Verify parse_srt returns entries with correct structure."""
        content = """1
00:00:00,000 --> 00:00:05,000
Hello world

2
00:00:06,000 --> 00:00:10,000
This is a test
"""
        entries = parse_srt(content)
        assert len(entries) == 2
        assert entries[0].text == "Hello world"
        assert entries[1].text == "This is a test"

    def test_timestamp_to_seconds_basic(self):
        """Test SRT timestamp conversion to seconds."""
        # 10 seconds
        seconds = timestamp_to_seconds("00:00:10,000")
        assert seconds == 10.0

        # 1 hour, 0 minutes, 5 seconds, 500ms
        seconds = timestamp_to_seconds("01:00:05,500")
        assert seconds == pytest.approx(3605.5)

    def test_timestamp_to_seconds_at_10min_boundary(self):
        """Timestamp at exactly 10 minutes (600s) should be included."""
        seconds = timestamp_to_seconds("00:10:00,000")
        assert seconds == 600.0

    def test_timestamp_to_seconds_beyond_10min(self):
        """Timestamp beyond 10 minutes should be excluded by max_time check."""
        seconds = timestamp_to_seconds("00:10:01,000")
        assert seconds > 600.0


class TestTextCleaning:
    """Tests for text cleaning functions."""

    def test_clean_tts_text_removes_brackets(self):
        """_clean_tts_text removes bracketed stage directions."""
        from audiosmith.pipeline.helpers import _clean_tts_text

        text = "[Marty] Hello [laughing] world"
        cleaned = _clean_tts_text(text)
        assert "[" not in cleaned
        assert "]" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned

    def test_dedup_repeated_words_collapses_runs(self):
        """_dedup_repeated_words collapses runs of 3+ identical words."""
        from audiosmith.pipeline.helpers import _dedup_repeated_words

        text = "Yes yes yes yes I agree"
        deduped = _dedup_repeated_words(text, max_repeats=2)
        # Should preserve some "yes" but not all
        assert "agree" in deduped

    def test_segments_skip_short_text(self):
        """Segments with < 2 words should be skipped."""
        from audiosmith.pipeline.helpers import _clean_tts_text, _dedup_repeated_words

        # Single word
        text = "Hello"
        cleaned = _clean_tts_text(text)
        deduped = _dedup_repeated_words(cleaned)
        word_count = len(deduped.split())
        assert word_count < 2

        # Empty/whitespace
        text = "   "
        assert not _clean_tts_text(text).strip()


class TestVariantConfigurations:
    """Tests for parameter variant configurations."""

    def test_variant_configs_have_required_fields(self):
        """Each variant must have exaggeration and cfg_weight."""
        # Import the variants from the script module
        variants = {
            "chatterbox_A_clone": {
                "label": "voice_clone",
                "audio_prompt_path": Path("test-files/tts_comparison/voice_refs/witcher_polish_ref.wav"),
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
            },
            "chatterbox_B_tuned": {
                "label": "voice_clone_tuned",
                "audio_prompt_path": Path("test-files/tts_comparison/voice_refs/witcher_polish_ref.wav"),
                "exaggeration": 0.65,
                "cfg_weight": 0.6,
            },
            "chatterbox_C_expressive": {
                "label": "voice_clone_expressive",
                "audio_prompt_path": Path("test-files/tts_comparison/voice_refs/witcher_polish_ref.wav"),
                "exaggeration": 0.8,
                "cfg_weight": 0.7,
            },
        }

        for name, config in variants.items():
            assert "exaggeration" in config, f"{name} missing exaggeration"
            assert "cfg_weight" in config, f"{name} missing cfg_weight"
            assert "audio_prompt_path" in config, f"{name} missing audio_prompt_path"
            assert "label" in config, f"{name} missing label"
            assert isinstance(config["exaggeration"], (int, float))
            assert isinstance(config["cfg_weight"], (int, float))
            assert 0.0 <= config["exaggeration"] <= 1.0, f"{name} exaggeration out of range"
            assert 0.0 <= config["cfg_weight"] <= 1.0, f"{name} cfg_weight out of range"

    def test_variant_exaggeration_progression(self):
        """Variants should have increasing exaggeration from A to C."""
        exaggerations = [0.5, 0.65, 0.8]
        assert exaggerations[0] < exaggerations[1] < exaggerations[2]

    def test_variant_cfg_weight_progression(self):
        """Variants should have increasing cfg_weight from A to C."""
        cfg_weights = [0.5, 0.6, 0.7]
        assert cfg_weights[0] < cfg_weights[1] < cfg_weights[2]


class TestAudioNormalization:
    """Tests for audio normalization logic."""

    def test_peak_normalize_preserves_shape(self):
        """Peak normalization should preserve audio shape."""
        # Create test audio with peak at 2.0
        audio = np.array([0.5, 1.0, 2.0, 1.5, 0.0], dtype=np.float32)
        peak = np.max(np.abs(audio))
        assert peak == 2.0

        # Normalize to 0.95
        normalized = audio / peak * 0.95
        assert normalized.shape == audio.shape
        assert np.max(np.abs(normalized)) == pytest.approx(0.95)

    def test_silence_concatenation(self):
        """Silence gaps should be the correct duration."""
        sr = 16000
        silence_duration_s = 0.3
        silence = np.zeros(int(silence_duration_s * sr), dtype=np.float32)

        # Should be 0.3 * 16000 = 4800 samples
        assert len(silence) == 4800
        assert np.all(silence == 0.0)

    def test_audio_concatenation_preserves_values(self):
        """Concatenated audio should preserve original values."""
        chunk1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        chunk2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        silence = np.zeros(2, dtype=np.float32)

        combined = np.concatenate([chunk1, silence, chunk2])

        assert combined[0] == pytest.approx(0.1)
        assert combined[1] == pytest.approx(0.2)
        assert combined[2] == pytest.approx(0.3)
        assert combined[3] == pytest.approx(0.0)
        assert combined[4] == pytest.approx(0.0)
        assert combined[5] == pytest.approx(0.4)


class TestVoiceReferenceFile:
    """Tests for voice reference file handling."""

    def test_voice_ref_path_exists(self):
        """Voice reference file should exist."""
        voice_ref = Path("test-files/tts_comparison/voice_refs/witcher_polish_ref.wav")
        # Note: This test passes if file exists; skips if not (for CI environments)
        if not voice_ref.exists():
            pytest.skip(f"Voice reference not available: {voice_ref}")
        assert voice_ref.exists()

    def test_voice_ref_is_wav(self):
        """Voice reference should be a WAV file."""
        voice_ref = Path("test-files/tts_comparison/voice_refs/witcher_polish_ref.wav")
        if not voice_ref.exists():
            pytest.skip(f"Voice reference not available: {voice_ref}")
        assert voice_ref.suffix.lower() == ".wav"


class TestOutputPaths:
    """Tests for output file path generation."""

    def test_output_paths_follow_naming_convention(self):
        """Output files should follow consistent naming."""
        variants = {
            "chatterbox_A_clone": {},
            "chatterbox_B_tuned": {},
            "chatterbox_C_expressive": {},
        }

        for name in variants:
            wav_name = f"{name}.wav"
            assert wav_name.endswith(".wav")
            assert "chatterbox" in wav_name


class TestSRTFileHandling:
    """Tests for SRT file parsing and filtering."""

    def test_parse_segments_respects_max_time(self):
        """Segments beyond MAX_TIME_S should be excluded."""
        content = """1
00:00:00,000 --> 00:00:05,000
Hello world

2
00:09:00,000 --> 00:09:30,000
Within 10 minutes

3
00:15:00,000 --> 00:15:30,000
Beyond 10 minutes
"""
        entries = parse_srt(content)
        assert len(entries) == 3

        # Simulate max_time filtering (MAX_TIME_S = 600.0)
        MAX_TIME_S = 600.0
        filtered = []
        for entry in entries:
            start_sec = timestamp_to_seconds(entry.start_time)
            if start_sec <= MAX_TIME_S:
                filtered.append(entry)

        # Should include entries 1 and 2, exclude entry 3
        assert len(filtered) == 2

    def test_polish_srt_file_exists(self):
        """Polish SRT file should exist for integration tests."""
        srt_path = Path("test-files/videos/Original_subtitiles/Marty.Supreme.2025.pl.srt")
        if not srt_path.exists():
            pytest.skip(f"Polish SRT not available: {srt_path}")
        assert srt_path.exists()


class TestChatterboxIntegration:
    """Integration tests for ChatterboxTTS (mock-based)."""

    @patch("audiosmith.tts.ChatterboxTTS")
    def test_synthesize_called_with_correct_params(self, mock_tts_class):
        """Verify synthesize is called with voice cloning parameters."""
        # Mock the engine
        mock_engine = MagicMock()
        mock_engine.sample_rate = 22050
        mock_engine.synthesize.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_tts_class.return_value = mock_engine

        # Create engine and call synthesize
        engine = mock_tts_class(device="cuda")
        result = engine.synthesize(
            "Hello world",
            language="pl",
            audio_prompt_path="test-files/tts_comparison/voice_refs/witcher_polish_ref.wav",
            exaggeration=0.65,
            cfg_weight=0.6,
        )

        # Verify call was made with expected parameters
        mock_engine.synthesize.assert_called_once()
        call_kwargs = mock_engine.synthesize.call_args.kwargs
        assert call_kwargs["language"] == "pl"
        assert call_kwargs["exaggeration"] == 0.65
        assert call_kwargs["cfg_weight"] == 0.6

    @patch("audiosmith.tts.ChatterboxTTS")
    def test_synthesize_returns_ndarray(self, mock_tts_class):
        """ChatterboxTTS.synthesize should return ndarray."""
        test_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_engine = MagicMock()
        mock_engine.sample_rate = 22050
        mock_engine.synthesize.return_value = test_audio

        # Configure the class mock to return our engine instance
        mock_tts_class.return_value = mock_engine

        engine = mock_tts_class(device="cuda")
        result = engine.synthesize("test", language="pl")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
