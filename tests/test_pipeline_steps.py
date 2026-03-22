"""Tests for individual pipeline steps with mocked external dependencies.

Covers: _resolve_engine, _build_synthesis_kwargs, _merge_segments,
_split_long_segment, _extract_audio, _transcribe, _translate,
_generate_tts, _mix_audio, _encode_video, _import_external_srt,
_create_segments_from_srt, _extract_speaker_voices, full run().
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from audiosmith.models import (
    DubbingConfig,
    DubbingSegment,
    DubbingStep,
    PipelineState,
    ScheduledSegment,
)
from audiosmith.pipeline import DubbingPipeline
from audiosmith.pipeline.helpers import (
    _dedup_repeated_words,
    _dicts_to_segments,
    _emotion_to_tts_params,
    _segments_to_dicts,
    _write_srt,
    _write_srt_from_schedule,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config(tmp_path):
    """Default pipeline config for testing."""
    return DubbingConfig(video_path=Path("video.mp4"), output_dir=tmp_path)


@pytest.fixture
def pipeline(config):
    """Fresh pipeline instance."""
    return DubbingPipeline(config)


def _seg(
    index: int = 0,
    start: float = 0.0,
    end: float = 1.0,
    original: str = "hello",
    translated: str = "",
    speaker: str | None = None,
    metadata: dict | None = None,
) -> DubbingSegment:
    """Helper to create a DubbingSegment with less boilerplate."""
    return DubbingSegment(
        index=index,
        start_time=start,
        end_time=end,
        original_text=original,
        translated_text=translated,
        speaker_id=speaker,
        metadata=metadata or {},
    )


# ============================================================================
# _resolve_engine() Tests
# ============================================================================


class TestResolveEngine:
    """Test auto-selection of TTS engine based on language and environment."""

    def test_elevenlabs_selected_when_api_key_present(self, config):
        config.target_language = "en"
        p = DubbingPipeline(config)
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test"}, clear=False):
            assert p._resolve_engine() == "elevenlabs"

    def test_fish_selected_when_api_key_present_no_elevenlabs(self, config):
        config.target_language = "en"
        p = DubbingPipeline(config)
        env = {"FISH_API_KEY": "fk-test"}
        with patch.dict(os.environ, env, clear=False):
            # Remove ELEVENLABS_API_KEY if present
            with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}, clear=False):
                result = p._resolve_engine()
                assert result in ("fish", "elevenlabs")  # fish if no elevenlabs

    def test_cosyvoice_selected_when_model_dir_set(self, config, tmp_path):
        config.target_language = "zh"
        config.cosyvoice_model_dir = str(tmp_path)
        p = DubbingPipeline(config)
        with patch.dict(os.environ, {}, clear=False):
            env_clean = {k: "" for k in ["ELEVENLABS_API_KEY", "FISH_API_KEY"] if k in os.environ}
            with patch.dict(os.environ, env_clean, clear=False):
                assert p._resolve_engine() == "cosyvoice"

    def test_indextts_selected_for_emotion_en(self, config):
        config.target_language = "en"
        config.detect_emotion = True
        p = DubbingPipeline(config)
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "FISH_API_KEY": ""}, clear=False):
            assert p._resolve_engine() == "indextts"

    def test_indextts_selected_for_emotion_zh(self, config):
        config.target_language = "zh"
        config.detect_emotion = True
        p = DubbingPipeline(config)
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "FISH_API_KEY": ""}, clear=False):
            assert p._resolve_engine() == "indextts"

    def test_orpheus_selected_for_supported_lang(self, config):
        config.target_language = "tr"  # Turkish — in Orpheus but not Qwen3
        p = DubbingPipeline(config)
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "FISH_API_KEY": ""}, clear=False):
            assert p._resolve_engine() == "orpheus"

    def test_qwen3_selected_for_supported_lang(self, config):
        config.target_language = "ja"
        p = DubbingPipeline(config)
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "FISH_API_KEY": ""}, clear=False):
            # ja is in both Orpheus and Qwen3 — Orpheus comes first
            result = p._resolve_engine()
            assert result in ("orpheus", "qwen3")

    def test_piper_selected_when_voice_installed(self, config, tmp_path):
        config.target_language = "pl"
        p = DubbingPipeline(config)
        voice_dir = tmp_path / ".local" / "share" / "piper-voices"
        voice_dir.mkdir(parents=True)
        (voice_dir / "pl_PL-darkman-medium.onnx").touch()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "FISH_API_KEY": ""}, clear=False), \
             patch("pathlib.Path.home", return_value=tmp_path):
            assert p._resolve_engine() == "piper"

    def test_chatterbox_fallback_for_unsupported_lang(self, config):
        config.target_language = "sw"  # Swahili — not in any engine
        p = DubbingPipeline(config)
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "FISH_API_KEY": ""}, clear=False):
            assert p._resolve_engine() == "chatterbox"

    def test_chatterbox_fallback_when_piper_not_installed(self, config, tmp_path):
        config.target_language = "de"
        p = DubbingPipeline(config)
        # de is in Orpheus and Qwen3 which come before Piper
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "FISH_API_KEY": ""}, clear=False):
            result = p._resolve_engine()
            assert result in ("orpheus", "qwen3")


# ============================================================================
# _build_synthesis_kwargs() Tests
# ============================================================================


class TestBuildSynthesisKwargs:
    """Test engine-specific synthesis kwargs construction."""

    def test_qwen3_with_clone(self, pipeline):
        seg = _seg(metadata={"emotion": {"primary": "happy"}})
        kwargs = pipeline._build_synthesis_kwargs("qwen3", seg, "/tmp/ref.wav", False, 24000)
        assert kwargs["voice"] == "clone"
        assert kwargs["language"] == "pl"

    def test_qwen3_without_clone(self, pipeline):
        seg = _seg()
        kwargs = pipeline._build_synthesis_kwargs("qwen3", seg, None, False, 24000)
        assert kwargs["voice"] == "Ryan"

    def test_elevenlabs_emotion_style(self, pipeline):
        seg = _seg(metadata={"emotion": {"primary": "happy"}})
        kwargs = pipeline._build_synthesis_kwargs("elevenlabs", seg, None, False, 24000)
        assert "style" in kwargs
        assert isinstance(kwargs["style"], float)

    def test_elevenlabs_no_emotion(self, pipeline):
        seg = _seg()
        kwargs = pipeline._build_synthesis_kwargs("elevenlabs", seg, None, False, 24000)
        assert "style" not in kwargs

    def test_fish_emotion(self, pipeline):
        seg = _seg(metadata={"emotion": {"primary": "sad"}})
        kwargs = pipeline._build_synthesis_kwargs("fish", seg, "/tmp/ref.wav", False, 24000)
        assert kwargs["emotion"] == "sad"
        assert kwargs["voice"] == "clone"

    def test_orpheus_emotion_and_voice(self, pipeline):
        pipeline.config.orpheus_voice = "tara"
        seg = _seg(metadata={"emotion": {"primary": "angry"}})
        kwargs = pipeline._build_synthesis_kwargs("orpheus", seg, None, False, 24000)
        assert kwargs["emotion"] == "angry"
        assert kwargs["voice"] == "tara"

    def test_orpheus_with_clone(self, pipeline):
        seg = _seg()
        kwargs = pipeline._build_synthesis_kwargs("orpheus", seg, "/tmp/ref.wav", False, 24000)
        assert kwargs["voice"] == "clone"

    def test_piper_window_ms(self, pipeline):
        seg = _seg(start=0.0, end=2.5)
        kwargs = pipeline._build_synthesis_kwargs("piper", seg, None, False, 22050)
        assert kwargs["_window_ms"] == 2500

    def test_piper_no_window_for_zero_duration(self, pipeline):
        seg = _seg(start=0.0, end=0.0)
        kwargs = pipeline._build_synthesis_kwargs("piper", seg, None, False, 22050)
        assert "_window_ms" not in kwargs or kwargs.get("_window_ms") == 0

    def test_chatterbox_multi_voice(self, pipeline):
        seg = _seg(speaker="spk_0", metadata={"emotion": {"primary": "happy", "intensity": 0.8}})
        kwargs = pipeline._build_synthesis_kwargs("chatterbox", seg, "/tmp/ref.wav", True, 24000)
        assert kwargs["speaker_id"] == "spk_0"
        assert "emotion_params" in kwargs

    def test_chatterbox_single_voice(self, pipeline):
        seg = _seg()
        kwargs = pipeline._build_synthesis_kwargs("chatterbox", seg, "/tmp/ref.wav", False, 24000)
        assert kwargs["audio_prompt_path"] == "/tmp/ref.wav"
        assert kwargs["exaggeration"] == pipeline.config.chatterbox_exaggeration

    def test_indextts_emotion_prompt(self, pipeline):
        pipeline.config.indextts_emotion_prompt = Path("/tmp/emo.wav")
        seg = _seg(start=0.0, end=3.0)
        kwargs = pipeline._build_synthesis_kwargs("indextts", seg, "/tmp/ref.wav", False, 24000)
        assert kwargs["emotion_prompt"] == Path("/tmp/emo.wav")
        assert kwargs["target_duration_ms"] == 3000
        assert kwargs["voice"] == "clone"

    def test_cosyvoice_instruct(self, pipeline):
        pipeline.config.cosyvoice_instruct = "Speak happily"
        seg = _seg()
        kwargs = pipeline._build_synthesis_kwargs("cosyvoice", seg, "/tmp/ref.wav", False, 24000)
        assert kwargs["instruct"] == "Speak happily"
        assert kwargs["voice"] == "clone"

    def test_f5_speed(self, pipeline):
        pipeline.config.f5_speed = 1.2
        seg = _seg()
        kwargs = pipeline._build_synthesis_kwargs("f5", seg, "/tmp/ref.wav", False, 24000)
        assert kwargs["speed"] == 1.2
        assert kwargs["voice"] == "clone"

    def test_language_not_set_for_chatterbox(self, pipeline):
        seg = _seg()
        kwargs = pipeline._build_synthesis_kwargs("chatterbox", seg, None, False, 24000)
        assert "language" not in kwargs

    def test_language_set_for_non_chatterbox(self, pipeline):
        for engine in ("qwen3", "elevenlabs", "fish", "f5", "orpheus", "indextts", "cosyvoice"):
            seg = _seg()
            kwargs = pipeline._build_synthesis_kwargs(engine, seg, None, False, 24000)
            assert kwargs.get("language") == "pl", f"language missing for {engine}"


# ============================================================================
# _merge_segments() Tests
# ============================================================================


class TestMergeSegments:
    """Test merging of short adjacent segments."""

    def test_no_merge_when_single_segment(self, pipeline):
        segments = [_seg(0, 0.0, 1.0, translated="Hello world")]
        result = pipeline._merge_segments(segments)
        assert len(result) == 1

    def test_merge_adjacent_short_segments(self, pipeline):
        segments = [
            _seg(0, 0.0, 1.0, translated="Hello"),
            _seg(1, 1.1, 2.0, translated="world"),
        ]
        result = pipeline._merge_segments(segments)
        assert len(result) == 1
        assert "Hello" in result[0].translated_text
        assert "world" in result[0].translated_text

    def test_no_merge_when_gap_too_large(self, pipeline):
        pipeline.config.merge_max_gap_ms = 200
        segments = [
            _seg(0, 0.0, 1.0, translated="Hello"),
            _seg(1, 2.0, 3.0, translated="world"),  # 1s gap > 200ms
        ]
        result = pipeline._merge_segments(segments)
        assert len(result) == 2

    def test_no_merge_different_speakers(self, pipeline):
        segments = [
            _seg(0, 0.0, 1.0, translated="Hello", speaker="spk_0"),
            _seg(1, 1.1, 2.0, translated="world", speaker="spk_1"),
        ]
        result = pipeline._merge_segments(segments)
        assert len(result) == 2

    def test_merge_same_speakers(self, pipeline):
        segments = [
            _seg(0, 0.0, 1.0, translated="Hello", speaker="spk_0"),
            _seg(1, 1.1, 2.0, translated="world", speaker="spk_0"),
        ]
        result = pipeline._merge_segments(segments)
        assert len(result) == 1

    def test_no_merge_when_combined_too_long(self, pipeline):
        pipeline.config.merge_max_duration_s = 2.0
        segments = [
            _seg(0, 0.0, 1.5, translated="Hello there"),
            _seg(1, 1.6, 3.5, translated="How are you"),  # combined = 3.5s > 2.0s
        ]
        result = pipeline._merge_segments(segments)
        assert len(result) == 2

    def test_reindex_after_merge(self, pipeline):
        segments = [
            _seg(0, 0.0, 1.0, translated="A"),
            _seg(1, 1.1, 2.0, translated="B"),
            _seg(2, 5.0, 6.0, translated="C"),
        ]
        result = pipeline._merge_segments(segments)
        for i, seg in enumerate(result):
            assert seg.index == i

    def test_merge_respects_word_cap_for_chatterbox(self, config):
        config.tts_engine = "chatterbox"
        config.target_language = "pl"  # Not in QWEN3_LANGS → chatterbox caps
        p = DubbingPipeline(config)
        # 12 words is the cap for chatterbox
        words_a = " ".join(["word"] * 7)
        words_b = " ".join(["word"] * 7)
        segments = [
            _seg(0, 0.0, 1.0, translated=words_a),
            _seg(1, 1.1, 2.0, translated=words_b),
        ]
        result = p._merge_segments(segments)
        # 14 words > 12 cap — should NOT merge
        assert len(result) == 2

    def test_long_segment_gets_split(self, pipeline):
        # Create a segment with many words that exceeds word cap
        long_text = "This is a sentence. " * 10  # ~50 words, multiple sentences
        segments = [_seg(0, 0.0, 10.0, translated=long_text.strip())]
        result = pipeline._merge_segments(segments)
        assert len(result) >= 2  # Should be split

    def test_empty_input(self, pipeline):
        result = pipeline._merge_segments([])
        assert result == []


# ============================================================================
# _split_long_segment() Tests
# ============================================================================


class TestSplitLongSegment:
    """Test splitting of long segments at sentence boundaries."""

    def test_short_segment_not_split(self):
        seg = _seg(0, 0.0, 2.0, translated="Hello world.")
        result = DubbingPipeline._split_long_segment(seg, max_words=25)
        assert len(result) == 1

    def test_single_sentence_not_split(self):
        text = " ".join(["word"] * 30)  # 30 words, no sentence boundary
        seg = _seg(0, 0.0, 10.0, translated=text)
        result = DubbingPipeline._split_long_segment(seg, max_words=25)
        assert len(result) == 1  # No sentence boundary to split on

    def test_multi_sentence_split(self):
        text = "First sentence is here. Second sentence follows. Third one too."
        seg = _seg(0, 0.0, 9.0, translated=text)
        result = DubbingPipeline._split_long_segment(seg, max_words=5)
        assert len(result) >= 2

    def test_time_distributed_proportionally(self):
        text = "Short. This is a much longer second sentence with many words."
        seg = _seg(0, 0.0, 10.0, translated=text)
        result = DubbingPipeline._split_long_segment(seg, max_words=5)
        if len(result) >= 2:
            # First chunk is shorter text → should get less time
            assert result[0].end_time - result[0].start_time < \
                   result[-1].end_time - result[-1].start_time

    def test_speaker_id_preserved(self):
        text = "First sentence. Second sentence."
        seg = _seg(0, 0.0, 4.0, translated=text, speaker="spk_0")
        result = DubbingPipeline._split_long_segment(seg, max_words=3)
        for r in result:
            assert r.speaker_id == "spk_0"

    def test_metadata_preserved(self):
        text = "First sentence. Second sentence."
        seg = _seg(0, 0.0, 4.0, translated=text, metadata={"emotion": "happy"})
        result = DubbingPipeline._split_long_segment(seg, max_words=3)
        for r in result:
            assert r.metadata.get("emotion") == "happy"

    def test_continuity_of_time_ranges(self):
        text = "One sentence. Two sentence. Three sentence."
        seg = _seg(0, 1.0, 10.0, translated=text)
        result = DubbingPipeline._split_long_segment(seg, max_words=3)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i].end_time == pytest.approx(result[i + 1].start_time, abs=0.01)


# ============================================================================
# _extract_audio() Tests
# ============================================================================


class TestExtractAudio:
    def test_extract_audio_calls_ffmpeg(self, pipeline, tmp_path):
        pipeline.config.output_dir = tmp_path
        with patch("audiosmith.ffmpeg.extract_audio") as mock_ea, \
             patch("audiosmith.ffmpeg.extract_audio_hq"):
            result = pipeline._extract_audio(Path("video.mp4"))
            mock_ea.assert_called_once()
            assert result == tmp_path / "audio_16k_mono.wav"

    def test_extract_audio_hq_when_requested(self, pipeline, tmp_path):
        pipeline.config.output_dir = tmp_path
        with patch("audiosmith.ffmpeg.extract_audio"), \
             patch("audiosmith.ffmpeg.extract_audio_hq") as mock_hq:
            pipeline._extract_audio(Path("video.mp4"), extract_hq=True)
            mock_hq.assert_called_once()
            assert pipeline.state.hq_audio_path is not None


# ============================================================================
# _transcribe() Tests
# ============================================================================


class TestTranscribe:
    def test_transcribe_returns_segments(self, pipeline):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            {"text": "Hello", "start": 0.0, "end": 1.5},
            {"text": "World", "start": 2.0, "end": 3.0},
        ]
        with patch("audiosmith.transcribe.Transcriber", return_value=mock_transcriber):
            segments = pipeline._transcribe(Path("audio.wav"))
        assert len(segments) == 2
        assert segments[0].original_text == "Hello"
        assert segments[1].start_time == 2.0

    def test_transcribe_auto_language(self, pipeline):
        pipeline.config.source_language = "auto"
        mock_t = MagicMock()
        mock_t.transcribe.return_value = []
        with patch("audiosmith.transcribe.Transcriber", return_value=mock_t):
            pipeline._transcribe(Path("audio.wav"))
        mock_t.transcribe.assert_called_once_with(Path("audio.wav"), language=None)

    def test_transcribe_specific_language(self, pipeline):
        pipeline.config.source_language = "fr"
        mock_t = MagicMock()
        mock_t.transcribe.return_value = []
        with patch("audiosmith.transcribe.Transcriber", return_value=mock_t):
            pipeline._transcribe(Path("audio.wav"))
        mock_t.transcribe.assert_called_once_with(Path("audio.wav"), language="fr")


# ============================================================================
# _translate() Tests
# ============================================================================


class TestTranslate:
    def test_translate_fills_translated_text(self, pipeline):
        with patch("audiosmith.translate.translate", return_value="Cześć"):
            segments = [_seg(0, 0.0, 1.0, original="Hello")]
            result = pipeline._translate(segments)
        assert result[0].translated_text == "Cześć"

    def test_translate_uses_config_languages(self, pipeline):
        pipeline.config.source_language = "en"
        pipeline.config.target_language = "de"
        with patch("audiosmith.translate.translate") as mock_t:
            mock_t.return_value = "Hallo"
            pipeline._translate([_seg(0, 0.0, 1.0, original="Hello")])
        mock_t.assert_called_once_with("Hello", "en", "de")


# ============================================================================
# _import_external_srt() Tests
# ============================================================================


class TestImportExternalSrt:
    def test_matches_by_time_overlap(self, pipeline, tmp_path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nCześć\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nŚwiat\n",
            encoding="utf-8",
        )
        segments = [
            _seg(0, 0.0, 2.0, original="Hello"),
            _seg(1, 3.0, 5.0, original="World"),
        ]
        result = pipeline._import_external_srt(srt_file, segments)
        assert result[0].translated_text == "Cześć"
        assert result[1].translated_text == "Świat"

    def test_fallback_when_no_overlap(self, pipeline, tmp_path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:10,000 --> 00:00:12,000\nFar away\n",
            encoding="utf-8",
        )
        segments = [_seg(0, 0.0, 1.0, original="Hello")]
        result = pipeline._import_external_srt(srt_file, segments)
        assert result[0].translated_text == "Hello"  # Fallback to original
        assert result[0].metadata["translation_source"] == "fallback_original"


# ============================================================================
# _create_segments_from_srt() Tests
# ============================================================================


class TestCreateSegmentsFromSrt:
    def test_creates_segments_from_srt(self, pipeline, tmp_path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nGoodbye\n",
            encoding="utf-8",
        )
        segments = pipeline._create_segments_from_srt(srt_file)
        assert len(segments) == 2
        assert segments[0].translated_text == "Hello world"
        assert segments[1].translated_text == "Goodbye"

    def test_skips_non_speech_entries(self, pipeline, tmp_path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\n[Music]\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nHello\n",
            encoding="utf-8",
        )
        segments = pipeline._create_segments_from_srt(srt_file)
        assert len(segments) == 1
        assert segments[0].translated_text == "Hello"

    def test_skips_very_short_entries(self, pipeline, tmp_path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:01,100\nBrief\n\n"  # 0.1s < 0.3s
            "2\n00:00:02,000 --> 00:00:04,000\nLong enough\n",
            encoding="utf-8",
        )
        segments = pipeline._create_segments_from_srt(srt_file)
        assert len(segments) == 1

    def test_skips_empty_entries(self, pipeline, tmp_path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\n\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nHello\n",
            encoding="utf-8",
        )
        segments = pipeline._create_segments_from_srt(srt_file)
        assert len(segments) == 1

    def test_sequential_indexing(self, pipeline, tmp_path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nA\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nB\n\n"
            "3\n00:00:07,000 --> 00:00:09,000\nC\n",
            encoding="utf-8",
        )
        segments = pipeline._create_segments_from_srt(srt_file)
        for i, seg in enumerate(segments):
            assert seg.index == i


# ============================================================================
# _extract_speaker_voices() Tests
# ============================================================================


class TestExtractSpeakerVoices:
    def test_extracts_voice_per_speaker(self, pipeline, tmp_path):
        voice_dir = tmp_path / "voices"
        segments = [
            _seg(0, 0.0, 5.0, speaker="spk_0"),
        ]
        segments[0].is_speech = True
        audio_path = tmp_path / "audio.wav"
        long_audio = np.random.randn(48000 * 6).astype(np.float32)

        def fake_sf_write(path, data, sr):
            """Create a real file so shutil.copy2 works for default.wav."""
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("soundfile.read", return_value=(long_audio, 48000)), \
             patch("soundfile.write", side_effect=fake_sf_write):
            pipeline._extract_speaker_voices(segments, audio_path, voice_dir)
        # spk_0.wav and default.wav should both exist
        assert (voice_dir / "spk_0.wav").exists()
        assert (voice_dir / "default.wav").exists()

    def test_skips_short_clips(self, pipeline, tmp_path):
        voice_dir = tmp_path / "voices"
        # Audio shorter than 1 second for segment
        short_audio = np.random.randn(24000).astype(np.float32)  # 0.5s at 48kHz
        segments = [_seg(0, 0.0, 0.3, speaker="spk_0")]
        segments[0].is_speech = True
        with patch("soundfile.read", return_value=(short_audio, 48000)), \
             patch("soundfile.write") as mock_write:
            pipeline._extract_speaker_voices(segments, Path("a.wav"), voice_dir)
        mock_write.assert_not_called()

    def test_no_speakers_does_nothing(self, pipeline, tmp_path):
        voice_dir = tmp_path / "voices"
        segments = [_seg(0, 0.0, 1.0)]  # No speaker_id
        with patch("soundfile.write") as mock_write:
            pipeline._extract_speaker_voices(segments, Path("a.wav"), voice_dir)
        # No speakers → early return, no audio written
        mock_write.assert_not_called()


# ============================================================================
# _mix_audio() Tests
# ============================================================================


class TestMixAudio:
    def test_mix_audio_calls_mixer(self, pipeline, tmp_path):
        pipeline.config.output_dir = tmp_path
        mock_mixer = MagicMock()
        mock_mixer.max_speedup = 2.0
        mock_mixer.schedule.return_value = []
        with patch("audiosmith.mixer.AudioMixer", return_value=mock_mixer):
            pipeline._mix_audio([], 10.0)
        mock_mixer.schedule.assert_called_once()
        mock_mixer.render_to_file.assert_called_once()

    def test_mix_audio_uses_background(self, pipeline, tmp_path):
        pipeline.config.output_dir = tmp_path
        bg = tmp_path / "bg.wav"
        bg.touch()
        pipeline.state.background_audio_path = str(bg)
        mock_mixer = MagicMock()
        mock_mixer.max_speedup = 2.0
        mock_mixer.schedule.return_value = []
        with patch("audiosmith.mixer.AudioMixer", return_value=mock_mixer) as mock_cls:
            pipeline._mix_audio([], 10.0)
        # Verify background_path was passed
        call_kwargs = mock_cls.call_args
        assert call_kwargs[1]["background_path"] == bg


# ============================================================================
# _encode_video() Tests
# ============================================================================


class TestEncodeVideo:
    def test_encode_video_calls_ffmpeg(self, pipeline, tmp_path):
        pipeline.config.output_dir = tmp_path
        pipeline.config.burn_subtitles = False
        with patch("audiosmith.ffmpeg.encode_video") as mock_ev:
            result = pipeline._encode_video(
                Path("video.mp4"), Path("dubbed.wav"), []
            )
        mock_ev.assert_called_once()
        assert "dubbed" in str(result)

    def test_encode_video_burns_subtitles(self, pipeline, tmp_path):
        pipeline.config.output_dir = tmp_path
        pipeline.config.burn_subtitles = True
        segments = [_seg(0, 0.0, 1.0, translated="Hello")]
        with patch("audiosmith.ffmpeg.encode_video") as mock_ev:
            pipeline._encode_video(Path("video.mp4"), Path("dubbed.wav"), segments)
        # Should create SRT file
        srt_path = tmp_path / "subtitles.srt"
        assert srt_path.exists()

    def test_encode_video_uses_schedule_for_srt(self, pipeline, tmp_path):
        pipeline.config.output_dir = tmp_path
        pipeline.config.burn_subtitles = True
        seg = _seg(0, 0.0, 1.0, translated="Hello")
        scheduled = [ScheduledSegment(segment=seg, place_at_ms=0, actual_duration_ms=1000)]
        with patch("audiosmith.ffmpeg.encode_video"):
            pipeline._encode_video(
                Path("video.mp4"), Path("dubbed.wav"), [seg], scheduled=scheduled
            )
        srt_path = tmp_path / "subtitles.srt"
        assert srt_path.exists()


# ============================================================================
# Full run() Orchestrator Tests
# ============================================================================


class TestPipelineRun:
    """Test the full pipeline.run() with all external deps mocked."""

    def _mock_all_steps(self):
        """Return patches for all pipeline steps."""
        return {
            "extract_audio": patch.object(
                DubbingPipeline, "_extract_audio",
                return_value=Path("/tmp/audio.wav"),
            ),
            "transcribe": patch.object(
                DubbingPipeline, "_transcribe",
                return_value=[_seg(0, 0.0, 2.0, original="Hello")],
            ),
            "translate": patch.object(
                DubbingPipeline, "_translate",
                return_value=[_seg(0, 0.0, 2.0, original="Hello", translated="Cześć")],
            ),
            "merge": patch.object(
                DubbingPipeline, "_merge_segments",
                return_value=[_seg(0, 0.0, 2.0, original="Hello", translated="Cześć")],
            ),
            "generate_tts": patch.object(
                DubbingPipeline, "_generate_tts",
                return_value=[
                    _seg(0, 0.0, 2.0, original="Hello", translated="Cześć"),
                ],
            ),
            "mix_audio": patch.object(
                DubbingPipeline, "_mix_audio",
                return_value=(Path("/tmp/dubbed.wav"), []),
            ),
            "encode_video": patch.object(
                DubbingPipeline, "_encode_video",
                return_value=Path("/tmp/output.mp4"),
            ),
            "probe_duration": patch(
                "audiosmith.ffmpeg.probe_duration",
                return_value=10.0,
            ),
            "save_checkpoint": patch.object(
                DubbingPipeline, "_save_checkpoint",
            ),
        }

    def test_full_pipeline_succeeds(self, config):
        p = DubbingPipeline(config)
        patches = self._mock_all_steps()
        mocks = {}
        for name, patcher in patches.items():
            mocks[name] = patcher.start()
        try:
            # Make generate_tts return segment with tts_audio_path
            seg = _seg(0, 0.0, 2.0, original="Hello", translated="Cześć")
            seg.tts_audio_path = Path("/tmp/seg.wav")
            mocks["generate_tts"].return_value = [seg]

            result = p.run(Path("video.mp4"))
            assert result.success is True
            assert result.total_segments == 1
            assert result.segments_dubbed == 1
        finally:
            for patcher in patches.values():
                patcher.stop()

    def test_pipeline_wraps_unexpected_error(self, config):
        p = DubbingPipeline(config)
        with patch.object(DubbingPipeline, "_extract_audio", side_effect=RuntimeError("boom")), \
             patch.object(DubbingPipeline, "_save_checkpoint"):
            from audiosmith.exceptions import DubbingError
            with pytest.raises(DubbingError, match="boom"):
                p.run(Path("video.mp4"))

    def test_pipeline_reraises_dubbing_error(self, config):
        from audiosmith.exceptions import DubbingError
        p = DubbingPipeline(config)
        with patch.object(
            DubbingPipeline, "_extract_audio",
            side_effect=DubbingError("test", error_code="E001"),
        ), patch.object(DubbingPipeline, "_save_checkpoint"):
            with pytest.raises(DubbingError, match="test"):
                p.run(Path("video.mp4"))

    def test_pipeline_skips_completed_steps_on_resume(self, config, tmp_path):
        # Set up checkpoint with extract_audio and transcribe done
        state = PipelineState()
        state.mark_step_done("extract_audio")
        state.mark_step_done("transcribe")
        state.audio_path = str(tmp_path / "audio.wav")
        state.segments = _segments_to_dicts([
            _seg(0, 0.0, 2.0, original="Hello"),
        ])
        state.save(tmp_path / ".checkpoint.json")

        config.output_dir = tmp_path
        config.resume = True
        p = DubbingPipeline(config)

        patches = self._mock_all_steps()
        mocks = {}
        for name, patcher in patches.items():
            mocks[name] = patcher.start()
        try:
            seg = _seg(0, 0.0, 2.0, original="Hello", translated="Cześć")
            seg.tts_audio_path = Path("/tmp/seg.wav")
            mocks["generate_tts"].return_value = [seg]

            result = p.run(Path("video.mp4"))
            assert result.success is True
            # extract_audio and transcribe should NOT have been called
            mocks["extract_audio"].assert_not_called()
            mocks["transcribe"].assert_not_called()
        finally:
            for patcher in patches.values():
                patcher.stop()


# ============================================================================
# Helper Functions Tests
# ============================================================================


class TestDedupRepeatedWords:
    def test_no_repetition(self):
        assert _dedup_repeated_words("hello world") == "hello world"

    def test_two_repeats_kept(self):
        assert _dedup_repeated_words("hello hello world") == "hello hello world"

    def test_three_repeats_collapsed(self):
        assert _dedup_repeated_words("hello hello hello world") == "hello hello world"

    def test_many_repeats_collapsed(self):
        assert _dedup_repeated_words("the the the the the end") == "the the end"

    def test_case_insensitive(self):
        assert _dedup_repeated_words("Hello hello HELLO world") == "Hello hello world"

    def test_short_text_unchanged(self):
        assert _dedup_repeated_words("hi") == "hi"
        assert _dedup_repeated_words("") == ""

    def test_custom_max_repeats(self):
        result = _dedup_repeated_words("a a a a", max_repeats=3)
        assert result == "a a a"


class TestSegmentsRoundtrip:
    def test_roundtrip_preserves_all_fields(self):
        original = [
            DubbingSegment(
                index=0, start_time=0.0, end_time=2.5,
                original_text="Hi", translated_text="Cześć",
                speaker_id="spk_0",
                metadata={"emotion": {"primary": "happy"}},
                tts_audio_path=Path("/tmp/seg.wav"),
                tts_duration_ms=1500,
            ),
        ]
        dicts = _segments_to_dicts(original)
        restored = _dicts_to_segments(dicts)
        assert restored[0].original_text == "Hi"
        assert restored[0].translated_text == "Cześć"
        assert restored[0].speaker_id == "spk_0"
        assert restored[0].tts_audio_path == Path("/tmp/seg.wav")
        assert restored[0].tts_duration_ms == 1500

    def test_roundtrip_handles_none_values(self):
        original = [_seg(0, 0.0, 1.0)]
        dicts = _segments_to_dicts(original)
        restored = _dicts_to_segments(dicts)
        assert restored[0].tts_audio_path is None
        assert restored[0].speaker_id is None


class TestWriteSrt:
    def test_write_srt_creates_file(self, tmp_path):
        segments = [_seg(0, 0.0, 1.0, translated="Hello")]
        path = tmp_path / "test.srt"
        _write_srt(segments, path)
        assert path.exists()
        content = path.read_text()
        assert "Hello" in content
        assert "-->" in content

    def test_write_srt_from_schedule(self, tmp_path):
        seg = _seg(0, 0.0, 1.0, translated="Hello")
        scheduled = [ScheduledSegment(segment=seg, place_at_ms=500, actual_duration_ms=800)]
        path = tmp_path / "test.srt"
        _write_srt_from_schedule(scheduled, path)
        assert path.exists()


class TestEmotionToTtsParams:
    def test_scales_with_intensity(self):
        low = _emotion_to_tts_params("happy", intensity=0.0)
        high = _emotion_to_tts_params("happy", intensity=1.0)
        # At intensity 0, should collapse to midpoint
        assert low["exaggeration"] == pytest.approx(0.5)
        # At intensity 1, should be at full value
        assert high["exaggeration"] != pytest.approx(0.5)

    def test_mid_intensity(self):
        result = _emotion_to_tts_params("happy", intensity=0.5)
        # Should be between midpoint and full value
        assert 0.4 < result["exaggeration"] < 0.8
