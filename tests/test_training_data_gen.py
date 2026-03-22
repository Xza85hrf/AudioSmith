"""Tests for audiosmith.polish_corpus and audiosmith.training_data_gen modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aiml_training", reason="aiml_training not available in CI")

from audiosmith.exceptions import TrainingError
from aiml_training.training.polish_corpus import (
    _ABBREVIATIONS, ALL_POLISH_CHARS, POLISH_DIACRITICS, PolishCorpusManager,
)
from aiml_training.training.training_data_gen import (
    Checkpoint, TrainingDataConfig, TrainingDataGenerator,
)
from audiosmith.emotion_config import EMOTION_TTS_MAP as _EMOTION_TTS_MAP

SR = 24000


def _make_audio(duration: float = 2.0, freq: float = 300.0) -> np.ndarray:
    """Create a speech-like synthetic audio signal for testing.

    Uses amplitude modulation to create varying loudness (like speech),
    which ensures the SNR estimator sees a difference between loud and quiet frames.
    """
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    # Carrier signal
    carrier = np.sin(2 * np.pi * freq * t)
    # Amplitude envelope: bursts of sound with gaps (like syllables)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)  # ~3 Hz modulation
    envelope = np.clip(envelope, 0.05, 1.0)
    return (0.3 * carrier * envelope).astype(np.float32)


# ── Polish Corpus Tests ──────────────────────────────────────────────


class TestPolishDiacritics:
    def test_diacritics_set_has_9_chars(self):
        assert len(POLISH_DIACRITICS) == 9

    def test_all_polish_chars_includes_upper(self):
        assert "Ą" in ALL_POLISH_CHARS
        assert "ą" in ALL_POLISH_CHARS

    def test_standard_polish_diacritics(self):
        for ch in "ąęćśźżłńó":
            assert ch in POLISH_DIACRITICS


class TestPolishCorpusValidation:
    def setup_method(self):
        self.manager = PolishCorpusManager()

    def test_validates_polish_text(self):
        assert self.manager.validate_polish("Zażółć gęślą jaźń.")

    def test_rejects_pure_ascii(self):
        assert not self.manager.validate_polish("This is English text.")

    def test_rejects_empty(self):
        assert not self.manager.validate_polish("")

    def test_validates_single_diacritic(self):
        assert self.manager.validate_polish("Krolowa ląd.")

    def test_validates_uppercase_diacritic(self):
        assert self.manager.validate_polish("Łódź jest miastem.")


class TestPolishCorpusSentenceExtraction:
    def setup_method(self):
        self.manager = PolishCorpusManager()

    def test_extracts_sentences_from_file(self, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text(
            "Zażółć gęślą jaźń. Pchnąć w tę łódź jeża.\n"
            "Krótkąś wieść. Too short.\n",
            encoding="utf-8",
        )
        sentences = self.manager.extract_sentences(text_file, min_len=10, max_len=200)
        assert len(sentences) >= 2
        assert all(self.manager.validate_polish(s) for s in sentences)

    def test_respects_min_length(self, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("Ąć. Zażółć gęślą jaźń jest piękne.\n", encoding="utf-8")
        sentences = self.manager.extract_sentences(text_file, min_len=15)
        # "Ąć." is too short (3 chars), only the longer sentence should pass
        for s in sentences:
            assert len(s) >= 15

    def test_respects_max_length(self, tmp_path):
        text_file = tmp_path / "test.txt"
        long_sent = "Zażółć gęślą jaźń " * 20 + "kończy się tutaj."
        text_file.write_text(long_sent + "\n", encoding="utf-8")
        sentences = self.manager.extract_sentences(text_file, max_len=50)
        for s in sentences:
            assert len(s) <= 50

    def test_deduplicates(self, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text(
            "Zażółć gęślą jaźń pięknie brzmi.\n" * 5,
            encoding="utf-8",
        )
        sentences = self.manager.extract_sentences(text_file)
        assert len(sentences) <= 1

    def test_max_sentences_limit(self, tmp_path):
        text_file = tmp_path / "test.txt"
        lines = [f"Zażółć gęślą jaźń numer {i} jest długa.\n" for i in range(100)]
        text_file.write_text("".join(lines), encoding="utf-8")
        sentences = self.manager.extract_sentences(text_file, max_sentences=10)
        assert len(sentences) <= 10

    def test_raises_on_missing_file(self):
        with pytest.raises(TrainingError, match="TRAIN_CORPUS_MISSING"):
            self.manager.extract_sentences(Path("/nonexistent/file.txt"))

    def test_rejects_digits(self, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text(
            "W roku 2023 powstała nowa łódź.\n"
            "Pchnąć w tę łódź jeża bez cyfr.\n",
            encoding="utf-8",
        )
        sentences = self.manager.extract_sentences(text_file, min_len=10)
        for s in sentences:
            assert not any(c.isdigit() for c in s)


class TestPolishCorpusDiversify:
    def setup_method(self):
        self.manager = PolishCorpusManager()

    def test_returns_target_count(self):
        sentences = [
            "Czy to jest pytanie po polsku?",
            "Zażółć gęślą jaźń pięknie brzmi.",
            "Łódź to piękne miasto nad rzeką.",
            "Pchnąć w tę łódź jeża ze złością!",
        ] * 100
        result = self.manager.diversify(sentences, target_count=50)
        assert len(result) == 50

    def test_preserves_small_corpus(self):
        sentences = ["Zażółć gęślą jaźń.", "Pchnąć w tę łódź jeża."]
        result = self.manager.diversify(sentences, target_count=100)
        assert len(result) == 2  # fewer than target, returns all


class TestPolishCorpusSaveLoad:
    def setup_method(self):
        self.manager = PolishCorpusManager()

    def test_round_trip(self, tmp_path):
        sentences = ["Zażółć gęślą jaźń.", "Pchnąć w tę łódź jeża."]
        corpus_path = tmp_path / "corpus.txt"
        self.manager.save_corpus(sentences, corpus_path)
        loaded = self.manager.load_corpus(corpus_path)
        assert loaded == sentences

    def test_load_missing_raises(self):
        with pytest.raises(TrainingError, match="TRAIN_CORPUS_MISSING"):
            self.manager.load_corpus(Path("/nonexistent/corpus.txt"))

    def test_utf8_encoding(self, tmp_path):
        sentences = ["Łódź — gęślą ąęćśźżłńó."]
        corpus_path = tmp_path / "corpus.txt"
        self.manager.save_corpus(sentences, corpus_path)
        loaded = self.manager.load_corpus(corpus_path)
        assert loaded[0] == sentences[0]


class TestAbbreviations:
    def test_abbreviations_dict_non_empty(self):
        assert len(_ABBREVIATIONS) > 10

    def test_common_abbreviations(self):
        assert "ul." in _ABBREVIATIONS
        assert "nr" in _ABBREVIATIONS
        assert "np." in _ABBREVIATIONS


# ── Training Data Config Tests ───────────────────────────────────────


class TestTrainingDataConfig:
    def test_defaults(self):
        config = TrainingDataConfig()
        assert config.sample_rate == 24000
        assert config.device == "cuda"
        assert config.target_sample_count == 8000
        assert config.enable_chatterbox is True
        assert config.enable_elevenlabs is False

    def test_custom_values(self):
        config = TrainingDataConfig(
            output_dir=Path("/tmp/test"),
            target_sample_count=100,
            enable_elevenlabs=True,
        )
        assert config.output_dir == Path("/tmp/test")
        assert config.target_sample_count == 100
        assert config.enable_elevenlabs is True

    def test_quality_thresholds(self):
        config = TrainingDataConfig()
        assert config.min_duration_s == 1.0
        assert config.max_duration_s == 20.0
        assert config.min_snr_db == 15.0
        assert config.max_silence_pct == 50.0

    def test_emotion_weights_match_emotions(self):
        config = TrainingDataConfig()
        assert len(config.emotions) == len(config.emotion_weights)


# ── Checkpoint Tests ─────────────────────────────────────────────────


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        ckpt = Checkpoint(stage=3, source="chatterbox", sample_idx=150, total=8000)
        ckpt.failed["42"] = "CUDA OOM"
        ckpt_path = tmp_path / "checkpoint.json"
        ckpt.save(ckpt_path)

        loaded = Checkpoint.load(ckpt_path)
        assert loaded.stage == 3
        assert loaded.source == "chatterbox"
        assert loaded.sample_idx == 150
        assert loaded.failed["42"] == "CUDA OOM"

    def test_default_values(self):
        ckpt = Checkpoint()
        assert ckpt.stage == 0
        assert ckpt.completed_stages == []
        assert ckpt.failed == {}


# ── Quality Filter Tests ─────────────────────────────────────────────


class TestQualityFilter:
    def setup_method(self):
        self.config = TrainingDataConfig(output_dir=Path(tempfile.mkdtemp()))
        self.gen = TrainingDataGenerator(self.config)

    def test_rejects_too_short(self):
        audio = _make_audio(duration=0.3)  # < 1.0s
        assert self.gen._check_quality(audio, SR) == "too_short"

    def test_rejects_too_long(self):
        audio = _make_audio(duration=25.0)  # > 20.0s
        assert self.gen._check_quality(audio, SR) == "too_long"

    def test_rejects_near_silent(self):
        audio = np.zeros(SR * 2, dtype=np.float32)  # silence
        assert self.gen._check_quality(audio, SR) == "near_silent"

    def test_rejects_clipping(self):
        audio = _make_audio(duration=2.0)
        audio = audio * 5.0  # clip above 0.95
        assert self.gen._check_quality(audio, SR) == "clipping"

    def test_passes_good_audio(self):
        audio = _make_audio(duration=3.0)
        result = self.gen._check_quality(audio, SR)
        assert result is None  # passes all checks

    def test_rejects_excess_silence(self):
        # Audio with 50% silence
        signal = _make_audio(duration=1.0)
        silence = np.zeros(SR * 3, dtype=np.float32)
        audio = np.concatenate([signal, silence])
        result = self.gen._check_quality(audio, SR)
        assert result == "excess_silence"


# ── Emotion Sampling Tests ───────────────────────────────────────────


class TestEmotionSampling:
    def test_emotion_map_has_required_emotions(self):
        for e in ["neutral", "happy", "sad", "excited"]:
            assert e in _EMOTION_TTS_MAP

    def test_emotion_params_structure(self):
        for emotion, params in _EMOTION_TTS_MAP.items():
            assert "exaggeration" in params
            assert "cfg_weight" in params
            assert 0.0 <= params["exaggeration"] <= 1.0
            assert 0.0 <= params["cfg_weight"] <= 1.0


# ── Resample Tests ───────────────────────────────────────────────────


class TestResample:
    def test_same_rate_noop(self):
        audio = _make_audio(duration=1.0)
        result = TrainingDataGenerator._resample(audio, 24000, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_downsample(self):
        audio = _make_audio(duration=1.0)
        result = TrainingDataGenerator._resample(audio, 48000, 24000)
        assert len(result) == len(audio) // 2

    def test_upsample(self):
        audio = _make_audio(duration=1.0)
        result = TrainingDataGenerator._resample(audio, 24000, 48000)
        assert len(result) == len(audio) * 2


# ── Dry Run Tests ────────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_returns_info(self, tmp_path):
        config = TrainingDataConfig(output_dir=tmp_path / "out")
        gen = TrainingDataGenerator(config)
        info = gen.dry_run()
        assert "sources" in info
        assert "target_samples" in info
        assert info["target_samples"] == 8000

    def test_dry_run_lists_enabled_sources(self, tmp_path):
        config = TrainingDataConfig(
            output_dir=tmp_path / "out",
            enable_chatterbox=True,
            enable_elevenlabs=True,
            enable_fish=False,
        )
        gen = TrainingDataGenerator(config)
        info = gen.dry_run()
        names = [s["name"] for s in info["sources"]]
        assert "chatterbox" in names
        assert "elevenlabs" in names
        assert "fish" not in names

    def test_dry_run_with_audiobook(self, tmp_path):
        audiobook_dir = tmp_path / "audiobook"
        audiobook_dir.mkdir()
        (audiobook_dir / "chapter1.mp3").touch()
        (audiobook_dir / "chapter2.mp3").touch()

        config = TrainingDataConfig(
            output_dir=tmp_path / "out",
            enable_audiobook=True,
            audiobook_dir=audiobook_dir,
        )
        gen = TrainingDataGenerator(config)
        info = gen.dry_run()
        ab_source = [s for s in info["sources"] if s["name"] == "audiobook"][0]
        assert ab_source["files"] == 2


# ── Sentence Distribution Tests ──────────────────────────────────────


class TestSentenceDistribution:
    def test_distributes_to_enabled_engines(self, tmp_path):
        config = TrainingDataConfig(
            output_dir=tmp_path / "out",
            enable_chatterbox=True,
            enable_elevenlabs=True,
            enable_fish=False,
            target_sample_count=100,
        )
        gen = TrainingDataGenerator(config)
        sentences = [f"Zażółć gęślą jaźń numer {i}." for i in range(100)]
        dist = gen._distribute_sentences(sentences)
        assert "chatterbox" in dist
        assert "elevenlabs" in dist
        assert "fish" not in dist
        assert len(dist["chatterbox"]) + len(dist["elevenlabs"]) == 100

    def test_chatterbox_only(self, tmp_path):
        config = TrainingDataConfig(
            output_dir=tmp_path / "out",
            enable_chatterbox=True,
            enable_elevenlabs=False,
            enable_fish=False,
            target_sample_count=50,
        )
        gen = TrainingDataGenerator(config)
        sentences = [f"Zażółć gęślą jaźń numer {i}." for i in range(50)]
        dist = gen._distribute_sentences(sentences)
        assert len(dist["chatterbox"]) == 50

    def test_empty_when_no_engines(self, tmp_path):
        config = TrainingDataConfig(
            output_dir=tmp_path / "out",
            enable_chatterbox=False,
            enable_elevenlabs=False,
            enable_fish=False,
        )
        gen = TrainingDataGenerator(config)
        assert gen._distribute_sentences(["test"]) == {}


# ── Manifest Tests ───────────────────────────────────────────────────


class TestManifest:
    def test_append_and_load(self, tmp_path):
        config = TrainingDataConfig(output_dir=tmp_path / "out")
        gen = TrainingDataGenerator(config)

        entries = [
            {"id": "cb_000001", "text": "Test zdanie.", "source": "chatterbox"},
            {"id": "cb_000002", "text": "Drugie zdanie.", "source": "chatterbox"},
        ]
        gen._append_manifest(entries)
        loaded = gen._load_manifest()
        assert len(loaded) == 2
        assert loaded[0]["id"] == "cb_000001"

    def test_append_empty_noop(self, tmp_path):
        config = TrainingDataConfig(output_dir=tmp_path / "out")
        gen = TrainingDataGenerator(config)
        gen._append_manifest([])
        assert not gen._manifest_path.exists()

    def test_load_missing_returns_empty(self, tmp_path):
        config = TrainingDataConfig(output_dir=tmp_path / "out")
        gen = TrainingDataGenerator(config)
        assert gen._load_manifest() == []


# ── Stage 1 Integration Test ────────────────────────────────────────


class TestStage1Corpus:
    def test_uses_existing_corpus(self, tmp_path):
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Zażółć gęślą jaźń.\nPchnąć w tę łódź jeża.\n")

        config = TrainingDataConfig(
            output_dir=tmp_path / "out",
            corpus_path=corpus_file,
        )
        gen = TrainingDataGenerator(config)
        result = gen._stage1_corpus()
        assert result["source"] == "existing"
        assert result["sentences"] == 2


# ── Stage 5 Quality Filter Integration ──────────────────────────────


class TestStage5Filter:
    def test_filters_and_reports(self, tmp_path):
        import soundfile as sf

        out_dir = tmp_path / "out"
        config = TrainingDataConfig(output_dir=out_dir)
        gen = TrainingDataGenerator(config)

        # Create a good audio file
        good_audio = _make_audio(duration=3.0)
        sf.write(str(out_dir / "raw" / "test_001.wav"), good_audio, SR)

        # Create a bad (silent) audio file
        bad_audio = np.zeros(SR * 2, dtype=np.float32)
        sf.write(str(out_dir / "raw" / "test_002.wav"), bad_audio, SR)

        # Write manifest
        gen._append_manifest([
            {"id": "test_001", "text": "Dobre zdanie.", "source": "chatterbox"},
            {"id": "test_002", "text": "Złe zdanie.", "source": "chatterbox"},
        ])

        result = gen._stage5_filter()
        assert result["passed"] == 1
        assert result["rejected"] == 1
        assert "near_silent" in result["reject_reasons"]

        # Verify filtered directory has the good file
        assert (out_dir / "filtered" / "test_001.wav").exists()
        assert not (out_dir / "filtered" / "test_002.wav").exists()

        # Verify quality report
        report_path = out_dir / "quality_report.json"
        assert report_path.exists()
