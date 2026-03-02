"""
Test suite for F5-TTS fine-tuning trainer.

This module contains comprehensive tests for the F5FineTuneConfig dataclass
and F5FineTuneTrainer class, covering data preparation, vocabulary extension,
model building, and training workflows.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.exceptions import TrainingError
from audiosmith.f5_finetune import F5FineTuneConfig, F5FineTuneTrainer


logger = logging.getLogger(__name__)


def _make_audio(duration: float = 2.0, freq: float = 300.0) -> np.ndarray:
    """Create synthetic audio for testing.

    Args:
        duration: Duration of audio in seconds
        freq: Frequency of the carrier sine wave

    Returns:
        Float32 numpy array of audio samples at 24kHz sample rate
    """
    SR = 24000
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    carrier = np.sin(2 * np.pi * freq * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    envelope = np.clip(envelope, 0.05, 1.0)
    return (0.3 * carrier * envelope).astype(np.float32)


class TestF5FineTuneConfig:
    """Test suite for F5FineTuneConfig dataclass."""

    def test_default_values_initialized_correctly(self):
        """Test that default config values are initialized properly."""
        config = F5FineTuneConfig()

        assert config.base_checkpoint == "Gregniuki/F5-tts_English_German_Polish"
        assert config.base_ckpt_file == "Polish/model_500000.pt"
        assert config.device == "cuda"
        assert config.epochs == 10
        assert config.batch_size_per_gpu == 3200
        assert config.learning_rate == 7.5e-5

    def test_custom_values_accepted(self):
        """Test that custom config values are properly accepted."""
        config = F5FineTuneConfig(
            train_dir=Path("/path/to/train"),
            output_dir=Path("/path/to/output"),
            device="cpu",
            epochs=20,
            batch_size_per_gpu=1600,
            learning_rate=5e-5,
        )

        assert config.train_dir == Path("/path/to/train")
        assert config.output_dir == Path("/path/to/output")
        assert config.device == "cpu"
        assert config.epochs == 20
        assert config.batch_size_per_gpu == 1600
        assert config.learning_rate == 5e-5

    def test_optional_fields_can_be_none(self):
        """Test that optional fields can explicitly be set to None."""
        config = F5FineTuneConfig(
            train_dir=None,
            output_dir=None,
            vocab_path=None,
            resume_checkpoint=None,
        )

        assert config.train_dir is None
        assert config.output_dir is None
        assert config.vocab_path is None
        assert config.resume_checkpoint is None

    def test_config_is_dataclass(self):
        """Test that F5FineTuneConfig is a proper dataclass."""
        assert hasattr(F5FineTuneConfig, '__dataclass_fields__')
        fields = F5FineTuneConfig.__dataclass_fields__
        assert 'train_dir' in fields
        assert 'output_dir' in fields
        assert 'device' in fields
        assert 'epochs' in fields
        assert 'batch_size_per_gpu' in fields
        assert 'learning_rate' in fields


class TestF5FineTuneTrainerInit:
    """Test suite for F5FineTuneTrainer initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = F5FineTuneConfig(
            train_dir=Path("/tmp/train"),
            output_dir=Path("/tmp/output"),
            device="cuda"
        )

    def test_initializes_with_config(self):
        """Test trainer initializes properly with config."""
        trainer = F5FineTuneTrainer(self.config)
        assert trainer.config == self.config

    def test_sets_model_none_initially(self):
        """Test that _model is set to None on initialization."""
        trainer = F5FineTuneTrainer(self.config)
        assert trainer._model is None

    def test_sets_trainer_none_initially(self):
        """Test that _trainer is set to None on initialization."""
        trainer = F5FineTuneTrainer(self.config)
        assert trainer._trainer is None


class TestF5FineTuneTrainerPrepareData:
    """Test suite for F5FineTuneTrainer.prepare_data() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = F5FineTuneConfig(
            train_dir=None,
            output_dir=None,
            device="cuda"
        )

    def test_raises_error_if_train_dir_not_set(self):
        """Test that TrainingError is raised if train_dir is not set."""
        trainer = F5FineTuneTrainer(self.config)

        with pytest.raises(TrainingError) as exc_info:
            trainer.prepare_data()

        assert exc_info.value.error_code == "F5_TRAIN_DATA_ERR"

    def test_raises_error_if_train_dir_not_exist(self):
        """Test that TrainingError is raised if train_dir doesn't exist."""
        self.config.train_dir = Path("/nonexistent/path/train")
        self.config.output_dir = Path("/tmp/output")

        trainer = F5FineTuneTrainer(self.config)

        with pytest.raises(TrainingError) as exc_info:
            trainer.prepare_data()

        assert exc_info.value.error_code == "F5_TRAIN_DATA_ERR"

    def test_raises_error_if_manifest_missing(self):
        """Test that TrainingError is raised if filtered_manifest.jsonl is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            self.config.train_dir = train_dir
            self.config.output_dir = output_dir

            trainer = F5FineTuneTrainer(self.config)

            with pytest.raises(TrainingError) as exc_info:
                trainer.prepare_data()

            assert exc_info.value.error_code == "F5_TRAIN_DATA_ERR"

    def test_raises_error_if_manifest_empty(self):
        """Test that TrainingError is raised if manifest is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            # Create empty manifest
            manifest_path = train_dir / "filtered_manifest.jsonl"
            manifest_path.write_text("")

            self.config.train_dir = train_dir
            self.config.output_dir = output_dir

            trainer = F5FineTuneTrainer(self.config)

            with pytest.raises(TrainingError) as exc_info:
                trainer.prepare_data()

            assert exc_info.value.error_code == "F5_TRAIN_DATA_ERR"

    def test_reads_jsonl_manifest_correctly(self):
        """Test that JSONL manifest is read correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            # Create actual audio files so audio_path.exists() passes
            for name in ["sample1.wav", "sample2.wav"]:
                (train_dir / name).write_bytes(b"\x00" * 100)

            # Create manifest with absolute paths
            manifest_path = train_dir / "filtered_manifest.jsonl"
            entries = [
                {"audio": str(train_dir / "sample1.wav"), "text": "Hello world", "duration": 2.0},
                {"audio": str(train_dir / "sample2.wav"), "text": "Test audio", "duration": 1.5},
            ]
            with open(manifest_path, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

            self.config.train_dir = train_dir
            self.config.output_dir = output_dir

            trainer = F5FineTuneTrainer(self.config)

            result = trainer.prepare_data(output_dir=output_dir)

            assert result["samples"] == 2
            assert result["output_dir"] == str(output_dir)

    def test_creates_metadata_csv_with_pipe_delimiter(self):
        """Test that metadata.csv is created with pipe-delimited format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            # Create actual audio files
            for name in ["sample1.wav", "sample2.wav"]:
                (train_dir / name).write_bytes(b"\x00" * 100)

            # Create manifest with absolute paths
            manifest_path = train_dir / "filtered_manifest.jsonl"
            entries = [
                {"audio": str(train_dir / "sample1.wav"), "text": "Hello world", "duration": 2.0},
                {"audio": str(train_dir / "sample2.wav"), "text": "Testing audio", "duration": 1.5},
            ]
            with open(manifest_path, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

            self.config.train_dir = train_dir
            self.config.output_dir = output_dir

            trainer = F5FineTuneTrainer(self.config)

            result = trainer.prepare_data(output_dir=output_dir)

            # Check metadata.csv was created
            metadata_path = output_dir / "metadata.csv"
            assert metadata_path.exists()

            # Check pipe-delimited format
            content = metadata_path.read_text()
            lines = content.strip().split('\n')
            assert len(lines) == 2

            # Each line should be basename.wav|text
            for line in lines:
                parts = line.split('|')
                assert len(parts) == 2
                assert parts[0].endswith('.wav')
                assert len(parts[1]) > 0

    def test_validates_polish_diacritics_in_text(self):
        """Test that Polish diacritics in text are validated/handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            # Create actual audio files
            for name in ["polish1.wav", "polish2.wav"]:
                (train_dir / name).write_bytes(b"\x00" * 100)

            # Create manifest with Polish diacritics and absolute paths
            manifest_path = train_dir / "filtered_manifest.jsonl"
            entries = [
                {"audio": str(train_dir / "polish1.wav"), "text": "Zażółć gęślą jaźń", "duration": 2.0},
                {"audio": str(train_dir / "polish2.wav"), "text": "Łódź i Północ", "duration": 1.5},
            ]
            with open(manifest_path, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

            self.config.train_dir = train_dir
            self.config.output_dir = output_dir

            trainer = F5FineTuneTrainer(self.config)

            result = trainer.prepare_data(output_dir=output_dir)

            # Should handle Polish characters without error
            assert result["samples"] == 2

            # Check metadata contains Polish chars
            metadata_path = output_dir / "metadata.csv"
            content = metadata_path.read_text()
            assert "Zażółć" in content
            assert "gęślą" in content
            assert "jaźń" in content


class TestF5FineTuneTrainerExtendVocab:
    """Test suite for F5FineTuneTrainer.extend_vocab() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = F5FineTuneConfig(
            train_dir=Path("/tmp/train"),
            output_dir=Path("/tmp/output"),
            device="cuda"
        )

    def test_takes_base_vocab_path(self):
        """Test that extend_vocab accepts base vocab.txt path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base vocab file
            vocab_path = Path(tmpdir) / "vocab.txt"
            vocab_path.write_text("a\nb\nc\n")

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            self.config.output_dir = output_dir
            trainer = F5FineTuneTrainer(self.config)

            result_path = trainer.extend_vocab(vocab_path)

            assert result_path is not None
            assert Path(result_path).exists()

    def test_ensures_polish_diacritics_in_vocab(self):
        """Test that Polish lowercase diacritics are added to vocab."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base vocab without Polish chars
            vocab_path = Path(tmpdir) / "vocab.txt"
            vocab_path.write_text("a\nb\nc\n")

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            self.config.output_dir = output_dir
            trainer = F5FineTuneTrainer(self.config)

            result_path = trainer.extend_vocab(vocab_path)

            # Read extended vocab
            extended_vocab = Path(result_path).read_text()

            # Check lowercase Polish chars
            polish_lower = "ąćęłńóśźż"
            for char in polish_lower:
                assert char in extended_vocab, f"Missing: {char}"

    def test_ensures_uppercase_polish_diacritics(self):
        """Test that Polish uppercase diacritics are added to vocab."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.txt"
            vocab_path.write_text("a\nb\nc\n")

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            self.config.output_dir = output_dir
            trainer = F5FineTuneTrainer(self.config)

            result_path = trainer.extend_vocab(vocab_path)

            extended_vocab = Path(result_path).read_text()

            # Check uppercase Polish chars
            polish_upper = "ĄĆĘŁŃÓŚŹŻ"
            for char in polish_upper:
                assert char in extended_vocab, f"Missing: {char}"

    def test_writes_extended_vocab_to_output_dir(self):
        """Test that extended vocab is written to output_dir/vocab_extended.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.txt"
            vocab_path.write_text("a\nb\n")

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            self.config.output_dir = output_dir
            trainer = F5FineTuneTrainer(self.config)

            result_path = trainer.extend_vocab(vocab_path)

            expected_path = output_dir / "vocab_extended.txt"
            assert str(result_path) == str(expected_path)
            assert expected_path.exists()

    def test_preserves_original_characters(self):
        """Test that original vocab characters are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_content = "abc\ndefgh\nijklm\nnopqr\n"
            vocab_path = Path(tmpdir) / "vocab.txt"
            vocab_path.write_text(original_content)

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            self.config.output_dir = output_dir
            trainer = F5FineTuneTrainer(self.config)

            result_path = trainer.extend_vocab(vocab_path)

            extended_vocab = Path(result_path).read_text()

            # Original chars should be present
            for char in "abcdefghijklmnopqr":
                assert char in extended_vocab


class TestF5FineTuneTrainerValidateConfig:
    """Test suite for F5FineTuneTrainer._validate_config() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = F5FineTuneConfig(
            train_dir=Path("/tmp/train"),
            output_dir=Path("/tmp/output"),
            device="cuda"
        )

    def test_accepts_cuda_device(self):
        """Test that device='cuda' is accepted when available."""
        self.config.device = "cuda"
        self.config.train_dir = None
        trainer = F5FineTuneTrainer(self.config)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.object(trainer, "_get_torch", return_value=mock_torch):
            trainer._validate_config()

    def test_accepts_cpu_device(self):
        """Test that device='cpu' is accepted."""
        self.config.device = "cpu"
        self.config.train_dir = None
        trainer = F5FineTuneTrainer(self.config)

        mock_torch = MagicMock()
        with patch.object(trainer, "_get_torch", return_value=mock_torch):
            trainer._validate_config()


class TestF5FineTuneTrainerBuildModel:
    """Test suite for F5FineTuneTrainer._build_model() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = F5FineTuneConfig(
            train_dir=Path("/tmp/train"),
            output_dir=Path("/tmp/output"),
            device="cuda"
        )

    def test_build_model_creates_cfm_instance(self):
        """Test _build_model creates CFM with DiT backbone."""
        trainer = F5FineTuneTrainer(self.config)

        mock_dit_class = MagicMock()
        mock_cfm_class = MagicMock()
        mock_cfm_class.return_value = MagicMock()

        vocab_size = 256
        model = trainer._build_model(mock_cfm_class, mock_dit_class, vocab_size)

        # CFM should be created with transformer kwarg
        assert mock_cfm_class.called
        assert mock_dit_class.called

    def test_build_model_passes_vocab_size(self):
        """Test _build_model passes correct vocab size to DiT."""
        trainer = F5FineTuneTrainer(self.config)

        mock_dit_class = MagicMock()
        mock_cfm_class = MagicMock()

        trainer._build_model(mock_cfm_class, mock_dit_class, 512)

        # DiT should receive text_num_embeds=512
        dit_kwargs = mock_dit_class.call_args.kwargs
        assert dit_kwargs["text_num_embeds"] == 512


class TestF5FineTuneTrainerTrain:
    """Test suite for F5FineTuneTrainer.train() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = F5FineTuneConfig(
            train_dir=None,
            device="cuda",
        )

    def test_train_requires_train_dir(self, tmp_path):
        """Test that TrainingError is raised if train_dir is not set."""
        self.config.train_dir = None
        self.config.output_dir = tmp_path / "output"
        trainer = F5FineTuneTrainer(self.config)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.load.return_value = {}
        mock_get_tok = MagicMock(return_value=({}, 256))

        with patch.object(trainer, "_get_torch", return_value=mock_torch), \
             patch.object(trainer, "_import_f5_training", return_value=(MagicMock(), MagicMock(), mock_get_tok, MagicMock())), \
             patch.object(trainer, "_download_vocab", return_value=Path("/fake/vocab")), \
             patch.object(trainer, "_download_checkpoint", return_value=Path("/fake/ckpt")), \
             patch("audiosmith.f5_finetune.shutil") as mock_shutil:
            with pytest.raises(TrainingError) as exc_info:
                trainer.train()

        assert exc_info.value.error_code == "F5_TRAIN_DATA_ERR"


class TestF5FineTuneIntegration:
    """Integration tests for F5FineTuneTrainer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = F5FineTuneConfig(
            train_dir=None,
            output_dir=None,
            device="cuda"
        )

    def test_full_prepare_data_workflow_manifest_to_metadata(self):
        """Test complete prepare_data workflow from manifest to metadata.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            # Create actual audio files
            for name in ["first.wav", "second.wav", "third.wav"]:
                (train_dir / name).write_bytes(b"\x00" * 100)

            # Create manifest with absolute paths
            manifest_path = train_dir / "filtered_manifest.jsonl"
            entries = [
                {"audio": str(train_dir / "first.wav"), "text": "First sample text", "duration": 2.0},
                {"audio": str(train_dir / "second.wav"), "text": "Second sample", "duration": 1.5},
                {"audio": str(train_dir / "third.wav"), "text": "Third entry here", "duration": 3.0},
            ]
            with open(manifest_path, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

            self.config.train_dir = train_dir
            self.config.output_dir = output_dir

            trainer = F5FineTuneTrainer(self.config)

            # Execute prepare_data with explicit output_dir
            result = trainer.prepare_data(output_dir=output_dir)

            # Verify results
            assert result["samples"] == 3

            # Check metadata.csv exists and has correct content
            metadata_path = output_dir / "metadata.csv"
            assert metadata_path.exists()

            metadata_content = metadata_path.read_text()
            lines = metadata_content.strip().split('\n')

            # Should have 3 lines (one per sample)
            assert len(lines) == 3

            # Each line should be pipe-delimited
            for line in lines:
                parts = line.split('|')
                assert len(parts) == 2
                basename, text = parts
                assert basename.endswith('.wav')
                assert len(text) > 0
