"""F5-TTS fine-tuning trainer for Polish language enhancement.

Fine-tunes F5-TTS (flow-matching) models using the built-in CFM Trainer.
Starts from the Gregniuki English/German/Polish checkpoint and continues
training on our filtered Polish data (M-AILABS + Wolne Lektury).

Data format:
  Input: filtered_manifest.jsonl (AudioSmith training pipeline output)
  Converted to: metadata.csv (pipe-delimited: audio_name|text) + audio/ dir

Architecture:
  CFM (Conditional Flow Matching)
    └── DiT (Diffusion Transformer) backbone
          dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
  Vocoder: vocos (built into f5-tts)
  Output: 24kHz mel spectrograms → waveform

License: Gregniuki checkpoint is CC-BY-NC-4.0 (non-commercial).
"""

import gc
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from audiosmith.exceptions import TrainingError

logger = logging.getLogger(__name__)

# Polish diacritics that must be in vocab
POLISH_CHARS: Set[str] = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

# Model architecture config (must match base checkpoint)
F5_MODEL_CONFIG = dict(
    dim=1024, depth=22, heads=16, ff_mult=2,
    text_dim=512, conv_layers=4,
)

# Mel spectrogram config (must match base)
F5_MEL_CONFIG = dict(
    target_sample_rate=24000,
    n_mel_channels=100,
    hop_length=256,
    win_length=1024,
    n_fft=1024,
)


@dataclass
class F5FineTuneConfig:
    """Configuration for F5-TTS fine-tuning."""

    base_checkpoint: str = "Gregniuki/F5-tts_English_German_Polish"
    base_ckpt_file: str = "Polish/model_500000.pt"
    base_vocab_file: str = "Polish/vocab.txt"

    train_dir: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("f5_checkpoints"))
    vocab_path: Optional[Path] = None

    # Training hyperparameters
    epochs: int = 10
    batch_size_per_gpu: int = 3200  # duration-based (frames)
    learning_rate: float = 7.5e-5
    warmup_steps: int = 500
    save_interval: int = 1000
    max_samples: int = 64
    grad_accumulation_steps: int = 1

    # Mel config (must match base model)
    target_sample_rate: int = 24000
    n_mel_channels: int = 100
    hop_length: int = 256

    # Hardware
    device: str = "cuda"
    resume_checkpoint: Optional[Path] = None


class F5FineTuneTrainer:
    """F5-TTS fine-tuning trainer using the built-in CFM Trainer."""

    def __init__(self, config: F5FineTuneConfig) -> None:
        self.config = config
        self._model: Any = None
        self._trainer: Any = None

    def prepare_data(
        self,
        manifest_jsonl: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Convert filtered_manifest.jsonl to F5-TTS training format.

        Reads our JSONL manifest and produces:
          - metadata.csv (pipe-delimited: audio_filename|text)
          - audio/ directory with symlinks to source WAVs

        Args:
            manifest_jsonl: Path to filtered_manifest.jsonl. If None,
                looks for it in config.train_dir.
            output_dir: Where to write converted data. Defaults to
                config.train_dir / 'f5_format'.

        Returns:
            Dict with stats: samples, total_hours, output_dir, missing_chars.
        """
        if manifest_jsonl is None:
            if self.config.train_dir is None:
                raise TrainingError(
                    "No manifest_jsonl or train_dir specified",
                    error_code="F5_TRAIN_DATA_ERR",
                )
            manifest_jsonl = self.config.train_dir / "filtered_manifest.jsonl"

        if not manifest_jsonl.exists():
            raise TrainingError(
                f"Manifest not found: {manifest_jsonl}",
                error_code="F5_TRAIN_DATA_ERR",
            )

        out_dir = output_dir or (self.config.train_dir or Path(".")) / "f5_format"
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        entries: List[Dict[str, Any]] = []
        with open(manifest_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if not entries:
            raise TrainingError(
                f"Empty manifest: {manifest_jsonl}",
                error_code="F5_TRAIN_DATA_ERR",
            )

        total_duration = 0.0
        written = 0
        missing_polish = set()
        metadata_lines: List[str] = []
        arrow_records: List[Dict[str, Any]] = []
        duration_list: List[float] = []

        for entry in entries:
            # Support both formats: {"audio": "path.wav"} and {"id": "name"}
            if "audio" in entry:
                audio_path = Path(entry["audio"])
            elif "id" in entry:
                audio_path = manifest_jsonl.parent / "filtered" / f"{entry['id']}.wav"
            else:
                continue
            text = entry.get("text", "").strip()
            duration = entry.get("duration", entry.get("duration_s", 0.0))

            if not text or not audio_path.exists():
                continue

            # Check for Polish diacritics coverage
            for ch in text:
                if ch in POLISH_CHARS:
                    pass  # good
                # Track chars that might need vocab extension
            text_chars = set(text)
            polish_in_text = text_chars & POLISH_CHARS
            if polish_in_text:
                missing_polish.update(polish_in_text)

            # Symlink audio into audio/ dir
            basename = audio_path.name
            link_path = audio_dir / basename
            if not link_path.exists():
                try:
                    link_path.symlink_to(audio_path.resolve())
                except OSError:
                    shutil.copy2(str(audio_path), str(link_path))

            abs_audio = str(link_path.resolve())
            metadata_lines.append(f"{basename}|{text}")
            arrow_records.append({"audio_path": abs_audio, "text": text, "duration": duration})
            duration_list.append(duration)
            total_duration += duration
            written += 1

        # Write metadata.csv
        metadata_path = out_dir / "metadata.csv"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines) + "\n")

        # Write raw.arrow + duration.json (F5-TTS dataset format)
        self._write_arrow_dataset(out_dir, arrow_records, duration_list)

        total_hours = total_duration / 3600.0
        logger.info(
            "Prepared %d samples (%.1fh) → %s",
            written, total_hours, out_dir,
        )

        return {
            "samples": written,
            "total_hours": round(total_hours, 2),
            "output_dir": str(out_dir),
            "metadata_path": str(metadata_path),
            "polish_chars_found": sorted(missing_polish),
        }

    def _write_arrow_dataset(
        self,
        out_dir: Path,
        records: List[Dict[str, Any]],
        durations: List[float],
    ) -> None:
        """Write raw.arrow + duration.json for F5-TTS dataset loading."""
        try:
            from datasets.arrow_writer import ArrowWriter
        except ImportError:
            raise TrainingError(
                "datasets not installed. Install: pip install datasets",
                error_code="F5_TRAIN_IMPORT_ERR",
            )

        arrow_path = out_dir / "raw.arrow"
        with ArrowWriter(path=str(arrow_path)) as writer:
            for record in records:
                writer.write(record)
            writer.finalize()

        dur_path = out_dir / "duration.json"
        with open(dur_path, "w", encoding="utf-8") as f:
            json.dump({"duration": durations}, f, ensure_ascii=False)

        logger.info(
            "Wrote arrow dataset: %d records → %s", len(records), arrow_path,
        )

    def extend_vocab(self, base_vocab_path: Optional[Path] = None) -> Path:
        """Ensure Polish diacritics are present in the vocab file.

        Args:
            base_vocab_path: Path to base vocab.txt. If None, downloads
                from HuggingFace.

        Returns:
            Path to the extended vocab file.
        """
        if base_vocab_path is None:
            base_vocab_path = self._download_vocab()

        if not base_vocab_path.exists():
            raise TrainingError(
                f"Vocab file not found: {base_vocab_path}",
                error_code="F5_TRAIN_VOCAB_ERR",
            )

        with open(base_vocab_path, "r", encoding="utf-8") as f:
            existing_chars = set(f.read())

        missing = POLISH_CHARS - existing_chars
        if not missing:
            logger.info("All Polish characters already in vocab")
            return base_vocab_path

        # Read lines and append missing chars
        with open(base_vocab_path, "r", encoding="utf-8") as f:
            lines = f.read().rstrip("\n").split("\n")

        for ch in sorted(missing):
            lines.append(ch)

        output_path = self.config.output_dir / "vocab_extended.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(
            "Extended vocab with %d Polish chars → %s",
            len(missing), output_path,
        )
        return output_path

    def train(self) -> Path:
        """Run F5-TTS fine-tuning.

        Uses the F5-TTS Trainer which auto-loads pretrained_*.pt files
        from checkpoint_path. We copy the base checkpoint there before
        creating the Trainer.

        Returns:
            Path to the output checkpoint directory.
        """
        self._validate_config()

        torch = self._get_torch()
        CFM, Trainer, get_tokenizer, DiT = self._import_f5_training()

        start_time = time.time()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve vocab
        vocab_path = self.config.vocab_path
        if vocab_path is None:
            vocab_path = self._download_vocab()

        vocab_char_map, vocab_size = get_tokenizer(
            str(vocab_path), tokenizer="custom",
        )
        logger.info("Vocab size: %d", vocab_size)

        # Copy base checkpoint to output dir (Trainer auto-discovers pretrained_*)
        ckpt_path = self._download_checkpoint()
        pretrained_name = f"pretrained_{ckpt_path.name}"
        pretrained_dest = self.config.output_dir / pretrained_name
        if not pretrained_dest.exists():
            shutil.copy2(str(ckpt_path), str(pretrained_dest))
            logger.info("Copied base checkpoint → %s", pretrained_dest)

        # Handle resume checkpoint
        if self.config.resume_checkpoint and self.config.resume_checkpoint.exists():
            resume_dest = self.config.output_dir / self.config.resume_checkpoint.name
            if not resume_dest.exists():
                shutil.copy2(str(self.config.resume_checkpoint), str(resume_dest))
            logger.info("Resume checkpoint: %s", resume_dest)

        # Mel spec config (shared by model + dataset)
        mel_spec_kwargs = dict(
            n_fft=F5_MEL_CONFIG["n_fft"],
            hop_length=self.config.hop_length,
            win_length=F5_MEL_CONFIG["win_length"],
            n_mel_channels=self.config.n_mel_channels,
            target_sample_rate=self.config.target_sample_rate,
            mel_spec_type="vocos",
        )

        # Build model
        model = self._build_model(CFM, DiT, vocab_size, vocab_char_map, mel_spec_kwargs)

        # Load dataset from arrow format
        train_data_dir = self.config.train_dir
        if train_data_dir is None:
            raise TrainingError(
                "train_dir must be set for training",
                error_code="F5_TRAIN_DATA_ERR",
            )

        f5_dir = train_data_dir / "f5_format"
        arrow_path = f5_dir / "raw.arrow"
        dur_path = f5_dir / "duration.json"
        if not arrow_path.exists() or not dur_path.exists():
            raise TrainingError(
                f"No raw.arrow/duration.json in {f5_dir}. Run prepare_data() first.",
                error_code="F5_TRAIN_DATA_ERR",
            )

        train_dataset = self._load_dataset(arrow_path, dur_path, mel_spec_kwargs)
        logger.info("Loaded %d training samples", len(train_dataset))

        # Create trainer
        trainer = Trainer(
            model,
            self.config.epochs,
            self.config.learning_rate,
            num_warmup_updates=self.config.warmup_steps,
            save_per_updates=self.config.save_interval,
            checkpoint_path=str(self.config.output_dir),
            batch_size_per_gpu=self.config.batch_size_per_gpu,
            batch_size_type="frame",
            max_samples=self.config.max_samples,
            grad_accumulation_steps=self.config.grad_accumulation_steps,
            logger="tensorboard",
            mel_spec_type="vocos",
        )
        self._trainer = trainer

        logger.info(
            "Starting F5-TTS training: %d epochs, lr=%s, batch=%d frames",
            self.config.epochs, self.config.learning_rate,
            self.config.batch_size_per_gpu,
        )

        try:
            trainer.train(
                train_dataset,
                resumable_with_seed=42,
            )
        except Exception as e:
            raise TrainingError(
                f"F5-TTS training failed: {e}",
                error_code="F5_TRAIN_ERR",
                original_error=e,
            )

        elapsed = time.time() - start_time
        logger.info("Training completed in %.1f minutes", elapsed / 60)

        # Find latest checkpoint
        ckpts = sorted(self.config.output_dir.glob("model_*.pt"))
        if ckpts:
            logger.info("Latest checkpoint: %s", ckpts[-1])
            return ckpts[-1]

        return self.config.output_dir

    def _build_model(
        self, CFM: Any, DiT: Any, vocab_size: int,
        vocab_char_map: Any = None, mel_spec_kwargs: Optional[Dict] = None,
    ) -> Any:
        """Build CFM model with DiT backbone."""
        model_cfg = {
            **F5_MODEL_CONFIG,
            "text_num_embeds": vocab_size,
            "mel_dim": self.config.n_mel_channels,
        }
        transformer = DiT(**model_cfg)
        if mel_spec_kwargs is None:
            mel_spec_kwargs = dict(
                n_fft=F5_MEL_CONFIG["n_fft"],
                hop_length=self.config.hop_length,
                win_length=F5_MEL_CONFIG["win_length"],
                n_mel_channels=self.config.n_mel_channels,
                target_sample_rate=self.config.target_sample_rate,
                mel_spec_type="vocos",
            )
        model = CFM(
            transformer=transformer,
            mel_spec_kwargs=mel_spec_kwargs,
            vocab_char_map=vocab_char_map,
        )
        return model

    def _load_dataset(
        self, arrow_path: Path, dur_path: Path, mel_spec_kwargs: Dict,
    ) -> Any:
        """Load training dataset from arrow format."""
        try:
            from datasets import Dataset as HFDataset
            from f5_tts.model.dataset import CustomDataset
        except ImportError:
            raise TrainingError(
                "datasets or f5-tts not installed",
                error_code="F5_TRAIN_IMPORT_ERR",
            )

        raw_data = HFDataset.from_file(str(arrow_path))
        with open(dur_path, "r", encoding="utf-8") as f:
            durations = json.load(f)["duration"]

        return CustomDataset(
            raw_data,
            durations=durations,
            preprocessed_mel=False,
            **mel_spec_kwargs,
        )

    def _validate_config(self) -> None:
        """Validate training config before starting."""
        torch = self._get_torch()

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise TrainingError(
                "CUDA requested but not available",
                error_code="F5_TRAIN_DEVICE_ERR",
            )

        if self.config.train_dir and not self.config.train_dir.exists():
            raise TrainingError(
                f"Training directory not found: {self.config.train_dir}",
                error_code="F5_TRAIN_DATA_ERR",
            )

        if self.config.epochs < 1:
            raise TrainingError(
                f"Epochs must be >= 1, got {self.config.epochs}",
                error_code="F5_TRAIN_CFG_ERR",
            )

    def _download_checkpoint(self) -> Path:
        """Download base checkpoint from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise TrainingError(
                "huggingface_hub not installed",
                error_code="F5_TRAIN_IMPORT_ERR",
            )
        return Path(hf_hub_download(
            repo_id=self.config.base_checkpoint,
            filename=self.config.base_ckpt_file,
        ))

    def _download_vocab(self) -> Path:
        """Download vocab.txt from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise TrainingError(
                "huggingface_hub not installed",
                error_code="F5_TRAIN_IMPORT_ERR",
            )
        return Path(hf_hub_download(
            repo_id=self.config.base_checkpoint,
            filename=self.config.base_vocab_file,
        ))

    def _get_torch(self) -> Any:
        """Lazy import torch."""
        try:
            import torch
            return torch
        except ImportError:
            raise TrainingError(
                "PyTorch not installed",
                error_code="F5_TRAIN_IMPORT_ERR",
            )

    def _import_f5_training(self) -> tuple:
        """Lazy import F5-TTS training modules."""
        try:
            from f5_tts.model import CFM, DiT
            from f5_tts.model.trainer import Trainer
            from f5_tts.model.utils import get_tokenizer
            return CFM, Trainer, get_tokenizer, DiT
        except ImportError:
            raise TrainingError(
                "f5-tts not installed. Install: pip install f5-tts",
                error_code="F5_TRAIN_IMPORT_ERR",
            )

    def cleanup(self) -> None:
        """Release model and trainer."""
        self._model = None
        self._trainer = None
        gc.collect()
        try:
            torch = self._get_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except TrainingError:
            pass
        logger.info("F5 trainer cleaned up")
