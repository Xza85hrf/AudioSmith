"""Vocal isolation using Demucs — separates vocals from background audio."""

import gc
import logging
from pathlib import Path
from typing import Dict, Optional

from audiosmith.exceptions import VocalIsolationError

logger = logging.getLogger(__name__)

# Process long audio in chunks to avoid CUDA OOM
# 5 minutes at 44100 Hz ≈ 13.2M samples ≈ 105MB per chunk (stereo float32)
CHUNK_DURATION_S = 300


class VocalIsolator:
    """Separates vocals from background audio using Demucs.

    Uses htdemucs model by default. Outputs 16kHz mono WAV files
    compatible with Whisper transcription.

    For long audio (>30 min), processes in chunks to avoid GPU OOM.
    """

    def __init__(self, model_name: str = 'htdemucs', device: str = 'cuda') -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._apply_model = None

    def load_model(self) -> None:
        """Load Demucs model onto device."""
        try:
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ImportError:
            raise VocalIsolationError(
                "Demucs is required for vocal isolation. "
                "Install with: pip install 'audiosmith[quality]'"
            )

        self._apply_model = apply_model
        self._model = get_model(self.model_name)
        self._model.to(self.device)
        logger.info("Loaded Demucs model '%s' on %s", self.model_name, self.device)

    def _isolate_chunked(
        self, wav: "torch.Tensor", chunk_samples: int,  # noqa: F821
    ) -> tuple["torch.Tensor", "torch.Tensor"]:  # noqa: F821
        """Process audio in chunks to avoid GPU OOM on long files.

        Args:
            wav: Stereo audio tensor [2, time] on CPU at model samplerate
            chunk_samples: Number of samples per chunk

        Returns:
            Tuple of (vocals, background) tensors [channels, time] on CPU
        """
        import torch

        n_samples = wav.shape[1]
        n_chunks = (n_samples + chunk_samples - 1) // chunk_samples
        vocals_idx = self._model.sources.index('vocals')
        bg_mask = [i for i in range(len(self._model.sources)) if i != vocals_idx]

        all_vocals: list[torch.Tensor] = []
        all_background: list[torch.Tensor] = []

        logger.info(
            "Processing %d samples (%.1fh) in %d chunks of %.1f min each",
            n_samples, n_samples / self._model.samplerate / 3600,
            n_chunks, chunk_samples / self._model.samplerate / 60,
        )

        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, n_samples)
            chunk = wav[:, start:end]  # [2, chunk_samples] on CPU

            logger.info("Processing chunk %d/%d (%.1f%%)", i + 1, n_chunks, 100 * end / n_samples)

            # Move chunk to GPU for inference
            chunk_dev = chunk.unsqueeze(0).to(self.device)  # [1, 2, time]

            with torch.no_grad():
                sources = self._apply_model(
                    self._model, chunk_dev, split=True, segment=7.5, overlap=0.25,
                    device=self.device,
                )

            # sources: [1, n_sources, channels, time] -> [n_sources, channels, time]
            sources = sources.squeeze(0).cpu()

            # Extract vocals and background
            vocals_chunk = sources[vocals_idx]  # [channels, time]
            background_chunk = sources[bg_mask].sum(dim=0)  # [channels, time]

            all_vocals.append(vocals_chunk)
            all_background.append(background_chunk)

            # Free GPU memory
            del chunk_dev, sources
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # Concatenate all chunks on CPU
        vocals_full = torch.cat(all_vocals, dim=1)
        background_full = torch.cat(all_background, dim=1)

        return vocals_full, background_full

    def isolate(
        self, audio_path: Path, output_dir: Optional[Path] = None,
        mixing_sample_rate: int = 48000,
    ) -> Dict[str, Path]:
        """Separate vocals from background audio.

        Returns dict with 'vocals_path', 'background_path', and
        'background_hq_path' keys.
        - vocals_path / background_path: 16kHz mono WAV (Whisper-compatible)
        - background_hq_path: stereo WAV at *mixing_sample_rate* for mixing
        """
        import torch
        import torchaudio

        if self._model is None:
            self.load_model()

        audio_path = Path(audio_path)
        if output_dir is None:
            output_dir = audio_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare audio
        wav, sr = torchaudio.load(str(audio_path))

        # Demucs requires stereo input
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Resample to model's expected rate
        if sr != self._model.samplerate:
            wav = torchaudio.transforms.Resample(sr, self._model.samplerate)(wav)

        # Determine if chunked processing is needed
        duration_s = wav.shape[1] / self._model.samplerate
        chunk_samples = int(CHUNK_DURATION_S * self._model.samplerate)

        # Use chunked processing for audio > 30 minutes to avoid OOM
        if duration_s > 1800:
            logger.info(
                "Audio duration %.1fh exceeds 30 min, using chunked processing",
                duration_s / 3600,
            )
            vocals, background = self._isolate_chunked(wav, chunk_samples)
        else:
            # Original single-pass processing for shorter audio
            wav_dev = wav.unsqueeze(0).to(self.device)
            with torch.no_grad():
                sources = self._apply_model(
                    self._model, wav_dev, split=True, segment=7.5, overlap=0.25,
                    device=self.device,
                )
            sources = sources.squeeze(0).cpu()

            vocals_idx = self._model.sources.index('vocals')
            vocals = sources[vocals_idx]
            bg_mask = [i for i in range(sources.shape[0]) if i != vocals_idx]
            background = sources[bg_mask].sum(dim=0)

            del wav_dev, sources
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # ── 16kHz mono outputs (Whisper-compatible) ──
        vocals_mono = vocals.mean(dim=0, keepdim=True)
        bg_mono = background.mean(dim=0, keepdim=True)

        target_sr = 16000
        if self._model.samplerate != target_sr:
            resample = torchaudio.transforms.Resample(self._model.samplerate, target_sr)
            vocals_mono = resample(vocals_mono)
            bg_mono = resample(bg_mono)

        stem = audio_path.stem
        vocals_path = output_dir / f'{stem}_vocals.wav'
        bg_path = output_dir / f'{stem}_background.wav'

        torchaudio.save(str(vocals_path), vocals_mono, target_sr)
        torchaudio.save(str(bg_path), bg_mono, target_sr)

        # ── High-quality stereo background for mixing ──
        background_hq = background  # [2, time] at model samplerate
        if self._model.samplerate != mixing_sample_rate:
            background_hq = torchaudio.transforms.Resample(
                self._model.samplerate, mixing_sample_rate,
            )(background_hq)
        bg_hq_path = output_dir / f'{stem}_background_hq.wav'
        torchaudio.save(str(bg_hq_path), background_hq, mixing_sample_rate)

        logger.info(
            "Isolated vocals -> %s, background -> %s, background_hq -> %s",
            vocals_path.name, bg_path.name, bg_hq_path.name,
        )
        return {
            'vocals_path': vocals_path,
            'background_path': bg_path,
            'background_hq_path': bg_hq_path,
        }

    def unload(self) -> None:
        """Release model and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._apply_model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Vocal isolator unloaded")
