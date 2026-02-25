"""Vocal isolation using Demucs — separates vocals from background audio."""

import gc
import logging
from pathlib import Path
from typing import Dict, Optional

from audiosmith.exceptions import VocalIsolationError

logger = logging.getLogger(__name__)


class VocalIsolator:
    """Separates vocals from background audio using Demucs.

    Uses htdemucs model by default. Outputs 16kHz mono WAV files
    compatible with Whisper transcription.
    """

    def __init__(self, model_name: str = 'htdemucs', device: str = 'cuda') -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._apply_model = None

    def load_model(self) -> None:
        """Load Demucs model onto device."""
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
        except ImportError:
            raise VocalIsolationError(
                "Demucs is required for vocal isolation. "
                "Install with: pip install 'audiosmith[quality]'"
            )

        self._apply_model = apply_model
        self._model = get_model(self.model_name)
        self._model.to(self.device)
        logger.info("Loaded Demucs model '%s' on %s", self.model_name, self.device)

    def isolate(self, audio_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Separate vocals from background audio.

        Returns dict with 'vocals_path' and 'background_path' keys.
        Output files are 16kHz mono WAV.
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

        # Run separation: apply_model expects [batch, channels, time]
        wav = wav.unsqueeze(0).to(self.device)
        with torch.no_grad():
            sources = self._apply_model(self._model, wav, device=self.device)
        # sources shape: [batch, n_sources, channels, time]
        sources = sources.squeeze(0)  # [n_sources, channels, time]

        # Extract vocals and sum remaining stems for background
        vocals_idx = self._model.sources.index('vocals')
        vocals = sources[vocals_idx]  # [channels, time]
        bg_mask = [i for i in range(sources.shape[0]) if i != vocals_idx]
        background = sources[bg_mask].sum(dim=0)  # [channels, time]

        # Convert to 16kHz mono for Whisper compatibility
        vocals_mono = vocals.mean(dim=0, keepdim=True).cpu()
        bg_mono = background.mean(dim=0, keepdim=True).cpu()

        target_sr = 16000
        if self._model.samplerate != target_sr:
            resample = torchaudio.transforms.Resample(self._model.samplerate, target_sr)
            vocals_mono = resample(vocals_mono)
            bg_mono = resample(bg_mono)

        # Save outputs
        stem = audio_path.stem
        vocals_path = output_dir / f'{stem}_vocals.wav'
        bg_path = output_dir / f'{stem}_background.wav'

        torchaudio.save(str(vocals_path), vocals_mono, target_sr)
        torchaudio.save(str(bg_path), bg_mono, target_sr)

        logger.info("Isolated vocals -> %s, background -> %s", vocals_path.name, bg_path.name)
        return {'vocals_path': vocals_path, 'background_path': bg_path}

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
