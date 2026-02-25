"""Multi-voice TTS — speaker-aware voice cloning with emotion modulation."""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from audiosmith.tts import ChatterboxTTS

logger = logging.getLogger(__name__)


class MultiVoiceTTS:
    """Multi-voice TTS wrapper — maps speakers to voice prompts with emotion modulation.

    Wraps ChatterboxTTS to add:
    - Speaker-to-voice-prompt mapping (from diarization)
    - Emotion-to-TTS-parameter bridging (from emotion detection)
    """

    def __init__(
        self,
        device: str = 'cuda',
        language: str = 'pl',
        default_exaggeration: float = 0.5,
        default_cfg_weight: float = 0.5,
    ) -> None:
        self.device = device
        self.language = language
        self.default_exaggeration = default_exaggeration
        self.default_cfg_weight = default_cfg_weight
        self._engine: Optional[ChatterboxTTS] = None
        self._voice_map: Dict[str, str] = {}  # speaker_id -> audio_prompt_path
        self._default_prompt: Optional[str] = None

    def load_model(self) -> None:
        """Initialize and load the Chatterbox TTS engine."""
        self._engine = ChatterboxTTS(device=self.device)
        self._engine.load_model()
        logger.info("Multi-voice TTS engine loaded on %s", self.device)

    def assign_voice(self, speaker_id: str, audio_prompt_path: str) -> None:
        """Map a speaker ID to an audio prompt file for voice cloning."""
        if not Path(audio_prompt_path).exists():
            raise ValueError(f"Audio prompt not found: {audio_prompt_path}")
        self._voice_map[speaker_id] = audio_prompt_path
        logger.info("Assigned speaker '%s' -> %s", speaker_id, audio_prompt_path)

    def set_default_voice(self, audio_prompt_path: str) -> None:
        """Set the default voice prompt for unmapped speakers."""
        if not Path(audio_prompt_path).exists():
            raise ValueError(f"Audio prompt not found: {audio_prompt_path}")
        self._default_prompt = audio_prompt_path

    def auto_assign_voices(self, speaker_ids: List[str], voice_dir: Path) -> None:
        """Auto-discover {speaker_id}.wav files in a directory."""
        voice_dir = Path(voice_dir)
        assigned = 0
        for sid in speaker_ids:
            prompt_path = voice_dir / f'{sid}.wav'
            if prompt_path.exists():
                self._voice_map[sid] = str(prompt_path)
                assigned += 1
        logger.info("Auto-assigned %d/%d voices from %s", assigned, len(speaker_ids), voice_dir)

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        emotion_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Synthesize speech with speaker voice and emotion modulation.

        Args:
            text: Text to synthesize.
            speaker_id: Speaker ID for voice prompt lookup.
            emotion_params: Dict with 'exaggeration' and/or 'cfg_weight' keys
                           (from emotion.get_prosody_params()).
        """
        if self._engine is None:
            self.load_model()

        prompt = self._voice_map.get(speaker_id) if speaker_id else None
        if prompt is None:
            prompt = self._default_prompt

        if emotion_params:
            exag = emotion_params.get('exaggeration', self.default_exaggeration)
            cfg = emotion_params.get('cfg_weight', self.default_cfg_weight)
        else:
            exag = self.default_exaggeration
            cfg = self.default_cfg_weight

        return self._engine.synthesize(
            text,
            language=self.language,
            audio_prompt_path=prompt,
            exaggeration=exag,
            cfg_weight=cfg,
        )

    @property
    def sample_rate(self) -> int:
        """Return engine sample rate, or 24000 if engine not loaded."""
        return self._engine.sample_rate if self._engine else 24000

    @property
    def voice_count(self) -> int:
        """Return number of assigned voice mappings."""
        return len(self._voice_map)

    def unload(self) -> None:
        """Release engine and GPU memory."""
        if self._engine is not None:
            self._engine.cleanup()
            self._engine = None
        self._voice_map.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Multi-voice TTS unloaded")
