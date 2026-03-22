"""TTS synthesis mixin for the DubbingPipeline.

This module contains all TTS engine initialization and synthesis logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from audiosmith.emotion_config import EMOTION_STYLE_MAP
from audiosmith.pipeline_config import ENGINE_PP_PRESETS, LANGUAGE_PP_OVERRIDES

if TYPE_CHECKING:
    from audiosmith.models import DubbingConfig, DubbingSegment, PipelineState

logger = logging.getLogger(__name__)


class TTSSynthesisMixin:
    """Mixin providing TTS engine initialization and synthesis methods."""

    # Languages each engine handles well
    _QWEN3_LANGS = {'en', 'zh', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it'}
    _PIPER_LANGS = {'en', 'pl', 'de'}
    _FISH_LANGS = {'en', 'zh', 'ja', 'ko', 'de', 'fr', 'es', 'pt', 'ru', 'nl', 'it', 'pl', 'ar'}
    _INDEXTTS_LANGS = {'en', 'zh'}
    _COSYVOICE_LANGS = {'zh', 'en', 'ja', 'ko', 'de', 'es', 'fr', 'it', 'ru'}
    _ORPHEUS_LANGS = {'en', 'zh', 'es', 'fr', 'de', 'it', 'pt', 'hi', 'ko', 'tr', 'ja', 'th', 'ar'}
    _ELEVENLABS_LANGS = {
        'en', 'es', 'de', 'fr', 'it', 'pt', 'pl', 'ru', 'ja', 'ko', 'zh',
        'ar', 'bg', 'cs', 'da', 'nl', 'fi', 'el', 'hi', 'hu', 'id', 'ms',
        'no', 'ro', 'sk', 'sv', 'ta', 'th', 'tr', 'uk', 'vi',
    }

    _PIPER_VOICES = {
        'en': 'en_US-lessac-medium',
        'pl': 'pl_PL-darkman-medium',
        'de': 'de_DE-thorsten-medium',
    }

    config: DubbingConfig
    state: PipelineState

    def _resolve_engine(self) -> str:
        """Pick the best TTS engine for the target language."""
        import os

        target = self.config.target_language
        # Prefer ElevenLabs when API key is available (best quality)
        if os.getenv('ELEVENLABS_API_KEY') and target in self._ELEVENLABS_LANGS:
            logger.info("Auto-selected ElevenLabs TTS for '%s' (API key present)", target)
            return 'elevenlabs'
        if os.getenv('FISH_API_KEY') and target in self._FISH_LANGS:
            logger.info("Auto-selected Fish Speech TTS for '%s' (API key present)", target)
            return 'fish'
        if self.config.cosyvoice_model_dir and target in self._COSYVOICE_LANGS:
            logger.info("Auto-selected CosyVoice2 for '%s' (local, highest MOS)", target)
            return 'cosyvoice'
        if self.config.detect_emotion and target in self._INDEXTTS_LANGS:
            logger.info("Auto-selected IndexTTS-2 for '%s' (emotion-aware)", target)
            return 'indextts'
        if target in self._ORPHEUS_LANGS:
            logger.info("Auto-selected Orpheus TTS for '%s' (expressive local)", target)
            return 'orpheus'
        if target in self._QWEN3_LANGS:
            logger.info("Auto-selected Qwen3 TTS for '%s'", target)
            return 'qwen3'
        if target in self._PIPER_LANGS:
            # Only use Piper if the native voice model is actually installed
            voice = self._PIPER_VOICES.get(target)
            if voice:
                voice_dir = Path.home() / '.local' / 'share' / 'piper-voices'
                if (voice_dir / f'{voice}.onnx').exists():
                    logger.info("Auto-selected Piper TTS for '%s' (native voice model)", target)
                    return 'piper'
                logger.info(
                    "Piper voice '%s' not installed, falling back to Chatterbox for '%s'",
                    voice, target,
                )
        logger.info("Auto-selected Chatterbox TTS for '%s' (multilingual fallback)", target)
        return 'chatterbox'

    def _generate_tts(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        """Generate TTS audio for all segments."""
        import soundfile as sf

        from audiosmith.pipeline.helpers import _dedup_repeated_words

        tts_dir = Path(self.config.output_dir) / 'tts_segments'
        tts_dir.mkdir(parents=True, exist_ok=True)

        tts_engine_name = getattr(self.config, 'tts_engine', 'chatterbox')
        if tts_engine_name == 'auto':
            tts_engine_name = self._resolve_engine()
        prompt = str(self.config.audio_prompt_path) if self.config.audio_prompt_path else None

        # Determine if this is Chatterbox with multi-voice features
        use_multi = False
        if tts_engine_name == 'chatterbox':
            engine, sample_rate, use_multi = self._init_chatterbox_engine(segments)
        else:
            # Use factory for all other engines
            engine = self._create_engine_with_factory(tts_engine_name)
            sample_rate = engine.sample_rate

        for seg in segments:
            text = seg.translated_text or seg.original_text
            if not text.strip():
                continue

            # Skip segments whose TTS audio already exists on disk (resume-safe)
            wav_path = tts_dir / f'seg_{seg.index:04d}.wav'
            if wav_path.exists() and wav_path.stat().st_size > 0:
                seg.tts_audio_path = wav_path
                info = sf.info(str(wav_path))
                seg.tts_duration_ms = int(info.duration * 1000)
                logger.debug("Skipping TTS for segment %d (cached: %s)", seg.index, wav_path)
                continue

            # Collapse 3+ consecutive identical words to 2 (prevents TTS looping)
            text = _dedup_repeated_words(text)

            try:
                # Build engine-specific kwargs
                synth_kwargs = self._build_synthesis_kwargs(
                    tts_engine_name, seg, prompt, use_multi, sample_rate
                )

                # Piper needs special handling for length_scale
                if tts_engine_name == 'piper':
                    window_ms = synth_kwargs.pop('_window_ms', 0)
                    audio, sr = engine.synthesize(text, **synth_kwargs)
                    first_dur_ms = int(len(audio) / sample_rate * 1000)
                    if first_dur_ms > window_ms and window_ms > 0:
                        ls = max(window_ms / first_dur_ms, 0.5)
                        logger.debug(
                            "Seg %d: %dms TTS > %dms window, re-gen with length_scale=%.2f",
                            seg.index, first_dur_ms, window_ms, ls,
                        )
                        audio, sr = engine.synthesize(text, length_scale=ls, **{k: v for k, v in synth_kwargs.items() if k != 'language'})
                    wav = audio
                    sample_rate = sr
                else:
                    # All other engines: synthesize normally (return tuple via protocol)
                    audio, sr = engine.synthesize(text, **synth_kwargs)  # type: ignore[assignment]
                    wav = audio
                    sample_rate = sr

            except Exception as e:
                logger.warning("TTS failed for segment %d, skipping: %s", seg.index, e)
                continue

            # Post-process TTS for naturalness (skip cloud/emotion-native engines)
            if self.config.post_process_tts and tts_engine_name not in ('elevenlabs', 'indextts', 'cosyvoice', 'orpheus'):
                try:
                    from audiosmith.tts_postprocessor import (
                        PostProcessConfig, TTSPostProcessor)
                    preset = ENGINE_PP_PRESETS.get(tts_engine_name, {}).copy()
                    lang_overrides = LANGUAGE_PP_OVERRIDES.get(self.config.target_language, {})
                    preset.update(lang_overrides)
                    preset['global_intensity'] = self.config.post_process_intensity
                    pp_config = PostProcessConfig(**preset)
                    pp = TTSPostProcessor(pp_config)
                    wav = pp.process(
                        wav, sample_rate,
                        text=text,
                        emotion=seg.metadata.get('emotion'),
                        language=self.config.target_language,
                    )
                except Exception as e:
                    logger.warning("Post-processing failed for seg %d, using raw: %s", seg.index, e)

            wav_path = tts_dir / f'seg_{seg.index:04d}.wav'
            sf.write(str(wav_path), wav, sample_rate)
            seg.tts_audio_path = wav_path
            info = sf.info(str(wav_path))
            seg.tts_duration_ms = int(info.duration * 1000)

        engine.cleanup() if hasattr(engine, 'cleanup') else None
        if hasattr(engine, 'unload'):
            engine.unload()
        return segments

    def _init_chatterbox_engine(self, segments: List[DubbingSegment]) -> Tuple:
        """Initialize Chatterbox TTS engine (single or multi-voice)."""
        has_speakers = any(s.speaker_id for s in segments)
        has_emotion = any(s.metadata.get('emotion') for s in segments)
        use_multi = has_speakers or has_emotion

        if use_multi:
            from audiosmith.multi_voice_tts import MultiVoiceTTS
            engine = MultiVoiceTTS(
                device=self.config.whisper_device,
                language=self.config.target_language,
                default_exaggeration=self.config.chatterbox_exaggeration,
                default_cfg_weight=self.config.chatterbox_cfg_weight,
            )
            if self.config.audio_prompt_path:
                engine.set_default_voice(str(self.config.audio_prompt_path))
                voice_dir = self.config.audio_prompt_path.parent
                speaker_ids = list({s.speaker_id for s in segments if s.speaker_id})
                engine.auto_assign_voices(speaker_ids, voice_dir)
            engine.load_model()
        else:
            from audiosmith.tts import ChatterboxTTS
            engine = ChatterboxTTS(device=self.config.whisper_device)  # type: ignore[assignment]
            engine.load_model()

        return engine, engine.sample_rate, use_multi

    def _init_elevenlabs_engine(self) -> Tuple:
        """Initialize ElevenLabs TTS engine."""
        from audiosmith.elevenlabs_tts import ElevenLabsTTS
        engine = ElevenLabsTTS(
            model_id=self.config.elevenlabs_model,
            voice_id=self.config.elevenlabs_voice_id,
            voice_name=self.config.elevenlabs_voice_name,
        )
        if self.config.audio_prompt_path:
            engine.create_voice_clone(
                voice_name='clone',
                audio_files=[str(self.config.audio_prompt_path)],
            )
        return engine, engine.sample_rate

    def _init_qwen3_engine(self) -> Tuple:
        """Initialize Qwen3 TTS engine."""
        from audiosmith.qwen3_tts import Qwen3TTS
        engine = Qwen3TTS(device=self.config.whisper_device)
        if self.config.audio_prompt_path:
            engine.load_model('base')
            engine.create_voice_clone(
                voice_name='clone',
                ref_audio=str(self.config.audio_prompt_path),
            )
        else:
            engine.load_model('custom_voice')
        return engine, engine.sample_rate

    def _init_piper_engine(self) -> Tuple:
        """Initialize Piper TTS engine."""
        from audiosmith.exceptions import TTSError
        from audiosmith.piper_tts import PiperTTS

        voice = self._PIPER_VOICES.get(
            self.config.target_language, 'en_US-lessac-medium',
        )
        voice_dir = Path.home() / '.local' / 'share' / 'piper-voices'
        model_path = voice_dir / f'{voice}.onnx'

        if not model_path.exists():
            available = list(voice_dir.glob('*.onnx')) if voice_dir.exists() else []
            if available:
                model_path = available[0]
                logger.warning(
                    "Piper voice '%s' not found, falling back to '%s'",
                    voice, model_path.stem,
                )
            else:
                raise TTSError(f"No Piper voice models found in {voice_dir}")

        logger.info("Using Piper voice: %s", model_path.stem)
        engine = PiperTTS(model_path=model_path)
        return engine, engine.sample_rate

    def _init_f5_engine(self) -> Tuple:
        """Initialize F5-TTS engine (local, flow-matching)."""
        from audiosmith.f5_tts import F5TTS
        engine = F5TTS(
            model_name=self.config.f5_model,
            device=self.config.whisper_device,
            checkpoint_path=str(self.config.f5_checkpoint) if self.config.f5_checkpoint else None,
        )
        if self.config.audio_prompt_path:
            engine.clone_voice(
                'clone',
                audio_path_or_array=str(self.config.audio_prompt_path),
                ref_text=self.config.f5_ref_text,
            )
        return engine, engine.sample_rate

    def _init_fish_engine(self) -> Tuple:
        """Initialize Fish Speech TTS (cloud API)."""
        from audiosmith.fish_speech_tts import FishSpeechTTS
        engine = FishSpeechTTS()
        if self.config.audio_prompt_path:
            engine.create_voice_clone(
                voice_name='clone',
                ref_audio=str(self.config.audio_prompt_path),
            )
        return engine, engine.sample_rate

    def _init_indextts_engine(self) -> Tuple:
        """Initialize IndexTTS-2 engine (local, emotion-aware EN/ZH)."""
        from audiosmith.indextts_tts import IndexTTS2TTS
        engine = IndexTTS2TTS(
            model_variant=self.config.indextts_model,
            device=self.config.whisper_device,
            emo_alpha=self.config.indextts_emo_alpha,
        )
        if self.config.audio_prompt_path:
            engine.create_voice_clone('clone', ref_audio=self.config.audio_prompt_path)
        return engine, engine.sample_rate

    def _init_cosyvoice_engine(self) -> Tuple:
        """Initialize CosyVoice2 engine (local, highest MOS)."""
        from audiosmith.cosyvoice_tts import CosyVoice2TTS
        engine = CosyVoice2TTS(
            model_dir=self.config.cosyvoice_model_dir,
            device=self.config.whisper_device,
        )
        if self.config.audio_prompt_path:
            engine.create_voice_clone('clone', ref_audio=self.config.audio_prompt_path)
        return engine, engine.sample_rate

    def _init_orpheus_engine(self) -> Tuple:
        """Initialize Orpheus TTS engine (local, expressive)."""
        from audiosmith.orpheus_tts import OrpheusTTS
        engine = OrpheusTTS(
            voice=self.config.orpheus_voice,
            temperature=self.config.orpheus_temperature,
        )
        if self.config.audio_prompt_path:
            engine.create_voice_clone('clone', ref_audio=self.config.audio_prompt_path)
        return engine, engine.sample_rate

    def _create_engine_with_factory(self, engine_name: str) -> Any:
        """Create an engine instance with the factory, handling voice cloning."""
        from audiosmith.tts_protocol import get_engine

        engine = get_engine(engine_name)

        # Handle voice cloning for engines that support it
        if self.config.audio_prompt_path:
            if engine_name == 'qwen3':
                engine.load_model('base')  # type: ignore[call-arg]
                engine.create_voice_clone(  # type: ignore[attr-defined]
                    voice_name='clone',
                    ref_audio=str(self.config.audio_prompt_path),
                )
            elif engine_name == 'f5':
                engine.clone_voice(  # type: ignore[attr-defined]
                    'clone',
                    audio_path_or_array=str(self.config.audio_prompt_path),
                    ref_text=self.config.f5_ref_text,
                )
            elif engine_name in ('fish', 'indextts', 'cosyvoice', 'orpheus'):
                engine.create_voice_clone(  # type: ignore[attr-defined]
                    'clone', ref_audio=self.config.audio_prompt_path
                )
            elif engine_name == 'elevenlabs':
                engine.create_voice_clone(  # type: ignore[attr-defined]
                    voice_name='clone',
                    audio_files=[str(self.config.audio_prompt_path)],
                )

        return engine

    def _build_synthesis_kwargs(
        self, engine_name: str, seg: DubbingSegment, prompt: Optional[str],
        use_multi: bool, sample_rate: int
    ) -> Dict[str, Any]:
        """Build engine-specific synthesis kwargs for a segment.

        All engines accept these kwargs but only use what applies to them
        (the protocol allows **kwargs, so extra params are safely ignored).
        """
        from audiosmith.pipeline.helpers import _emotion_to_tts_params

        kwargs: Dict[str, Any] = {}

        # Emotion/style parameters (most engines support)
        emo_data = seg.metadata.get('emotion')
        if emo_data:
            if engine_name == 'elevenlabs':
                # ElevenLabs uses style (0.0 = neutral, 1.0 = expressive)
                kwargs['style'] = EMOTION_STYLE_MAP.get(
                    emo_data.get('primary', 'neutral'), 0.0
                )
            elif engine_name == 'fish':
                # Fish Speech uses emotion name
                kwargs['emotion'] = emo_data.get('primary')
            elif engine_name == 'orpheus':
                # Orpheus uses emotion name
                kwargs['emotion'] = emo_data.get('primary')

        # Voice parameters (cloning or preset)
        if engine_name == 'qwen3':
            kwargs['voice'] = 'clone' if prompt else 'Ryan'
        elif engine_name in ('fish', 'f5', 'indextts', 'cosyvoice'):
            kwargs['voice'] = 'clone' if prompt else None
        elif engine_name == 'orpheus':
            kwargs['voice'] = 'clone' if prompt else self.config.orpheus_voice

        # Language (most engines support it)
        if engine_name != 'chatterbox':
            kwargs['language'] = self.config.target_language

        # Engine-specific parameters
        if engine_name == 'indextts':
            kwargs['emotion_prompt'] = self.config.indextts_emotion_prompt
            if seg.duration_ms > 0:
                kwargs['target_duration_ms'] = seg.duration_ms
        elif engine_name == 'cosyvoice':
            kwargs['instruct'] = self.config.cosyvoice_instruct
        elif engine_name == 'f5':
            kwargs['speed'] = self.config.f5_speed
        elif engine_name == 'piper':
            # Special: Piper needs length_scale to fit the segment window
            window_ms = int((seg.end_time - seg.start_time) * 1000)
            if window_ms > 0:
                # Will be handled in calling code after first synthesis
                kwargs['_window_ms'] = window_ms
        elif engine_name == 'chatterbox' and use_multi:
            # Multi-voice Chatterbox uses speaker_id and emotion_params
            kwargs['speaker_id'] = seg.speaker_id
            if emo_data:
                kwargs['emotion_params'] = _emotion_to_tts_params(
                    emo_data['primary'], emo_data.get('intensity', 0.5),
                )
        elif engine_name == 'chatterbox':
            # Single-voice Chatterbox uses audio prompt
            kwargs['audio_prompt_path'] = prompt
            kwargs['exaggeration'] = self.config.chatterbox_exaggeration
            kwargs['cfg_weight'] = self.config.chatterbox_cfg_weight

        return kwargs
