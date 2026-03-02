"""6-step dubbing pipeline with JSON checkpoint resume."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from audiosmith.models import (
    DubbingConfig, DubbingSegment, DubbingStep, DubbingResult, PipelineState,
)
from audiosmith.exceptions import DubbingError
from audiosmith.error_codes import ErrorCode

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = '.checkpoint.json'

# Emotion → Chatterbox TTS parameter offsets (exaggeration, cfg_weight)
_EMOTION_TTS_MAP: Dict[str, Dict[str, float]] = {
    'happy': {'exaggeration': 0.7, 'cfg_weight': 0.5},
    'sad': {'exaggeration': 0.3, 'cfg_weight': 0.4},
    'angry': {'exaggeration': 0.9, 'cfg_weight': 0.7},
    'fearful': {'exaggeration': 0.6, 'cfg_weight': 0.6},
    'surprised': {'exaggeration': 0.8, 'cfg_weight': 0.5},
    'whisper': {'exaggeration': 0.2, 'cfg_weight': 0.3},
    'sarcastic': {'exaggeration': 0.6, 'cfg_weight': 0.5},
    'tender': {'exaggeration': 0.3, 'cfg_weight': 0.4},
    'excited': {'exaggeration': 0.8, 'cfg_weight': 0.6},
    'determined': {'exaggeration': 0.7, 'cfg_weight': 0.6},
}


# Per-engine post-processing presets (calibrated to match ElevenLabs quality)
_ENGINE_PP_PRESETS: Dict[str, Dict] = {
    'piper': dict(
        enable_silence=True, enable_dynamics=True, enable_breath=True,
        enable_warmth=False, enable_spectral_matching=True,
        enable_micro_dynamics=True, enable_normalize=True,
        target_rms_adaptive=True, spectral_intensity=0.8,
    ),
    'chatterbox': dict(
        enable_silence=True, enable_dynamics=True, enable_breath=True,
        enable_warmth=True, enable_spectral_matching=True,
        enable_micro_dynamics=True, spectral_intensity=0.6,
    ),
    'fish': dict(
        enable_silence=False, enable_dynamics=True, enable_breath=True,
        enable_warmth=False, enable_spectral_matching=True,
        enable_micro_dynamics=False, enable_normalize=True,
        enable_silence_trim=True, max_silence_ms=100,
        target_rms_adaptive=True, spectral_intensity=0.5,
    ),
    'qwen3': dict(
        enable_silence=True, enable_dynamics=True, enable_breath=True,
        enable_warmth=True, enable_spectral_matching=True,
        enable_micro_dynamics=True, spectral_intensity=0.5,
    ),
    'indextts': dict(
        enable_silence=False, enable_dynamics=False, enable_breath=False,
        enable_warmth=False, enable_spectral_matching=False,
        enable_micro_dynamics=False, enable_normalize=False,
    ),
    'cosyvoice': dict(
        enable_silence=False, enable_dynamics=False, enable_breath=False,
        enable_warmth=False, enable_spectral_matching=False,
        enable_micro_dynamics=False, enable_normalize=True,
    ),
    'orpheus': dict(
        enable_silence=False, enable_dynamics=False, enable_breath=False,
        enable_warmth=False, enable_spectral_matching=False,
        enable_micro_dynamics=False, enable_normalize=True,
    ),
}


# Per-language post-processing overrides (stronger correction for non-English)
_LANGUAGE_PP_OVERRIDES: Dict[str, Dict[str, Any]] = {
    'pl': {
        'spectral_intensity': 0.3,
        'enable_spectral_matching': True,
        'enable_dynamics': True,
        'enable_breath': False,
        'enable_normalize': True,
        'target_rms_adaptive': False,
        'target_rms': 0.13,
    },
}


def _emotion_to_tts_params(emotion: str, intensity: float = 0.5) -> Dict[str, float]:
    """Convert emotion label + intensity to Chatterbox TTS parameters."""
    params = _EMOTION_TTS_MAP.get(emotion, {'exaggeration': 0.5, 'cfg_weight': 0.5})
    # Scale towards defaults at low intensity, towards full values at high intensity
    return {
        'exaggeration': 0.5 + (params['exaggeration'] - 0.5) * intensity,
        'cfg_weight': 0.5 + (params['cfg_weight'] - 0.5) * intensity,
    }


class DubbingPipeline:
    """Orchestrates the 6-step dubbing pipeline with optional resume."""

    def __init__(self, config: DubbingConfig):
        self.config = config
        self.checkpoint_path = Path(config.output_dir) / CHECKPOINT_FILE
        self.state = PipelineState()
        if config.resume and self.checkpoint_path.exists():
            self.state = PipelineState.load(self.checkpoint_path)
            logger.info("Resumed from checkpoint: %s", self.state.completed_steps)

    # ------------------------------------------------------------------
    # Step 1: Extract audio
    # ------------------------------------------------------------------
    def _extract_audio(self, video_path: Path) -> Path:
        from audiosmith.ffmpeg import extract_audio

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / 'audio_16k_mono.wav'
        extract_audio(video_path, audio_path)
        return audio_path

    # ------------------------------------------------------------------
    # Step 2: Transcribe
    # ------------------------------------------------------------------
    def _transcribe(self, audio_path: Path) -> List[DubbingSegment]:
        from audiosmith.transcribe import Transcriber

        lang = None if self.config.source_language == 'auto' else self.config.source_language
        t = Transcriber(
            model=self.config.whisper_model,
            compute_type=self.config.whisper_compute_type,
            device=self.config.whisper_device,
        )
        raw_segments = t.transcribe(audio_path, language=lang)
        t.unload()

        segments = []
        for i, seg in enumerate(raw_segments):
            segments.append(DubbingSegment(
                index=i,
                start_time=seg['start'],
                end_time=seg['end'],
                original_text=seg['text'],
            ))
        return segments

    # ------------------------------------------------------------------
    # Step 3: Translate
    # ------------------------------------------------------------------
    def _translate(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        from audiosmith.translate import translate

        src = self.config.source_language
        tgt = self.config.target_language
        for seg in segments:
            seg.translated_text = translate(seg.original_text, src, tgt)
        return segments

    # ------------------------------------------------------------------
    # Step 3.5: Merge short adjacent segments
    # ------------------------------------------------------------------
    def _merge_segments(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        """Merge short adjacent segments from the same speaker.

        Combines segments that are close together (gap < merge_max_gap_ms)
        and would fit within merge_max_duration_s when combined.  This gives
        the TTS engine longer phrases with better prosody and bigger time
        windows that need less speedup.
        """
        max_gap = self.config.merge_max_gap_ms / 1000.0
        max_dur = self.config.merge_max_duration_s
        # Chatterbox generates longer audio per word — use tighter limits
        engine = getattr(self.config, 'tts_engine', 'piper')
        if engine in ('chatterbox', 'auto') and self.config.target_language not in self._QWEN3_LANGS:
            max_word_cap = 12
            max_dur = min(max_dur, 4.0)
        else:
            max_word_cap = 30
        merged: List[DubbingSegment] = []

        for seg in segments:
            if not merged:
                merged.append(seg)
                continue

            prev = merged[-1]
            gap = seg.start_time - prev.end_time
            combined_dur = seg.end_time - prev.start_time
            same_speaker = (prev.speaker_id or '') == (seg.speaker_id or '')

            prev_words = len(prev.translated_text.split()) if prev.translated_text else len(prev.original_text.split())
            seg_words = len(seg.translated_text.split()) if seg.translated_text else len(seg.original_text.split())
            combined_words = prev_words + seg_words

            if same_speaker and gap < max_gap and combined_dur <= max_dur and combined_words <= max_word_cap:
                # Merge into previous: extend end time, join text
                prev.end_time = seg.end_time
                orig_sep = ' ' if prev.original_text.rstrip()[-1:] in '.!?' else ' '
                prev.original_text = prev.original_text.rstrip() + orig_sep + seg.original_text.strip()
                if prev.translated_text and seg.translated_text:
                    trans_sep = ' ' if prev.translated_text.rstrip()[-1:] in '.!?' else ' '
                    prev.translated_text = prev.translated_text.rstrip() + trans_sep + seg.translated_text.strip()
                # Keep emotion/metadata from first segment
            else:
                merged.append(seg)

        # Split any segments that are still too long (from transcription)
        final: List[DubbingSegment] = []
        for seg in merged:
            text = seg.translated_text or seg.original_text
            if len(text.split()) > max_word_cap:
                final.extend(self._split_long_segment(seg, max_words=max_word_cap))
            else:
                final.append(seg)

        # Re-index
        for i, seg in enumerate(final):
            seg.index = i

        return final

    @staticmethod
    def _split_long_segment(seg: DubbingSegment, max_words: int = 25) -> List[DubbingSegment]:
        """Split a long segment at sentence boundaries, distributing time proportionally."""
        import re
        text = seg.translated_text or seg.original_text
        orig = seg.original_text

        # Split on sentence-ending punctuation
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) <= 1:
            return [seg]

        # Group sentences into chunks of ~max_words
        chunks: List[str] = []
        current: List[str] = []
        current_words = 0
        for sent in sentences:
            w = len(sent.split())
            if current_words + w > max_words and current:
                chunks.append(' '.join(current))
                current = [sent]
                current_words = w
            else:
                current.append(sent)
                current_words += w
        if current:
            chunks.append(' '.join(current))

        # Distribute time proportionally by character count
        total_chars = sum(len(c) for c in chunks)
        total_dur = seg.end_time - seg.start_time
        result = []
        t = seg.start_time
        for chunk in chunks:
            frac = len(chunk) / total_chars if total_chars else 1.0 / len(chunks)
            dur = total_dur * frac
            result.append(DubbingSegment(
                index=0,
                start_time=t,
                end_time=t + dur,
                original_text=orig,
                translated_text=chunk,
                speaker_id=seg.speaker_id,
                metadata=dict(seg.metadata),
            ))
            t += dur

        return result

    # ------------------------------------------------------------------
    # Step 4: Generate TTS
    # ------------------------------------------------------------------

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
        import soundfile as sf

        tts_dir = Path(self.config.output_dir) / 'tts_segments'
        tts_dir.mkdir(parents=True, exist_ok=True)

        tts_engine_name = getattr(self.config, 'tts_engine', 'chatterbox')
        if tts_engine_name == 'auto':
            tts_engine_name = self._resolve_engine()
        prompt = str(self.config.audio_prompt_path) if self.config.audio_prompt_path else None

        if tts_engine_name == 'elevenlabs':
            engine, sample_rate = self._init_elevenlabs_engine()
        elif tts_engine_name == 'qwen3':
            engine, sample_rate = self._init_qwen3_engine()
        elif tts_engine_name == 'fish':
            engine, sample_rate = self._init_fish_engine()
        elif tts_engine_name == 'indextts':
            engine, sample_rate = self._init_indextts_engine()
        elif tts_engine_name == 'cosyvoice':
            engine, sample_rate = self._init_cosyvoice_engine()
        elif tts_engine_name == 'orpheus':
            engine, sample_rate = self._init_orpheus_engine()
        elif tts_engine_name == 'piper':
            engine, sample_rate = self._init_piper_engine()
        else:
            engine, sample_rate, use_multi = self._init_chatterbox_engine(segments)

        for seg in segments:
            text = seg.translated_text or seg.original_text
            if not text.strip():
                continue

            # Collapse 3+ consecutive identical words to 2 (prevents TTS looping)
            text = self._dedup_repeated_words(text)

            try:
                if tts_engine_name == 'elevenlabs':
                    audio, sr = engine.synthesize(text)
                    wav = audio
                    sample_rate = sr
                elif tts_engine_name == 'qwen3':
                    voice = 'clone' if prompt else 'Ryan'
                    audio, sr = engine.synthesize(text, voice=voice)
                    wav = audio
                    sample_rate = sr
                elif tts_engine_name == 'fish':
                    voice = 'clone' if prompt else None
                    emotion_name = None
                    emo_data = seg.metadata.get('emotion')
                    if emo_data:
                        emotion_name = emo_data.get('primary')
                    audio, sr = engine.synthesize(
                        text, voice=voice, language=self.config.target_language,
                        emotion=emotion_name,
                    )
                    wav = audio
                    sample_rate = sr
                elif tts_engine_name == 'indextts':
                    voice = 'clone' if prompt else None
                    emotion_prompt = self.config.indextts_emotion_prompt
                    target_dur = seg.duration_ms if seg.duration_ms > 0 else None
                    audio, sr = engine.synthesize(
                        text, voice=voice, language=self.config.target_language,
                        emotion_prompt=emotion_prompt,
                        target_duration_ms=target_dur,
                    )
                    wav = audio
                    sample_rate = sr
                elif tts_engine_name == 'cosyvoice':
                    voice = 'clone' if prompt else None
                    audio, sr = engine.synthesize(
                        text, voice=voice, language=self.config.target_language,
                        instruct=self.config.cosyvoice_instruct,
                    )
                    wav = audio
                    sample_rate = sr
                elif tts_engine_name == 'orpheus':
                    voice = 'clone' if prompt else self.config.orpheus_voice
                    emotion_name = None
                    emo_data = seg.metadata.get('emotion')
                    if emo_data:
                        emotion_name = emo_data.get('primary')
                    audio, sr = engine.synthesize(
                        text, voice=voice, language=self.config.target_language,
                        emotion=emotion_name,
                    )
                    wav = audio
                    sample_rate = sr
                elif tts_engine_name == 'piper':
                    # Calculate length_scale to fit window without post-hoc time_stretch
                    window_ms = int((seg.end_time - seg.start_time) * 1000)
                    wav = engine.synthesize(text)
                    first_dur_ms = int(len(wav) / sample_rate * 1000)
                    if first_dur_ms > window_ms and window_ms > 0:
                        ls = max(window_ms / first_dur_ms, 0.5)
                        logger.debug(
                            "Seg %d: %dms TTS > %dms window, re-gen with length_scale=%.2f",
                            seg.index, first_dur_ms, window_ms, ls,
                        )
                        wav = engine.synthesize(text, length_scale=ls)
                else:  # chatterbox
                    if use_multi:
                        emotion_params = None
                        emo_data = seg.metadata.get('emotion')
                        if emo_data:
                            emotion_params = _emotion_to_tts_params(
                                emo_data['primary'], emo_data.get('intensity', 0.5),
                            )
                        wav = engine.synthesize(text, speaker_id=seg.speaker_id, emotion_params=emotion_params)
                    else:
                        wav = engine.synthesize(
                            text, language=self.config.target_language,
                            audio_prompt_path=prompt,
                            exaggeration=self.config.chatterbox_exaggeration,
                            cfg_weight=self.config.chatterbox_cfg_weight,
                        )
            except Exception as e:
                logger.warning("TTS failed for segment %d, skipping: %s", seg.index, e)
                continue

            # Post-process TTS for naturalness (skip cloud/emotion-native engines)
            if self.config.post_process_tts and tts_engine_name not in ('elevenlabs', 'indextts', 'cosyvoice', 'orpheus'):
                try:
                    from audiosmith.tts_postprocessor import TTSPostProcessor, PostProcessConfig
                    preset = _ENGINE_PP_PRESETS.get(tts_engine_name, {}).copy()
                    lang_overrides = _LANGUAGE_PP_OVERRIDES.get(self.config.target_language, {})
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

    def _init_chatterbox_engine(self, segments):
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
            engine = ChatterboxTTS(device=self.config.whisper_device)
            engine.load_model()

        return engine, engine.sample_rate, use_multi

    def _init_elevenlabs_engine(self):
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

    def _init_qwen3_engine(self):
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

    _PIPER_VOICES = {
        'en': 'en_US-lessac-medium',
        'pl': 'pl_PL-darkman-medium',
        'de': 'de_DE-thorsten-medium',
    }

    def _init_piper_engine(self):
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
                from audiosmith.exceptions import TTSError
                raise TTSError(f"No Piper voice models found in {voice_dir}")

        logger.info("Using Piper voice: %s", model_path.stem)
        engine = PiperTTS(model_path=model_path)
        return engine, engine.sample_rate

    def _init_fish_engine(self):
        """Initialize Fish Speech TTS (cloud API)."""
        from audiosmith.fish_speech_tts import FishSpeechTTS
        engine = FishSpeechTTS()
        if self.config.audio_prompt_path:
            engine.create_voice_clone(
                voice_name='clone',
                ref_audio=str(self.config.audio_prompt_path),
            )
        return engine, engine.sample_rate

    def _init_indextts_engine(self):
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

    def _init_cosyvoice_engine(self):
        """Initialize CosyVoice2 engine (local, highest MOS)."""
        from audiosmith.cosyvoice_tts import CosyVoice2TTS
        engine = CosyVoice2TTS(
            model_dir=self.config.cosyvoice_model_dir,
            device=self.config.whisper_device,
        )
        if self.config.audio_prompt_path:
            engine.create_voice_clone('clone', ref_audio=self.config.audio_prompt_path)
        return engine, engine.sample_rate

    def _init_orpheus_engine(self):
        """Initialize Orpheus TTS engine (local, expressive)."""
        from audiosmith.orpheus_tts import OrpheusTTS
        engine = OrpheusTTS(
            voice=self.config.orpheus_voice,
            temperature=self.config.orpheus_temperature,
        )
        if self.config.audio_prompt_path:
            engine.create_voice_clone('clone', ref_audio=self.config.audio_prompt_path)
        return engine, engine.sample_rate

    # ------------------------------------------------------------------
    # Step 5: Mix audio
    # ------------------------------------------------------------------
    def _mix_audio(self, segments: List[DubbingSegment], total_duration: float):
        from audiosmith.mixer import AudioMixer

        mixer = AudioMixer(self.config)
        # Neural TTS (Chatterbox): prefer truncation over time_stretch distortion
        engine = getattr(self.config, 'tts_engine', 'piper')
        if engine in ('chatterbox', 'auto') and self.config.target_language not in self._QWEN3_LANGS:
            mixer.max_speedup = min(mixer.max_speedup, 1.3)
        scheduled = mixer.schedule(segments)
        out_path = Path(self.config.output_dir) / 'dubbed_audio.wav'
        mixer.render_to_file(scheduled, total_duration, out_path)
        return out_path, scheduled

    # ------------------------------------------------------------------
    # Step 6: Encode video
    # ------------------------------------------------------------------
    def _encode_video(
        self, video_path: Path, dubbed_audio: Path, segments: List[DubbingSegment],
        scheduled: Optional[List] = None,
    ) -> Path:
        from audiosmith.ffmpeg import encode_video

        srt_path = None
        if self.config.burn_subtitles:
            srt_path = Path(self.config.output_dir) / 'subtitles.srt'
            if scheduled:
                self._write_srt_from_schedule(scheduled, srt_path)
            else:
                self._write_srt(segments, srt_path)

        out_path = Path(self.config.output_dir) / f'{video_path.stem}_dubbed.mp4'
        encode_video(video_path, dubbed_audio, out_path, subtitle_path=srt_path)
        return out_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _segments_to_dicts(segments: List[DubbingSegment]) -> List[Dict[str, Any]]:
        result = []
        for s in segments:
            result.append({
                'index': s.index,
                'start_time': s.start_time,
                'end_time': s.end_time,
                'original_text': s.original_text,
                'translated_text': s.translated_text,
                'speaker_id': s.speaker_id,
                'metadata': s.metadata,
                'tts_audio_path': str(s.tts_audio_path) if s.tts_audio_path else None,
                'tts_duration_ms': s.tts_duration_ms,
            })
        return result

    @staticmethod
    def _dicts_to_segments(dicts: List[Dict[str, Any]]) -> List[DubbingSegment]:
        segments = []
        for d in dicts:
            seg = DubbingSegment(
                index=d['index'],
                start_time=d['start_time'],
                end_time=d['end_time'],
                original_text=d['original_text'],
                translated_text=d.get('translated_text', ''),
            )
            seg.speaker_id = d.get('speaker_id')
            seg.metadata = d.get('metadata', {})
            if d.get('tts_audio_path'):
                seg.tts_audio_path = Path(d['tts_audio_path'])
            seg.tts_duration_ms = d.get('tts_duration_ms')
            segments.append(seg)
        return segments

    @staticmethod
    def _write_srt(segments: List[DubbingSegment], path: Path) -> None:
        from audiosmith.srt import write_srt
        from audiosmith.srt_formatter import SRTFormatter

        formatter = SRTFormatter()
        raw_segments = [
            {
                'text': seg.translated_text or seg.original_text,
                'start': seg.start_time,
                'end': seg.end_time,
                'words': [],
            }
            for seg in segments
        ]
        entries = formatter.format_segments(raw_segments)
        write_srt(entries, path)

    @staticmethod
    def _write_srt_from_schedule(scheduled: List, path: Path) -> None:
        """Write SRT using the mixer's actual placement times, not original segment times."""
        from audiosmith.srt import write_srt
        from audiosmith.srt_formatter import SRTFormatter

        formatter = SRTFormatter()
        raw_segments = [
            {
                'text': item.segment.translated_text or item.segment.original_text,
                'start': item.place_at_ms / 1000.0,
                'end': (item.place_at_ms + item.actual_duration_ms) / 1000.0,
                'words': [],
            }
            for item in scheduled
        ]
        entries = formatter.format_segments(raw_segments)
        write_srt(entries, path)

    @staticmethod
    def _dedup_repeated_words(text: str, max_repeats: int = 2) -> str:
        """Collapse runs of 3+ identical consecutive words to max_repeats."""
        words = text.split()
        if len(words) < 3:
            return text
        result = [words[0]]
        count = 1
        for w in words[1:]:
            if w.lower() == result[-1].lower():
                count += 1
                if count <= max_repeats:
                    result.append(w)
            else:
                result.append(w)
                count = 1
        return ' '.join(result)

    def _extract_speaker_voices(
        self,
        segments: List[DubbingSegment],
        audio_path: Path,
        voice_dir: Path,
    ) -> None:
        """Extract a representative voice sample per speaker for voice cloning."""
        import soundfile as sf

        voice_dir.mkdir(parents=True, exist_ok=True)
        speakers: Dict[str, List[DubbingSegment]] = {}
        for seg in segments:
            if seg.speaker_id and seg.is_speech:
                speakers.setdefault(seg.speaker_id, []).append(seg)

        if not speakers:
            return

        audio_data, sr = sf.read(str(audio_path))
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)

        first_voice_path = None
        for spk_id, spk_segs in speakers.items():
            # Pick the segment closest to 5 seconds (ideal for voice cloning)
            best = min(spk_segs, key=lambda s: abs((s.end_time - s.start_time) - 5.0))
            start_sample = int(best.start_time * sr)
            end_sample = min(int(best.end_time * sr), len(audio_data))
            clip = audio_data[start_sample:end_sample]
            if len(clip) < sr:  # skip clips under 1 second
                continue
            out_path = voice_dir / f'{spk_id}.wav'
            sf.write(str(out_path), clip, sr)
            if first_voice_path is None:
                first_voice_path = out_path
            logger.info(
                "Extracted voice sample for %s: %.1fs (%s)",
                spk_id, len(clip) / sr, out_path.name,
            )

        # Create a default.wav symlink/copy for fallback
        if first_voice_path:
            default_path = voice_dir / 'default.wav'
            if not default_path.exists():
                import shutil
                shutil.copy2(first_voice_path, default_path)

    def _save_checkpoint(self) -> None:
        self.state.save(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------
    def run(self, video_path: Path) -> DubbingResult:
        """Execute the full 6-step dubbing pipeline."""
        t0 = time.time()
        result = DubbingResult(success=False)
        video_path = Path(video_path)
        segments: List[DubbingSegment] = []
        step = ''

        try:
            # Step 1: Extract audio
            step = DubbingStep.EXTRACT_AUDIO.value
            if self.state.is_step_done(step) and self.state.audio_path:
                audio_path = Path(self.state.audio_path)
                logger.info("Skipping %s (cached)", step)
            else:
                ts = time.time()
                audio_path = self._extract_audio(video_path)
                result.step_times[step] = time.time() - ts
                self.state.audio_path = str(audio_path)
                self.state.mark_step_done(step)
                self._save_checkpoint()

            # Get total duration for later mixing
            from audiosmith.ffmpeg import probe_duration
            total_duration = probe_duration(video_path)
            self.state.duration = total_duration

            # Step 1.5: Isolate vocals (optional)
            step = DubbingStep.ISOLATE_VOCALS.value
            if self.config.isolate_vocals:
                if self.state.is_step_done(step):
                    logger.info("Skipping %s (cached)", step)
                else:
                    from audiosmith.vocal_isolator import VocalIsolator
                    ts = time.time()
                    vi = VocalIsolator(device=self.config.whisper_device)
                    paths = vi.isolate(audio_path, output_dir=Path(self.config.output_dir))
                    vi.unload()
                    audio_path = paths['vocals_path']
                    self.state.audio_path = str(audio_path)
                    result.step_times[step] = time.time() - ts
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            # Step 2: Transcribe
            step = DubbingStep.TRANSCRIBE.value
            if self.state.is_step_done(step) and self.state.segments:
                segments = self._dicts_to_segments(self.state.segments)
                logger.info("Skipping %s (cached: %d segments)", step, len(segments))
            else:
                ts = time.time()
                segments = self._transcribe(audio_path)
                result.step_times[step] = time.time() - ts
                self.state.segments = self._segments_to_dicts(segments)
                self.state.mark_step_done(step)
                self._save_checkpoint()

            # Step 2.5: Post-process transcription (optional, on by default)
            step = DubbingStep.POST_PROCESS.value
            if self.config.post_process:
                if self.state.is_step_done(step):
                    segments = self._dicts_to_segments(self.state.segments)
                    logger.info("Skipping %s (cached)", step)
                else:
                    from audiosmith.transcription_post_processor import TranscriptionPostProcessor
                    ts = time.time()
                    pp = TranscriptionPostProcessor()
                    segments = pp.process(segments)
                    result.step_times[step] = time.time() - ts
                    self.state.segments = self._segments_to_dicts(segments)
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            result.total_segments = len(segments)

            # Step 2.7: Diarize (optional)
            step = DubbingStep.DIARIZE.value
            if self.config.diarize:
                if self.state.is_step_done(step):
                    # Restore speaker_id from checkpoint
                    segments = self._dicts_to_segments(self.state.segments)
                    logger.info("Skipping %s (cached)", step)
                else:
                    from audiosmith.diarizer import Diarizer
                    ts = time.time()
                    d = Diarizer(device=self.config.whisper_device)
                    diar_segments = d.diarize(audio_path)
                    d.unload()
                    # Assign speaker_id to each DubbingSegment by timing overlap
                    trans_dicts = [
                        {'start': s.start_time, 'end': s.end_time}
                        for s in segments
                    ]
                    labeled = Diarizer.apply_to_transcription(trans_dicts, diar_segments)
                    for seg, lbl in zip(segments, labeled):
                        seg.speaker_id = lbl.get('speaker')
                    result.step_times[step] = time.time() - ts
                    self.state.segments = self._segments_to_dicts(segments)
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            # Extract per-speaker voice samples for voice cloning
            if self.config.diarize and not self.config.audio_prompt_path:
                voice_dir = Path(self.config.output_dir) / 'speaker_voices'
                if not voice_dir.exists() or not list(voice_dir.glob('*.wav')):
                    self._extract_speaker_voices(segments, audio_path, voice_dir)
                if list(voice_dir.glob('*.wav')):
                    self.config.audio_prompt_path = voice_dir / 'default.wav'

            # Step 2.7: Detect emotion (optional)
            step = DubbingStep.DETECT_EMOTION.value
            if self.config.detect_emotion:
                if self.state.is_step_done(step):
                    segments = self._dicts_to_segments(self.state.segments)
                    logger.info("Skipping %s (cached)", step)
                else:
                    from audiosmith.emotion import EmotionEngine
                    ts = time.time()
                    engine = EmotionEngine(use_classifier=False)
                    for seg in segments:
                        emo = engine.analyze(seg.original_text)
                        seg.metadata['emotion'] = {
                            'primary': emo.primary_emotion.value,
                            'confidence': emo.confidence,
                            'intensity': emo.intensity,
                        }
                    result.step_times[step] = time.time() - ts
                    self.state.segments = self._segments_to_dicts(segments)
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            # Step 3: Translate
            step = DubbingStep.TRANSLATE.value
            if self.state.is_step_done(step):
                segments = self._dicts_to_segments(self.state.segments)
                logger.info("Skipping %s (cached)", step)
            else:
                ts = time.time()
                segments = self._translate(segments)
                result.step_times[step] = time.time() - ts
                self.state.segments = self._segments_to_dicts(segments)
                self.state.mark_step_done(step)
                self._save_checkpoint()

            # Step 3.5: Merge short adjacent segments (on by default)
            step = DubbingStep.MERGE_SEGMENTS.value
            if self.config.merge_segments:
                if self.state.is_step_done(step):
                    segments = self._dicts_to_segments(self.state.segments)
                    logger.info("Skipping %s (cached)", step)
                else:
                    ts = time.time()
                    before = len(segments)
                    segments = self._merge_segments(segments)
                    result.step_times[step] = time.time() - ts
                    logger.info(
                        "Merged %d → %d segments (combined %d)",
                        before, len(segments), before - len(segments),
                    )
                    self.state.segments = self._segments_to_dicts(segments)
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            # Step 4: Generate TTS
            step = DubbingStep.GENERATE_TTS.value
            if self.state.is_step_done(step):
                segments = self._dicts_to_segments(self.state.segments)
                logger.info("Skipping %s (cached)", step)
            else:
                ts = time.time()
                segments = self._generate_tts(segments)
                result.step_times[step] = time.time() - ts
                self.state.segments = self._segments_to_dicts(segments)
                self.state.mark_step_done(step)
                self._save_checkpoint()

            result.segments_dubbed = sum(1 for s in segments if s.tts_audio_path)

            # Step 5: Mix audio
            scheduled = None
            step = DubbingStep.MIX_AUDIO.value
            if self.state.is_step_done(step) and self.state.dubbed_audio_path:
                dubbed_audio = Path(self.state.dubbed_audio_path)
                logger.info("Skipping %s (cached)", step)
            else:
                ts = time.time()
                dubbed_audio, scheduled = self._mix_audio(segments, total_duration)
                result.step_times[step] = time.time() - ts
                self.state.dubbed_audio_path = str(dubbed_audio)
                self.state.mark_step_done(step)
                self._save_checkpoint()

            result.dubbed_audio_path = dubbed_audio

            # Step 6: Encode video
            step = DubbingStep.ENCODE_VIDEO.value
            if self.state.is_step_done(step):
                logger.info("Skipping %s (cached)", step)
            else:
                ts = time.time()
                output_video = self._encode_video(
                    video_path, dubbed_audio, segments, scheduled=scheduled,
                )
                result.step_times[step] = time.time() - ts
                result.output_video_path = output_video
                self.state.mark_step_done(step)
                self._save_checkpoint()

            result.success = True
            result.total_time = time.time() - t0
            logger.info(
                "Pipeline complete: %d segments dubbed in %.1fs",
                result.segments_dubbed, result.total_time,
            )

        except DubbingError:
            raise
        except Exception as e:
            raise DubbingError(
                f"Pipeline failed at step '{step}': {e}",
                error_code=str(ErrorCode.DUBBING_PIPELINE_ERROR.value),
                original_error=e,
            ) from e

        return result
