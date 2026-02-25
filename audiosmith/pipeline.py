"""6-step dubbing pipeline with JSON checkpoint resume."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any

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
    # Step 4: Generate TTS
    # ------------------------------------------------------------------
    def _generate_tts(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        import soundfile as sf

        tts_dir = Path(self.config.output_dir) / 'tts_segments'
        tts_dir.mkdir(parents=True, exist_ok=True)

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

        prompt = str(self.config.audio_prompt_path) if self.config.audio_prompt_path else None

        for seg in segments:
            text = seg.translated_text or seg.original_text
            if not text.strip():
                continue

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

            wav_path = tts_dir / f'seg_{seg.index:04d}.wav'
            sf.write(str(wav_path), wav, engine.sample_rate)
            seg.tts_audio_path = wav_path
            info = sf.info(str(wav_path))
            seg.tts_duration_ms = int(info.duration * 1000)

        if use_multi:
            engine.unload()
        else:
            engine.cleanup()
        return segments

    # ------------------------------------------------------------------
    # Step 5: Mix audio
    # ------------------------------------------------------------------
    def _mix_audio(self, segments: List[DubbingSegment], total_duration: float) -> Path:
        from audiosmith.mixer import AudioMixer

        mixer = AudioMixer(self.config)
        scheduled = mixer.schedule(segments)
        out_path = Path(self.config.output_dir) / 'dubbed_audio.wav'
        mixer.render_to_file(scheduled, total_duration, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Step 6: Encode video
    # ------------------------------------------------------------------
    def _encode_video(
        self, video_path: Path, dubbed_audio: Path, segments: List[DubbingSegment],
    ) -> Path:
        from audiosmith.ffmpeg import encode_video

        srt_path = None
        if self.config.burn_subtitles:
            srt_path = Path(self.config.output_dir) / 'subtitles.srt'
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
            step = DubbingStep.MIX_AUDIO.value
            if self.state.is_step_done(step) and self.state.dubbed_audio_path:
                dubbed_audio = Path(self.state.dubbed_audio_path)
                logger.info("Skipping %s (cached)", step)
            else:
                ts = time.time()
                dubbed_audio = self._mix_audio(segments, total_duration)
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
                output_video = self._encode_video(video_path, dubbed_audio, segments)
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
