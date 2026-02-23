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
        from audiosmith.tts import ChatterboxTTS

        tts_dir = Path(self.config.output_dir) / 'tts_segments'
        tts_dir.mkdir(parents=True, exist_ok=True)

        engine = ChatterboxTTS()
        engine.load_model()

        prompt = str(self.config.audio_prompt_path) if self.config.audio_prompt_path else None

        for seg in segments:
            text = seg.translated_text or seg.original_text
            if not text.strip():
                continue
            wav = engine.synthesize(
                text,
                language=self.config.target_language,
                audio_prompt_path=prompt,
                exaggeration=self.config.chatterbox_exaggeration,
                cfg_weight=self.config.chatterbox_cfg_weight,
            )
            wav_path = tts_dir / f'seg_{seg.index:04d}.wav'
            sf.write(str(wav_path), wav, engine.sample_rate)
            seg.tts_audio_path = wav_path
            info = sf.info(str(wav_path))
            seg.tts_duration_ms = int(info.duration * 1000)

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
            if d.get('tts_audio_path'):
                seg.tts_audio_path = Path(d['tts_audio_path'])
            seg.tts_duration_ms = d.get('tts_duration_ms')
            segments.append(seg)
        return segments

    @staticmethod
    def _write_srt(segments: List[DubbingSegment], path: Path) -> None:
        from audiosmith.srt import SRTEntry, write_srt, seconds_to_timestamp

        entries = []
        for i, seg in enumerate(segments, 1):
            text = seg.translated_text or seg.original_text
            entries.append(SRTEntry(
                index=i,
                start_time=seconds_to_timestamp(seg.start_time),
                end_time=seconds_to_timestamp(seg.end_time),
                text=text,
            ))
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

            result.total_segments = len(segments)

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
