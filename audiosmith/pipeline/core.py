"""Core DubbingPipeline orchestrator class."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from audiosmith.error_codes import ErrorCode
from audiosmith.exceptions import DubbingError
from audiosmith.models import (DubbingConfig, DubbingResult, DubbingSegment,
                               DubbingStep, PipelineState)
from audiosmith.pipeline.helpers import (
    _dicts_to_segments, _dedup_repeated_words, _segments_to_dicts,
    _write_srt, _write_srt_from_schedule,
)
from audiosmith.pipeline.tts_synthesis import TTSSynthesisMixin

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = '.checkpoint.json'


class DubbingPipeline(TTSSynthesisMixin):
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
    def _extract_audio(self, video_path: Path, extract_hq: bool = False) -> Path:
        from audiosmith.ffmpeg import extract_audio, extract_audio_hq

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / 'audio_16k_mono.wav'
        extract_audio(video_path, audio_path)

        # Extract high-quality audio for Demucs vocal isolation
        if extract_hq:
            hq_audio_path = out_dir / 'audio_hq.wav'
            extract_audio_hq(video_path, hq_audio_path)
            self.state.hq_audio_path = str(hq_audio_path)

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

    def _import_external_srt(
        self, srt_path: Path, segments: List[DubbingSegment],
    ) -> List[DubbingSegment]:
        """Match official SRT entries to transcription segments by time overlap.

        This replaces the ML-translated text with official human-verified subtitles,
        preserving the timing from the original transcription.
        """
        from audiosmith.srt import parse_srt_file, timestamp_to_seconds

        srt_entries = parse_srt_file(srt_path)
        srt_segments: list[dict[str, float | str]] = [
            {'start': timestamp_to_seconds(e.start_time),
             'end': timestamp_to_seconds(e.end_time),
             'text': e.text}
            for e in srt_entries
        ]

        matched_count = 0
        for seg in segments:
            # Find the SRT segment with maximum time overlap
            best_match: dict[str, float | str] | None = None
            best_overlap = 0.0
            for srt_seg in srt_segments:
                # Calculate overlap duration
                srt_start = float(srt_seg['start'])  # type: ignore[index, assignment]
                srt_end = float(srt_seg['end'])  # type: ignore[index]
                overlap_start = max(seg.start_time, srt_start)
                overlap_end = min(seg.end_time, srt_end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = srt_seg

            if best_match and best_overlap > 0.0:
                seg.translated_text = best_match['text']  # type: ignore[assignment]
                seg.metadata['translation_source'] = 'external_srt'
                matched_count += 1
            else:
                # Fallback: use original text as translation
                seg.translated_text = seg.original_text
                seg.metadata['translation_source'] = 'fallback_original'

        logger.info(
            "Imported external SRT: %d/%d segments matched",
            matched_count, len(segments),
        )
        return segments

    def _create_segments_from_srt(self, srt_path: Path) -> List[DubbingSegment]:
        """Create segments directly from SRT entries using SRT timing.

        This is the preferred mode when you have a high-quality official SRT.
        It creates segments that match the SRT timing exactly, avoiding
        the alignment issues that occur when Whisper segments are much longer
        than SRT entries.
        """
        from audiosmith.srt import parse_srt_file, timestamp_to_seconds

        srt_entries = parse_srt_file(srt_path)
        segments: List[DubbingSegment] = []

        for i, entry in enumerate(srt_entries):
            text = entry.text.strip()

            # Skip non-speech entries (sound effects, music)
            if text.startswith('[') and text.endswith(']'):
                continue

            # Skip empty entries
            if not text:
                continue

            start = timestamp_to_seconds(entry.start_time)
            end = timestamp_to_seconds(entry.end_time)

            # Skip very short entries (< 0.3s) - likely artifacts
            if end - start < 0.3:
                continue

            segments.append(DubbingSegment(
                index=len(segments),
                start_time=start,
                end_time=end,
                original_text='',  # No English source when using SRT timing
                translated_text=text,
                is_speech=True,
                metadata={'translation_source': 'external_srt_timing'},
            ))

        logger.info(
            "Created %d segments from SRT (skipped %d non-speech/empty)",
            len(segments), len(srt_entries) - len(segments),
        )
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
        # Engine-specific merge limits
        engine = getattr(self.config, 'tts_engine', 'piper')
        if engine in ('chatterbox', 'auto') and self.config.target_language not in self._QWEN3_LANGS:
            max_word_cap = 12
            max_dur = min(max_dur, 4.0)
        elif engine == 'fish':
            # Fish Speech S2-Pro hallucinates on short inputs (<8 words).
            # Use wider merge: more words per segment, larger gap tolerance.
            max_gap = max(max_gap, 1.5)
            max_dur = max(max_dur, 12.0)
            max_word_cap = 40
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
    # Step 5: Mix audio
    # ------------------------------------------------------------------
    def _mix_audio(self, segments: List[DubbingSegment], total_duration: float):
        from audiosmith.mixer import AudioMixer

        bg_path = None
        if self.state.background_audio_path:
            bg_path = Path(self.state.background_audio_path)
            if not bg_path.exists():
                logger.warning("Background audio not found: %s", bg_path)
                bg_path = None
        mixer = AudioMixer(self.config, background_path=bg_path)
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
                _write_srt_from_schedule(scheduled, srt_path)
            else:
                _write_srt(segments, srt_path)

        out_path = Path(self.config.output_dir) / f'{video_path.stem}_dubbed.mp4'
        encode_video(video_path, dubbed_audio, out_path, subtitle_path=srt_path)
        return out_path

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------
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
    # Backward-compatible wrapper methods for helpers
    # ------------------------------------------------------------------
    def _segments_to_dicts(self, segments: List[DubbingSegment]) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper for the helper function."""
        return _segments_to_dicts(segments)

    def _dicts_to_segments(self, dicts: List[Dict[str, Any]]) -> List[DubbingSegment]:
        """Backward-compatible wrapper for the helper function."""
        return _dicts_to_segments(dicts)

    def _write_srt(self, segments: List[DubbingSegment], path: Path) -> None:
        """Backward-compatible wrapper for the helper function."""
        return _write_srt(segments, path)

    def _write_srt_from_schedule(self, scheduled: List, path: Path) -> None:
        """Backward-compatible wrapper for the helper function."""
        return _write_srt_from_schedule(scheduled, path)

    def _dedup_repeated_words(self, text: str, max_repeats: int = 2) -> str:
        """Backward-compatible wrapper for the helper function."""
        return _dedup_repeated_words(text, max_repeats)

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
                audio_path = self._extract_audio(video_path, extract_hq=self.config.isolate_vocals)
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
                    # Use HQ audio (48kHz stereo) for Demucs, not 16kHz mono
                    hq_path = Path(self.state.hq_audio_path) if self.state.hq_audio_path else audio_path
                    paths = vi.isolate(
                        hq_path, output_dir=Path(self.config.output_dir),
                        mixing_sample_rate=self.config.dubbed_sample_rate,
                    )
                    vi.unload()
                    audio_path = paths['vocals_path']
                    self.state.audio_path = str(audio_path)
                    self.state.background_audio_path = str(paths['background_hq_path'])
                    result.step_times[step] = time.time() - ts
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            # Step 2: Transcribe
            step = DubbingStep.TRANSCRIBE.value
            if self.state.is_step_done(step) and self.state.segments:
                segments = _dicts_to_segments(self.state.segments)
                logger.info("Skipping %s (cached: %d segments)", step, len(segments))
            else:
                ts = time.time()
                segments = self._transcribe(audio_path)
                result.step_times[step] = time.time() - ts
                self.state.segments = _segments_to_dicts(segments)
                self.state.mark_step_done(step)
                self._save_checkpoint()

            # Step 2.5: Post-process transcription (optional, on by default)
            step = DubbingStep.POST_PROCESS.value
            if self.config.post_process:
                if self.state.is_step_done(step):
                    segments = _dicts_to_segments(self.state.segments)
                    logger.info("Skipping %s (cached)", step)
                else:
                    from audiosmith.transcription_post_processor import \
                        TranscriptionPostProcessor
                    ts = time.time()
                    pp = TranscriptionPostProcessor()
                    segments = pp.process(segments)
                    result.step_times[step] = time.time() - ts
                    self.state.segments = _segments_to_dicts(segments)
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            result.total_segments = len(segments)

            # Step 2.7: Diarize (optional)
            step = DubbingStep.DIARIZE.value
            if self.config.diarize:
                if self.state.is_step_done(step):
                    # Restore speaker_id from checkpoint
                    segments = _dicts_to_segments(self.state.segments)
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
                    self.state.segments = _segments_to_dicts(segments)
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
                    segments = _dicts_to_segments(self.state.segments)
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
                    self.state.segments = _segments_to_dicts(segments)
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            # Step 3: Translate (or import from external SRT)
            step = DubbingStep.TRANSLATE.value
            if self.state.is_step_done(step):
                segments = _dicts_to_segments(self.state.segments)
                logger.info("Skipping %s (cached)", step)
            else:
                ts = time.time()
                if self.config.external_srt_path:
                    # Use official SRT instead of ML translation
                    if getattr(self.config, 'use_srt_timing', False):
                        # Use SRT timing directly - creates clean segments from SRT
                        logger.info(
                            "Creating segments from SRT timing: %s",
                            self.config.external_srt_path,
                        )
                        segments = self._create_segments_from_srt(
                            self.config.external_srt_path,
                        )
                    else:
                        # Legacy mode: match SRT to Whisper segments (may cause duplicates)
                        logger.info(
                            "Importing external SRT: %s",
                            self.config.external_srt_path,
                        )
                        segments = self._import_external_srt(
                            self.config.external_srt_path, segments,
                        )
                else:
                    segments = self._translate(segments)
                result.step_times[step] = time.time() - ts
                self.state.segments = _segments_to_dicts(segments)
                self.state.mark_step_done(step)
                self._save_checkpoint()

            # Step 3.5: Merge short adjacent segments (on by default)
            step = DubbingStep.MERGE_SEGMENTS.value
            if self.config.merge_segments:
                if self.state.is_step_done(step):
                    segments = _dicts_to_segments(self.state.segments)
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
                    self.state.segments = _segments_to_dicts(segments)
                    self.state.mark_step_done(step)
                    self._save_checkpoint()

            # Step 4: Generate TTS
            step = DubbingStep.GENERATE_TTS.value
            if self.state.is_step_done(step):
                segments = _dicts_to_segments(self.state.segments)
                logger.info("Skipping %s (cached)", step)
            else:
                ts = time.time()
                segments = self._generate_tts(segments)
                result.step_times[step] = time.time() - ts
                self.state.segments = _segments_to_dicts(segments)
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
