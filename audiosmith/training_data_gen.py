"""Polish TTS training data generation pipeline.

Generates paired text+audio training data for Qwen3-TTS LoRA fine-tuning
from multiple sources:
  1. Audiobook — transcribe existing Polish audiobook (Wiedzmin) via Whisper
  2. Chatterbox — synthesize from text corpus on local GPU
  3. ElevenLabs — synthesize from text corpus via cloud API
  4. Fish Speech — synthesize from text corpus via cloud API

6-stage pipeline:
  Stage 1: Corpus preparation (Wikipedia text for TTS sources)
  Stage 2: Audiobook processing (transcribe → segment → extract clips)
  Stage 3: TTS generation (Chatterbox / ElevenLabs / Fish Speech)
  Stage 4: Post-processing (Polish PP on TTS-generated audio)
  Stage 5: Quality filtering (duration, silence, SNR, clipping)
  Stage 6: Tokenization (Qwen3-TTS-Tokenizer-12Hz → JSONL)
"""

import gc
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from audiosmith.exceptions import TrainingError

logger = logging.getLogger(__name__)

# Emotion → Chatterbox TTS parameters (reused from pipeline.py)
_EMOTION_TTS_MAP: Dict[str, Dict[str, float]] = {
    "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5},
    "happy": {"exaggeration": 0.7, "cfg_weight": 0.5},
    "sad": {"exaggeration": 0.3, "cfg_weight": 0.4},
    "excited": {"exaggeration": 0.8, "cfg_weight": 0.6},
    "angry": {"exaggeration": 0.9, "cfg_weight": 0.7},
    "fearful": {"exaggeration": 0.6, "cfg_weight": 0.6},
}


@dataclass
class TrainingDataConfig:
    """Configuration for training data generation."""

    output_dir: Path = field(default_factory=lambda: Path("data/polish_training"))
    corpus_path: Optional[Path] = None
    sample_rate: int = 24_000
    device: str = "cuda"

    # Sources to enable
    enable_audiobook: bool = False
    audiobook_dir: Optional[Path] = None

    enable_chatterbox: bool = True
    enable_elevenlabs: bool = False
    enable_fish: bool = False

    # ElevenLabs config
    elevenlabs_model: str = "eleven_v3"
    elevenlabs_voice_name: Optional[str] = None

    # Fish Speech config
    fish_model: str = "speech-01"

    # Generation parameters
    emotions: List[str] = field(
        default_factory=lambda: ["neutral", "happy", "sad", "excited"]
    )
    emotion_weights: List[float] = field(
        default_factory=lambda: [0.4, 0.3, 0.15, 0.15]
    )
    target_sample_count: int = 8_000
    checkpoint_interval: int = 100

    # Post-processing
    enable_postprocess: bool = True

    # Quality filter thresholds
    min_duration_s: float = 1.0
    max_duration_s: float = 20.0
    min_snr_db: float = 15.0
    max_silence_pct: float = 50.0  # Natural speech has pauses; PP adds silence at punctuation
    max_peak: float = 1.0  # PP limiter clips to 1.0; only reject >1.0 (corrupted)
    min_rms: float = 0.01


@dataclass
class Checkpoint:
    """Checkpoint state for resumable generation."""

    stage: int = 0
    source: str = ""
    sample_idx: int = 0
    total: int = 0
    completed_stages: List[int] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class TrainingDataGenerator:
    """Orchestrates multi-source Polish TTS training data generation."""

    def __init__(self, config: TrainingDataConfig):
        self.config = config
        self._checkpoint_path = config.output_dir / "checkpoint.json"
        self._manifest_path = config.output_dir / "manifest.jsonl"
        self._checkpoint = Checkpoint()

        # Create output subdirectories
        for sub in ["raw", "processed", "filtered"]:
            (config.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def run(self, stage: str = "all", resume: bool = False) -> Dict[str, Any]:
        """Execute the training data pipeline.

        Args:
            stage: Which stage to run — "1" through "6", or "all".
            resume: Whether to resume from checkpoint.

        Returns:
            Summary dict with counts and timing per stage.
        """
        if resume and self._checkpoint_path.exists():
            self._checkpoint = Checkpoint.load(self._checkpoint_path)
            logger.info("Resumed from checkpoint: stage %d, sample %d",
                        self._checkpoint.stage, self._checkpoint.sample_idx)

        summary: Dict[str, Any] = {}
        stages = {
            "1": self._stage1_corpus,
            "2": self._stage2_audiobook,
            "3": self._stage3_generate,
            "4": self._stage4_postprocess,
            "5": self._stage5_filter,
            "6": self._stage6_tokenize,
        }

        if stage == "all":
            run_stages = ["1", "2", "3", "4", "5", "6"]
        else:
            run_stages = [stage]

        for s in run_stages:
            stage_num = int(s)
            if resume and stage_num in self._checkpoint.completed_stages:
                logger.info("Stage %s already complete, skipping", s)
                continue

            if s not in stages:
                raise TrainingError(f"Unknown stage: {s}", error_code="TRAIN_BAD_STAGE")

            t0 = time.time()
            logger.info("=== Stage %s starting ===", s)
            result = stages[s]()
            elapsed = time.time() - t0

            summary[f"stage_{s}"] = {
                "elapsed_s": round(elapsed, 1),
                **result,
            }
            self._checkpoint.completed_stages.append(stage_num)
            self._checkpoint.save(self._checkpoint_path)
            logger.info("=== Stage %s complete (%.1fs) ===", s, elapsed)

        return summary

    # ── Stage 1: Corpus Preparation ──────────────────────────────────

    def _stage1_corpus(self) -> Dict[str, Any]:
        """Download Wikipedia, extract Polish sentences, save corpus."""
        from audiosmith.polish_corpus import PolishCorpusManager

        corpus_path = self.config.corpus_path
        if corpus_path and corpus_path.exists():
            manager = PolishCorpusManager()
            sentences = manager.load_corpus(corpus_path)
            return {"sentences": len(sentences), "source": "existing"}

        manager = PolishCorpusManager()
        wiki_path = manager.download_wikipedia()
        sentences = manager.extract_sentences(wiki_path, max_sentences=50_000)
        sentences = manager.diversify(sentences, target_count=self.config.target_sample_count)

        corpus_path = self.config.output_dir / "corpus.txt"
        manager.save_corpus(sentences, corpus_path)
        self.config.corpus_path = corpus_path

        return {"sentences": len(sentences), "source": "wikipedia"}

    # ── Stage 2: Audiobook Processing ────────────────────────────────

    def _stage2_audiobook(self) -> Dict[str, Any]:
        """Transcribe and segment audiobook into training clips."""
        if not self.config.enable_audiobook or not self.config.audiobook_dir:
            logger.info("Audiobook source disabled, skipping stage 2")
            return {"skipped": True}

        audiobook_dir = self.config.audiobook_dir
        if not audiobook_dir.exists():
            raise TrainingError(
                f"Audiobook directory not found: {audiobook_dir}",
                error_code="TRAIN_AUDIOBOOK_MISSING",
            )

        import soundfile as sf

        # Find all audio files
        audio_files = sorted(
            p for p in audiobook_dir.rglob("*")
            if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        )
        logger.info("Found %d audiobook files in %s", len(audio_files), audiobook_dir)

        if not audio_files:
            return {"skipped": True, "reason": "no audio files found"}

        raw_dir = self.config.output_dir / "raw"
        clip_count = 0
        manifest_entries: List[Dict] = []

        # Lazy import Whisper
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device=self.config.device,
                             compute_type="float16")

        for file_idx, audio_path in enumerate(audio_files):
            logger.info("Processing audiobook file %d/%d: %s",
                        file_idx + 1, len(audio_files), audio_path.name)

            try:
                segments_iter, info = model.transcribe(
                    str(audio_path), language="pl",
                    vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
                )

                # Load audio for clipping
                audio_data, sr = sf.read(str(audio_path), dtype="float32")
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)

                for seg in segments_iter:
                    text = seg.text.strip()
                    if not text or len(text) < 10:
                        continue

                    start_sample = int(seg.start * sr)
                    end_sample = int(seg.end * sr)
                    clip = audio_data[start_sample:end_sample]

                    duration = len(clip) / sr
                    if duration < self.config.min_duration_s or duration > self.config.max_duration_s:
                        continue

                    # Resample to target if needed
                    if sr != self.config.sample_rate:
                        clip = self._resample(clip, sr, self.config.sample_rate)

                    clip_name = f"audiobook_{clip_count:06d}.wav"
                    sf.write(str(raw_dir / clip_name), clip, self.config.sample_rate)

                    manifest_entries.append({
                        "id": clip_name.replace(".wav", ""),
                        "text": text,
                        "source": "audiobook",
                        "source_file": audio_path.name,
                        "duration_s": round(duration, 2),
                        "language": "pl",
                    })
                    clip_count += 1

                    if clip_count % 100 == 0:
                        logger.info("Extracted %d audiobook clips", clip_count)

            except Exception as e:
                logger.warning("Failed to process %s: %s", audio_path.name, e)
                self._checkpoint.failed[str(audio_path)] = str(e)

        # Append to manifest
        self._append_manifest(manifest_entries)

        del model
        gc.collect()

        return {"clips": clip_count, "files_processed": len(audio_files)}

    # ── Stage 3: TTS Generation ──────────────────────────────────────

    def _stage3_generate(self) -> Dict[str, Any]:
        """Generate audio from text corpus using enabled TTS engines."""
        corpus_path = self.config.corpus_path or self.config.output_dir / "corpus.txt"
        if not corpus_path.exists():
            raise TrainingError(
                "No corpus found. Run stage 1 first.",
                error_code="TRAIN_NO_CORPUS",
            )

        from audiosmith.polish_corpus import PolishCorpusManager
        manager = PolishCorpusManager()
        sentences = manager.load_corpus(corpus_path)

        results: Dict[str, Any] = {}

        # Split sentences across enabled engines
        engine_sentences = self._distribute_sentences(sentences)

        if self.config.enable_chatterbox and "chatterbox" in engine_sentences:
            count = self._generate_chatterbox(engine_sentences["chatterbox"])
            results["chatterbox"] = count

        if self.config.enable_elevenlabs and "elevenlabs" in engine_sentences:
            count = self._generate_elevenlabs(engine_sentences["elevenlabs"])
            results["elevenlabs"] = count

        if self.config.enable_fish and "fish" in engine_sentences:
            count = self._generate_fish(engine_sentences["fish"])
            results["fish"] = count

        return results

    def _distribute_sentences(self, sentences: List[str]) -> Dict[str, List[str]]:
        """Distribute sentences across enabled engines.

        Chatterbox gets the majority (free, Grade A).
        Cloud engines get smaller shares for diversity.
        """
        engines: List[str] = []
        weights: List[float] = []

        if self.config.enable_chatterbox:
            engines.append("chatterbox")
            weights.append(0.6)
        if self.config.enable_elevenlabs:
            engines.append("elevenlabs")
            weights.append(0.2)
        if self.config.enable_fish:
            engines.append("fish")
            weights.append(0.2)

        if not engines:
            return {}

        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Assign sentences
        random.shuffle(sentences)
        result: Dict[str, List[str]] = {e: [] for e in engines}
        for i, sent in enumerate(sentences[:self.config.target_sample_count]):
            # Weighted assignment
            r = random.random()
            cumulative = 0.0
            for eng, w in zip(engines, weights):
                cumulative += w
                if r <= cumulative:
                    result[eng].append(sent)
                    break

        for eng, sents in result.items():
            logger.info("Assigned %d sentences to %s", len(sents), eng)

        return result

    def _generate_chatterbox(self, sentences: List[str]) -> int:
        """Generate Polish audio using Chatterbox (local GPU)."""
        import soundfile as sf
        from audiosmith.tts import ChatterboxTTS

        raw_dir = self.config.output_dir / "raw"
        cb = ChatterboxTTS(device=self.config.device)
        cb.load_model()
        sr = cb.sample_rate

        count = 0
        manifest_entries: List[Dict] = []

        # Resume support
        start_idx = 0
        if (self._checkpoint.source == "chatterbox" and
                self._checkpoint.stage == 3):
            start_idx = self._checkpoint.sample_idx + 1

        for i, text in enumerate(sentences[start_idx:], start=start_idx):
            emotion = random.choices(
                self.config.emotions, weights=self.config.emotion_weights
            )[0]
            params = _EMOTION_TTS_MAP.get(emotion, _EMOTION_TTS_MAP["neutral"])

            try:
                audio = cb.synthesize(
                    text, language="pl",
                    exaggeration=params["exaggeration"],
                    cfg_weight=params["cfg_weight"],
                )

                clip_name = f"cb_{count:06d}.wav"
                sf.write(str(raw_dir / clip_name), audio, sr)

                manifest_entries.append({
                    "id": clip_name.replace(".wav", ""),
                    "text": text,
                    "source": "chatterbox",
                    "emotion": emotion,
                    "duration_s": round(len(audio) / sr, 2),
                    "language": "pl",
                })
                count += 1

            except Exception as e:
                logger.warning("Chatterbox failed on sample %d: %s", i, e)
                self._checkpoint.failed[f"cb_{i}"] = str(e)

            # Checkpoint
            if count % self.config.checkpoint_interval == 0 and count > 0:
                self._checkpoint.stage = 3
                self._checkpoint.source = "chatterbox"
                self._checkpoint.sample_idx = i
                self._checkpoint.save(self._checkpoint_path)
                self._append_manifest(manifest_entries)
                manifest_entries.clear()
                logger.info("Checkpoint: %d Chatterbox samples generated", count)

        # Final flush
        self._append_manifest(manifest_entries)
        cb.cleanup()

        return count

    def _generate_elevenlabs(self, sentences: List[str]) -> int:
        """Generate Polish audio using ElevenLabs cloud API."""
        import soundfile as sf
        from audiosmith.elevenlabs_tts import ElevenLabsTTS

        raw_dir = self.config.output_dir / "raw"
        el = ElevenLabsTTS()

        count = 0
        manifest_entries: List[Dict] = []

        for i, text in enumerate(sentences):
            try:
                audio, sr = el.synthesize(
                    text,
                    voice_name=self.config.elevenlabs_voice_name,
                )

                clip_name = f"el_{count:06d}.wav"
                sf.write(str(raw_dir / clip_name), audio, sr)

                manifest_entries.append({
                    "id": clip_name.replace(".wav", ""),
                    "text": text,
                    "source": "elevenlabs",
                    "model": self.config.elevenlabs_model,
                    "duration_s": round(len(audio) / sr, 2),
                    "language": "pl",
                })
                count += 1

            except Exception as e:
                logger.warning("ElevenLabs failed on sample %d: %s", i, e)
                self._checkpoint.failed[f"el_{i}"] = str(e)

            if count % 50 == 0 and count > 0:
                self._append_manifest(manifest_entries)
                manifest_entries.clear()
                logger.info("ElevenLabs: %d samples generated", count)

        self._append_manifest(manifest_entries)

        return count

    def _generate_fish(self, sentences: List[str]) -> int:
        """Generate Polish audio using Fish Speech cloud API."""
        import soundfile as sf
        from audiosmith.fish_speech_tts import FishSpeechTTS

        raw_dir = self.config.output_dir / "raw"
        fish = FishSpeechTTS()

        count = 0
        manifest_entries: List[Dict] = []

        for i, text in enumerate(sentences):
            try:
                audio, sr = fish.synthesize(
                    text, language="pl",
                )

                clip_name = f"fish_{count:06d}.wav"
                sf.write(str(raw_dir / clip_name), audio, sr)

                manifest_entries.append({
                    "id": clip_name.replace(".wav", ""),
                    "text": text,
                    "source": "fish",
                    "model": self.config.fish_model,
                    "duration_s": round(len(audio) / sr, 2),
                    "language": "pl",
                })
                count += 1

            except Exception as e:
                logger.warning("Fish Speech failed on sample %d: %s", i, e)
                self._checkpoint.failed[f"fish_{i}"] = str(e)

            if count % 50 == 0 and count > 0:
                self._append_manifest(manifest_entries)
                manifest_entries.clear()
                logger.info("Fish Speech: %d samples generated", count)

        self._append_manifest(manifest_entries)

        return count

    # ── Stage 4: Post-Processing ─────────────────────────────────────

    def _stage4_postprocess(self) -> Dict[str, Any]:
        """Apply Polish post-processing to TTS-generated audio.

        Skips audiobook clips (already professional quality).
        """
        if not self.config.enable_postprocess:
            logger.info("Post-processing disabled, skipping stage 4")
            return {"skipped": True}

        import soundfile as sf
        from audiosmith.tts_postprocessor import TTSPostProcessor, PostProcessConfig

        raw_dir = self.config.output_dir / "raw"
        proc_dir = self.config.output_dir / "processed"

        # Load manifest to know which clips are TTS-generated
        manifest = self._load_manifest()

        # Polish PP config (same as pipeline.py overrides)
        pp_config = PostProcessConfig(
            enable_silence=True,
            enable_dynamics=True,
            enable_breath=False,
            enable_warmth=False,
            enable_spectral_matching=True,
            enable_micro_dynamics=True,
            enable_normalize=True,
            target_rms_adaptive=False,
            target_rms=0.13,
            spectral_intensity=0.3,
            global_intensity=0.7,
            language="pl",
        )
        processor = TTSPostProcessor(pp_config)

        processed = 0
        skipped = 0

        for entry in manifest:
            source = entry.get("source", "")
            clip_id = entry["id"]

            # Skip audiobook clips — already professional quality
            if source == "audiobook":
                # Copy to processed as-is
                src = raw_dir / f"{clip_id}.wav"
                dst = proc_dir / f"{clip_id}.wav"
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
                skipped += 1
                continue

            src = raw_dir / f"{clip_id}.wav"
            dst = proc_dir / f"{clip_id}.wav"
            if not src.exists() or dst.exists():
                continue

            try:
                audio, sr = sf.read(str(src), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                text = entry.get("text", "")
                emotion_name = entry.get("emotion", "neutral")
                emotion_dict = {"primary": emotion_name, "intensity": 0.5}
                audio_out = processor.process(audio, sr, text=text, emotion=emotion_dict)

                sf.write(str(dst), audio_out, sr)
                processed += 1

            except Exception as e:
                logger.warning("Post-processing failed for %s: %s", clip_id, e)

            if processed % 100 == 0 and processed > 0:
                logger.info("Post-processed %d clips", processed)

        return {"processed": processed, "skipped_audiobook": skipped}

    # ── Stage 5: Quality Filtering ───────────────────────────────────

    def _stage5_filter(self) -> Dict[str, Any]:
        """Filter processed clips by quality criteria."""
        import soundfile as sf

        proc_dir = self.config.output_dir / "processed"
        filt_dir = self.config.output_dir / "filtered"
        raw_dir = self.config.output_dir / "raw"

        # Use processed dir if it has files, otherwise raw
        source_dir = proc_dir if any(proc_dir.glob("*.wav")) else raw_dir

        manifest = self._load_manifest()
        passed = 0
        rejected = 0
        reject_reasons: Dict[str, int] = {}

        for entry in manifest:
            clip_id = entry["id"]
            wav_path = source_dir / f"{clip_id}.wav"
            if not wav_path.exists():
                continue

            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                reason = self._check_quality(audio, sr)
                if reason:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                    rejected += 1
                    continue

                # Copy to filtered
                import shutil
                dst = filt_dir / f"{clip_id}.wav"
                shutil.copy2(wav_path, dst)
                passed += 1

            except Exception as e:
                logger.warning("Quality check failed for %s: %s", clip_id, e)
                rejected += 1

        # Save quality report
        report = {
            "passed": passed,
            "rejected": rejected,
            "reject_reasons": reject_reasons,
            "pass_rate": round(passed / max(passed + rejected, 1) * 100, 1),
        }
        report_path = self.config.output_dir / "quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Quality filter: %d passed, %d rejected (%.1f%% pass rate)",
                     passed, rejected, report["pass_rate"])
        return report

    def _check_quality(self, audio: np.ndarray, sr: int) -> Optional[str]:
        """Check audio quality against thresholds.

        Returns None if audio passes, or a rejection reason string.
        """
        duration = len(audio) / sr

        if duration < self.config.min_duration_s:
            return "too_short"
        if duration > self.config.max_duration_s:
            return "too_long"

        # RMS check
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.config.min_rms:
            return "near_silent"

        # Sustained clipping check — reject if >5% of samples are at max
        # (PP limiter hitting 1.0 occasionally is normal; sustained clipping is not)
        peak = float(np.max(np.abs(audio)))
        if peak > self.config.max_peak:
            return "clipping"
        clipped_pct = float(np.mean(np.abs(audio) >= 0.999)) * 100
        if clipped_pct > 5.0:
            return "sustained_clipping"

        # Silence percentage (frames below -40dB threshold)
        silence_threshold = 10 ** (-40 / 20)
        frame_size = int(0.025 * sr)  # 25ms frames
        n_frames = max(1, len(audio) // frame_size)
        silent_frames = 0
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            if np.sqrt(np.mean(frame ** 2)) < silence_threshold:
                silent_frames += 1
        silence_pct = (silent_frames / n_frames) * 100
        if silence_pct > self.config.max_silence_pct:
            return "excess_silence"

        # SNR estimate (signal vs noise floor from quietest 10% of frames)
        frame_rms_values = []
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            frame_rms_values.append(float(np.sqrt(np.mean(frame ** 2))))
        frame_rms_values.sort()
        noise_floor = np.mean(frame_rms_values[:max(1, n_frames // 10)])
        if noise_floor > 0:
            snr = 20 * np.log10(rms / noise_floor)
            if snr < self.config.min_snr_db:
                return "low_snr"

        return None

    # ── Stage 6: Tokenization ────────────────────────────────────────

    def _stage6_tokenize(self) -> Dict[str, Any]:
        """Tokenize filtered audio into Qwen3-TTS format.

        Produces JSONL with text + discrete audio codes.
        """
        import soundfile as sf

        filt_dir = self.config.output_dir / "filtered"
        output_jsonl = self.config.output_dir / "training_data.jsonl"
        manifest = self._load_manifest()

        # Try to load Qwen3 tokenizer
        tokenizer = self._load_qwen3_tokenizer()

        tokenized = 0
        with open(output_jsonl, "w", encoding="utf-8") as out:
            for entry in manifest:
                clip_id = entry["id"]
                wav_path = filt_dir / f"{clip_id}.wav"
                if not wav_path.exists():
                    continue

                try:
                    audio, sr = sf.read(str(wav_path), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)

                    # Tokenize audio
                    if tokenizer is not None:
                        codes = tokenizer.encode(audio, sr)
                    else:
                        # Fallback: store raw audio path for later tokenization
                        codes = None

                    record = {
                        "text": entry["text"],
                        "language": entry.get("language", "pl"),
                        "source": entry.get("source", "unknown"),
                        "emotion": entry.get("emotion", "neutral"),
                        "duration_s": entry.get("duration_s", round(len(audio) / sr, 2)),
                    }
                    if codes is not None:
                        record["audio_codes"] = codes
                    else:
                        record["audio_path"] = str(wav_path)

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    tokenized += 1

                except Exception as e:
                    logger.warning("Tokenization failed for %s: %s", clip_id, e)

        logger.info("Tokenized %d samples to %s", tokenized, output_jsonl)
        return {"tokenized": tokenized, "output": str(output_jsonl)}

    def _load_qwen3_tokenizer(self):
        """Try to load the Qwen3-TTS-Tokenizer-12Hz. Returns None if unavailable."""
        try:
            from qwen_tts import Qwen3TTSTokenizer
            tokenizer = Qwen3TTSTokenizer.from_pretrained(
                "Qwen/Qwen3-TTS-Tokenizer-12Hz"
            )
            logger.info("Loaded Qwen3-TTS-Tokenizer-12Hz")
            return tokenizer
        except ImportError:
            logger.warning("qwen_tts package not available — storing audio paths instead of codes")
            return None
        except Exception as e:
            logger.warning("Failed to load Qwen3 tokenizer: %s — storing audio paths", e)
            return None

    # ── Helpers ───────────────────────────────────────────────────────

    def _append_manifest(self, entries: List[Dict]) -> None:
        """Append entries to the JSONL manifest file."""
        if not entries:
            return
        with open(self._manifest_path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _load_manifest(self) -> List[Dict]:
        """Load all entries from the JSONL manifest."""
        if not self._manifest_path.exists():
            return []
        entries = []
        with open(self._manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling (no scipy dependency)."""
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def dry_run(self) -> Dict[str, Any]:
        """Validate config and estimate workload without generating anything."""
        info: Dict[str, Any] = {
            "output_dir": str(self.config.output_dir),
            "sources": [],
            "target_samples": self.config.target_sample_count,
            "estimated_disk_gb": 0.0,
        }

        if self.config.enable_audiobook and self.config.audiobook_dir:
            audio_files = list(self.config.audiobook_dir.rglob("*.mp3"))
            info["sources"].append({
                "name": "audiobook",
                "files": len(audio_files),
                "dir": str(self.config.audiobook_dir),
            })

        if self.config.enable_chatterbox:
            info["sources"].append({"name": "chatterbox", "type": "local_gpu"})
        if self.config.enable_elevenlabs:
            info["sources"].append({"name": "elevenlabs", "type": "cloud"})
        if self.config.enable_fish:
            info["sources"].append({"name": "fish", "type": "cloud"})

        # Rough disk estimate: ~450KB per sample WAV at 24kHz
        n = self.config.target_sample_count
        info["estimated_disk_gb"] = round(n * 450_000 / (1024 ** 3) * 3, 1)  # raw+proc+filtered

        corpus_path = self.config.corpus_path
        if corpus_path and corpus_path.exists():
            with open(corpus_path) as f:
                lines = sum(1 for _ in f)
            info["corpus_sentences"] = lines

        return info

    def export_for_f5(self, output_dir: Optional[Path] = None) -> Path:
        """Export filtered training data in F5-TTS format.

        Reads filtered_manifest.jsonl, writes metadata.csv (pipe-delimited)
        and symlinks audio files into an audio/ subdirectory.

        Args:
            output_dir: Output directory. Defaults to train_dir/f5_format.

        Returns:
            Path to the output directory containing metadata.csv + audio/.
        """
        from audiosmith.f5_finetune import F5FineTuneConfig, F5FineTuneTrainer

        config = F5FineTuneConfig(train_dir=self.config.output_dir)
        trainer = F5FineTuneTrainer(config)
        manifest = self.config.output_dir / "filtered_manifest.jsonl"
        stats = trainer.prepare_data(
            manifest_jsonl=manifest,
            output_dir=output_dir,
        )
        logger.info(
            "Exported %d samples (%.1fh) for F5-TTS → %s",
            stats["samples"], stats["total_hours"], stats["output_dir"],
        )
        return Path(stats["output_dir"])
