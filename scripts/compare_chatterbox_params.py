#!/usr/bin/env python3
"""Compare Chatterbox TTS with 3 parameter configurations for Polish SRT.

Uses voice cloning with varying exaggeration and cfg_weight settings.

Output: test-files/tts_comparison/ with WAV files for each variant.
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiosmith.pipeline.helpers import _clean_tts_text, _dedup_repeated_words
from audiosmith.srt import parse_srt, timestamp_to_seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("compare_chatterbox")

MAX_TIME_S = 600.0  # First 10 minutes
OUTPUT_DIR = Path("test-files/tts_comparison")
VOICE_REF = Path("test-files/tts_comparison/voice_refs/witcher_polish_ref.wav")

# Chatterbox parameter variants
VARIANTS = {
    "chatterbox_A_clone": {
        "label": "voice_clone",
        "audio_prompt_path": VOICE_REF,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
    },
    "chatterbox_B_tuned": {
        "label": "voice_clone_tuned",
        "audio_prompt_path": VOICE_REF,
        "exaggeration": 0.65,
        "cfg_weight": 0.6,
    },
    "chatterbox_C_expressive": {
        "label": "voice_clone_expressive",
        "audio_prompt_path": VOICE_REF,
        "exaggeration": 0.8,
        "cfg_weight": 0.7,
    },
}


def parse_segments(srt_path: Path):
    """Parse SRT and return segments within first 10 minutes."""
    content = srt_path.read_text(encoding='utf-8')
    entries = parse_srt(content)

    segments = []
    for entry in entries:
        # Convert timestamp to seconds
        start_sec = timestamp_to_seconds(entry.start_time)

        if start_sec > MAX_TIME_S:
            break

        text = _clean_tts_text(entry.text)
        if not text or not text.strip():
            continue

        text = _dedup_repeated_words(text)

        # Skip very short segments (< 2 words)
        if len(text.split()) < 2:
            continue

        segments.append({
            "start": start_sec,
            "end": timestamp_to_seconds(entry.end_time),
            "text": text,
        })

    return segments


def generate_variant(variant_name: str, variant_config: dict, segments: list, engine, output_dir: Path):
    """Generate TTS for all segments with one variant config."""
    wav_path = output_dir / f"{variant_name}.wav"

    if wav_path.exists():
        logger.info("Skipping %s (already exists: %s)", variant_name, wav_path)
        return wav_path

    logger.info("=== Generating %s (%s) ===", variant_name, variant_config["label"])

    sr = engine.sample_rate
    all_audio = []
    success = 0
    start_time = time.time()

    for i, seg in enumerate(segments):
        try:
            # Call synthesize with variant-specific parameters
            result = engine.synthesize(
                seg["text"],
                language="pl",
                audio_prompt_path=str(variant_config["audio_prompt_path"]),
                exaggeration=variant_config["exaggeration"],
                cfg_weight=variant_config["cfg_weight"],
            )

            # ChatterboxTTS returns ndarray directly, not tuple
            audio = result

            if len(audio) > 0:
                all_audio.append(audio)
                success += 1

            if (i + 1) % 20 == 0:
                logger.info("  %s: %d/%d segments done", variant_name, i + 1, len(segments))

        except Exception as e:
            logger.warning("  %s seg %d failed: %s", variant_name, i, str(e)[:100])
            continue

    elapsed = time.time() - start_time
    logger.info("  %s: %d/%d segments in %.1fs", variant_name, success, len(segments), elapsed)

    if all_audio:
        # Concatenate with 300ms silence gaps
        silence = np.zeros(int(0.3 * sr), dtype=np.float32)
        combined = []
        for chunk in all_audio:
            combined.append(chunk)
            combined.append(silence)

        full_audio = np.concatenate(combined)

        # Peak-normalize to 0.95
        peak = np.max(np.abs(full_audio))
        if peak > 0.01:
            full_audio = full_audio / peak * 0.95

        sf.write(str(wav_path), full_audio, sr)
        duration = len(full_audio) / sr
        logger.info("  %s: saved %.1fs of audio to %s", variant_name, duration, wav_path)
        return wav_path
    else:
        logger.error("  %s: no audio generated!", variant_name)
        return None


def main():
    srt_path = Path("test-files/videos/Original_subtitiles/Marty.Supreme.2025.pl.srt")

    if not srt_path.exists():
        logger.error("SRT file not found: %s", srt_path)
        sys.exit(1)

    if not VOICE_REF.exists():
        logger.error("Voice reference file not found: %s", VOICE_REF)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    segments = parse_segments(srt_path)
    logger.info("Parsed %d segments from first %.0f seconds", len(segments), MAX_TIME_S)

    # Show sample of text
    for seg in segments[:3]:
        logger.info("  Sample: [%.1f-%.1f] %s", seg["start"], seg["end"], seg["text"][:60])

    # Load ChatterboxTTS once (it's 3GB)
    logger.info("Loading ChatterboxTTS (this may take 1-2 minutes)...")
    try:
        from audiosmith.tts import ChatterboxTTS
        engine = ChatterboxTTS(device="cuda")
        engine.load_model()
        logger.info("ChatterboxTTS loaded successfully, sr=%d", engine.sample_rate)
    except Exception as e:
        logger.error("Failed to load ChatterboxTTS: %s", e)
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

    results = {}

    # Generate all variants
    for variant_name, variant_config in VARIANTS.items():
        try:
            result = generate_variant(variant_name, variant_config, segments, engine, OUTPUT_DIR)
            results[variant_name] = result
        except Exception as e:
            logger.error("%s failed entirely: %s", variant_name, e)
            import traceback
            logger.debug(traceback.format_exc())
            results[variant_name] = None

    # Cleanup engine
    if hasattr(engine, 'cleanup'):
        try:
            engine.cleanup()
        except Exception as e:
            logger.warning("Engine cleanup error: %s", str(e)[:80])

    # Summary
    print("\n" + "=" * 70)
    print("CHATTERBOX PARAMETER COMPARISON — RESULTS")
    print("=" * 70)
    for variant_name, variant_config in VARIANTS.items():
        path = results.get(variant_name)
        if path and Path(path).exists():
            info = sf.info(str(path))
            print(f"  {variant_name:30s} ({variant_config['label']:25s}): {info.duration:6.1f}s")
        else:
            print(f"  {variant_name:30s}: FAILED")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Voice reference: {VOICE_REF}")


if __name__ == "__main__":
    main()
