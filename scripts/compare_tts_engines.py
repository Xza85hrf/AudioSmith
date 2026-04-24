#!/usr/bin/env python3
"""Compare TTS engines by generating first 10 minutes of a Polish SRT.

Usage:
    python scripts/compare_tts_engines.py test-files/videos/Original_subtitiles/Marty.Supreme.2025.pl.srt

Output: test-files/tts_comparison/ with one WAV per engine.
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
logger = logging.getLogger("compare_tts")

MAX_TIME_S = 600.0  # First 10 minutes
OUTPUT_DIR = Path("test-files/tts_comparison")

# Engine configs: name -> init kwargs
ENGINES = {
    "piper": {
        "voice": "pl_PL-darkman-medium",
        "model_path": Path.home() / ".local/share/piper-voices/pl_PL-darkman-medium.onnx",
    },
    "f5": {
        "model_name": "f5-polish",
        "device": "cuda",
    },
    "chatterbox": {
        "device": "cuda",
    },
    "fish": {
        "base_url": "http://127.0.0.1:8080",
        "temperature": 0.5,
        "top_p": 0.7,
    },
    "qwen3": {
        "device": "cuda",
    },
}

# Per-engine synth kwargs (engines that don't accept certain params)
# Piper doesn't accept 'language' param — it's set via voice name
ENGINE_SYNTH_KWARGS = {
    "piper": {},  # Piper uses voice, not language
    "f5": {"language": "pl"},
    "chatterbox": {"language": "pl"},
    "fish": {"language": "pl"},
    "qwen3": {"language": "pl"},
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


def generate_with_engine(engine_name: str, engine_kwargs: dict, segments: list, output_dir: Path):
    """Generate TTS for all segments with one engine, concatenate to single WAV."""
    from audiosmith.tts_protocol import get_engine

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{engine_name}.wav"

    if wav_path.exists():
        logger.info("Skipping %s (already exists: %s)", engine_name, wav_path)
        return wav_path

    logger.info("=== Generating with %s ===", engine_name.upper())

    engine = None
    try:
        # Create engine via factory
        engine = get_engine(engine_name, **engine_kwargs)

        # Load model if available
        if hasattr(engine, 'load_model'):
            try:
                engine.load_model()
            except Exception as e:
                logger.warning("  %s: load_model() raised: %s (continuing)", engine_name, str(e)[:80])

        sr = engine.sample_rate
        all_audio = []
        success = 0
        start_time = time.time()

        for i, seg in enumerate(segments):
            try:
                synth_kwargs = ENGINE_SYNTH_KWARGS.get(engine_name, {"language": "pl"})
                result = engine.synthesize(seg["text"], **synth_kwargs)

                # Handle engines that return ndarray vs (ndarray, sr) tuple
                if isinstance(result, tuple):
                    audio = result[0]
                else:
                    audio = result

                if len(audio) > 0:
                    all_audio.append(audio)
                    success += 1

                if (i + 1) % 20 == 0:
                    logger.info("  %s: %d/%d segments done", engine_name, i + 1, len(segments))

            except Exception as e:
                logger.warning("  %s seg %d failed: %s", engine_name, i, str(e)[:100])
                continue

        elapsed = time.time() - start_time
        logger.info("  %s: %d/%d segments in %.1fs", engine_name, success, len(segments), elapsed)

        if all_audio:
            # Concatenate with 300ms silence gaps
            silence = np.zeros(int(0.3 * sr), dtype=np.float32)
            combined = []
            for chunk in all_audio:
                combined.append(chunk)
                combined.append(silence)

            full_audio = np.concatenate(combined)

            # Normalize to prevent clipping
            peak = np.max(np.abs(full_audio))
            if peak > 0.01:
                full_audio = full_audio / peak * 0.95

            sf.write(str(wav_path), full_audio, sr)
            duration = len(full_audio) / sr
            logger.info("  %s: saved %.1fs of audio to %s", engine_name, duration, wav_path)
            return wav_path
        else:
            logger.error("  %s: no audio generated!", engine_name)
            return None

    except Exception as e:
        logger.error("  %s FAILED: %s", engine_name, e)
        import traceback
        logger.debug(traceback.format_exc())
        return None
    finally:
        # Cleanup engine resources
        if engine is not None and hasattr(engine, 'cleanup'):
            try:
                engine.cleanup()
            except Exception as e:
                logger.warning("  %s cleanup error: %s", engine_name, str(e)[:80])


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_tts_engines.py <srt_file>")
        sys.exit(1)

    srt_path = Path(sys.argv[1])
    if not srt_path.exists():
        print(f"SRT file not found: {srt_path}")
        sys.exit(1)

    segments = parse_segments(srt_path)
    logger.info("Parsed %d segments from first %.0f seconds", len(segments), MAX_TIME_S)

    # Show sample of text
    for seg in segments[:3]:
        logger.info("  Sample: [%.1f-%.1f] %s", seg["start"], seg["end"], seg["text"][:60])

    results = {}

    # Run engines in order: CPU first (Piper), then GPU engines
    engine_order = ["piper", "f5", "chatterbox", "qwen3", "fish"]

    for name in engine_order:
        if name not in ENGINES:
            continue
        try:
            result = generate_with_engine(name, ENGINES[name], segments, OUTPUT_DIR)
            results[name] = result
        except Exception as e:
            logger.error("%s failed entirely: %s", name, e)
            import traceback
            logger.debug(traceback.format_exc())
            results[name] = None

    # Summary
    print("\n" + "=" * 60)
    print("TTS ENGINE COMPARISON — RESULTS")
    print("=" * 60)
    for name in engine_order:
        path = results.get(name)
        if path and Path(path).exists():
            info = sf.info(str(path))
            print(f"  {name:12s}: {info.duration:6.1f}s  {Path(path).name}")
        else:
            print(f"  {name:12s}: FAILED")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
