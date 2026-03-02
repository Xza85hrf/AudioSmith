"""Compare Fish Speech TTS quality metrics to ElevenLabs, with post-processing variants.

Loads existing Fish Speech WAV files (or generates them if missing), applies
multiple post-processing configurations, and compares against ElevenLabs reference.

Usage:
    export FISH_API_KEY="your-key"
    python3 test_fish_vs_elevenlabs.py              # full run (generate + compare)
    python3 test_fish_vs_elevenlabs.py --reuse       # reuse existing Fish files
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


# Same test sentences as ElevenLabs
EMOTION_TEXTS = {
    "neutral": "The quarterly report shows steady growth across all sectors.",
    "angry": "I can't believe they ignored every single warning we gave them!",
    "sad": "She looked at the empty chair where he used to sit every evening.",
    "excited": "We just won the championship! This is absolutely incredible!",
    "whisper": "Listen carefully, I'm going to tell you a secret nobody knows.",
}

LANG_TEXTS = {
    "polish": "Dzień dobry, jak się masz? Mam nadzieję, że wszystko w porządku.",
    "spanish": "Buenos días, ¿cómo estás? Espero que todo esté bien.",
    "french": "Bonjour, comment allez-vous? J'espère que tout va bien.",
    "german": "Guten Tag, wie geht es Ihnen? Ich hoffe, alles ist in Ordnung.",
    "japanese": "こんにちは、お元気ですか？すべてがうまくいっていることを願っています。",
    "italian": "Buongiorno, come stai? Spero che tutto vada bene.",
}

HELLO_TEXT = "Hello world, this is a test of the text to speech engine."

LANG_CODES = {
    "polish": "pl", "spanish": "es", "french": "fr",
    "german": "de", "japanese": "ja", "italian": "it",
}

# Emotion metadata for post-processor
EMOTION_META = {
    "neutral": {"primary": "neutral", "intensity": 0.7},
    "angry": {"primary": "angry", "intensity": 0.9},
    "sad": {"primary": "sad", "intensity": 0.7},
    "excited": {"primary": "excited", "intensity": 0.9},
    "whisper": {"primary": "whisper", "intensity": 0.5},
}


def analyze_audio(wav: np.ndarray, sr: int) -> dict:
    """Compute quality metrics for an audio array."""
    if len(wav) == 0:
        return {}

    duration = len(wav) / sr
    rms = float(np.sqrt(np.mean(wav ** 2)))
    peak = float(np.max(np.abs(wav)))

    if rms > 1e-8:
        dynamic_range_db = 20 * np.log10(peak / rms)
    else:
        dynamic_range_db = 0.0

    frame_size = 512
    n_frames = len(wav) // frame_size
    silent_frames = 0
    frame_energies = []
    for i in range(n_frames):
        frame = wav[i * frame_size : (i + 1) * frame_size]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms < 0.01:
            silent_frames += 1
        frame_energies.append(np.mean(frame ** 2))
    silence_pct = (silent_frames / max(n_frames, 1)) * 100
    energy_var = float(np.var(frame_energies)) if frame_energies else 0.0

    fft = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)
    if fft.sum() > 1e-8:
        spectral_centroid = float(np.sum(freqs * fft) / np.sum(fft))
    else:
        spectral_centroid = 0.0

    zcr = float(np.mean(np.abs(np.diff(np.sign(wav))) > 0))

    return {
        "duration_s": round(duration, 2),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "silence_pct": round(silence_pct, 1),
        "spectral_centroid_hz": round(spectral_centroid, 0),
        "energy_variance": round(energy_var, 6),
        "zcr": round(zcr, 4),
    }


def analyze_existing_elevenlabs(el_dir: Path) -> dict:
    """Analyze all existing ElevenLabs test files."""
    results = {}
    for wav_file in sorted(el_dir.glob("*.wav")):
        wav, sr = sf.read(str(wav_file), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        results[wav_file.stem] = analyze_audio(wav, sr)
    return results


def generate_fish_tests(output_dir: Path, reuse: bool = False) -> dict:
    """Generate Fish Speech TTS samples via cloud API (or reuse existing)."""
    results = {}

    if reuse:
        print("  Reusing existing Fish Speech WAV files...")
        for wav_file in sorted(output_dir.glob("*.wav")):
            wav, sr = sf.read(str(wav_file), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            results[wav_file.stem] = analyze_audio(wav, sr)
        return results

    from audiosmith.fish_speech_tts import FishSpeechTTS
    fish = FishSpeechTTS()

    for emotion, text in EMOTION_TEXTS.items():
        key = f"emotion_{emotion}"
        print(f"  Generating {key}...")
        try:
            audio, sr = fish.synthesize(text, language="en")
            sf.write(str(output_dir / f"{key}.wav"), audio, sr)
            results[key] = analyze_audio(audio, sr)
            time.sleep(0.5)
        except Exception as e:
            print(f"    FAILED: {e}")

    print("  Generating hello_world...")
    try:
        audio, sr = fish.synthesize(HELLO_TEXT, language="en")
        sf.write(str(output_dir / "hello.wav"), audio, sr)
        results["hello"] = analyze_audio(audio, sr)
        time.sleep(0.5)
    except Exception as e:
        print(f"    FAILED: {e}")

    for lang_name, text in LANG_TEXTS.items():
        key = f"lang_{lang_name}"
        print(f"  Generating {key} ({LANG_CODES[lang_name]})...")
        try:
            audio, sr = fish.synthesize(text, language=LANG_CODES[lang_name])
            sf.write(str(output_dir / f"{key}.wav"), audio, sr)
            results[key] = analyze_audio(audio, sr)
            time.sleep(0.5)
        except Exception as e:
            print(f"    FAILED: {e}")

    fish.cleanup()
    return results


def apply_postprocessing(fish_dir: Path, output_dir: Path, config_name: str, pp_config) -> dict:
    """Apply post-processing to Fish Speech WAV files and return metrics."""
    from audiosmith.tts_postprocessor import TTSPostProcessor

    pp = TTSPostProcessor(config=pp_config)
    results = {}

    for wav_file in sorted(fish_dir.glob("*.wav")):
        wav, sr = sf.read(str(wav_file), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        key = wav_file.stem

        # Determine text and emotion for this file
        text = None
        emotion = None
        if key.startswith("emotion_"):
            emotion_name = key.replace("emotion_", "")
            text = EMOTION_TEXTS.get(emotion_name)
            emotion = EMOTION_META.get(emotion_name)
        elif key == "hello":
            text = HELLO_TEXT
        else:
            # Language test
            lang = key.replace("lang_", "")
            text = LANG_TEXTS.get(lang)

        try:
            processed = pp.process(wav, sr, text=text, emotion=emotion)
            sf.write(str(output_dir / f"{key}.wav"), processed, sr)
            results[key] = analyze_audio(processed, sr)
        except Exception as e:
            print(f"    PP[{config_name}] failed for {key}: {e}")
            results[key] = analyze_audio(wav, sr)  # fallback to raw

    return results


def print_multi_comparison(el_metrics: dict, variants: dict):
    """Print comparison of ElevenLabs vs multiple Fish Speech variants."""

    emotion_keys = ["emotion_neutral", "emotion_angry", "emotion_sad",
                    "emotion_excited", "emotion_whisper"]
    lang_keys = sorted([k for k in el_metrics if k.startswith("lang_")
                        and any(k in v for v in variants.values())])
    core_metrics = ["dynamic_range_db", "silence_pct", "spectral_centroid_hz",
                    "rms", "energy_variance"]

    variant_names = list(variants.keys())
    col_width = 14

    # ---- Emotion detail table ----
    print(f"\n{'='*120}")
    print(f" EMOTION TESTS: ElevenLabs vs Fish Speech Variants")
    print(f"{'='*120}")

    header = f"{'Emotion':<12} {'Metric':<22} {'ElevenLabs':>{col_width}}"
    for vn in variant_names:
        header += f" {vn:>{col_width}}"
    print(header)
    print("-" * (36 + col_width + col_width * len(variant_names) + len(variant_names)))

    for ek in emotion_keys:
        el = el_metrics.get(ek, {})
        if not el:
            continue
        for i, metric in enumerate(core_metrics):
            el_val = el.get(metric, 0)
            label = ek.replace("emotion_", "") if i == 0 else ""
            row = f"{label:<12} {metric:<22} {el_val:>{col_width}.4f}"
            for vn in variant_names:
                fs_val = variants[vn].get(ek, {}).get(metric, 0)
                pct = ((fs_val - el_val) / el_val * 100) if el_val else 0
                row += f" {fs_val:>{col_width - 8}.4f}({pct:>+.0f}%)"
            print(row)
        print()

    # ---- Summary averages ----
    print(f"\n{'='*120}")
    print(f" SUMMARY AVERAGES (Emotions)")
    print(f"{'='*120}")

    header = f"{'Metric':<25} {'ElevenLabs':>{col_width}}"
    for vn in variant_names:
        header += f" {vn:>{col_width + 6}}"
    print(header)
    print("-" * (25 + col_width + (col_width + 6) * len(variant_names)))

    for metric in core_metrics + ["zcr", "duration_s"]:
        el_vals = [el_metrics[k].get(metric, 0) for k in emotion_keys if k in el_metrics]
        if not el_vals:
            continue
        el_avg = sum(el_vals) / len(el_vals)

        row = f"{metric:<25} {el_avg:>{col_width}.4f}"
        for vn in variant_names:
            fs_vals = [variants[vn].get(k, {}).get(metric, 0) for k in emotion_keys if k in variants[vn]]
            if fs_vals:
                fs_avg = sum(fs_vals) / len(fs_vals)
                pct = ((fs_avg - el_avg) / el_avg * 100) if el_avg else 0
                row += f" {fs_avg:>{col_width}.4f} ({pct:>+6.1f}%)"
            else:
                row += f" {'N/A':>{col_width + 6}}"
        print(row)

    # ---- Language averages ----
    if lang_keys:
        print(f"\n{'='*120}")
        print(f" SUMMARY AVERAGES (Languages)")
        print(f"{'='*120}")

        header = f"{'Metric':<25} {'ElevenLabs':>{col_width}}"
        for vn in variant_names:
            header += f" {vn:>{col_width + 6}}"
        print(header)
        print("-" * (25 + col_width + (col_width + 6) * len(variant_names)))

        for metric in core_metrics + ["duration_s"]:
            el_vals = [el_metrics[k].get(metric, 0) for k in lang_keys if k in el_metrics]
            if not el_vals:
                continue
            el_avg = sum(el_vals) / len(el_vals)

            row = f"{metric:<25} {el_avg:>{col_width}.4f}"
            for vn in variant_names:
                fs_vals = [variants[vn].get(k, {}).get(metric, 0) for k in lang_keys if k in variants[vn]]
                if fs_vals:
                    fs_avg = sum(fs_vals) / len(fs_vals)
                    pct = ((fs_avg - el_avg) / el_avg * 100) if el_avg else 0
                    row += f" {fs_avg:>{col_width}.4f} ({pct:>+6.1f}%)"
                else:
                    row += f" {'N/A':>{col_width + 6}}"
            print(row)

    # ---- Scorecard ----
    print(f"\n{'='*120}")
    print(f" QUALITY SCORECARD (lower abs diff% = closer to ElevenLabs)")
    print(f"{'='*120}")

    score_metrics = ["dynamic_range_db", "silence_pct", "spectral_centroid_hz", "rms"]
    for vn in variant_names:
        total_diff = 0
        count = 0
        print(f"\n  [{vn}]")
        for metric in score_metrics:
            el_vals = [el_metrics[k].get(metric, 0) for k in emotion_keys if k in el_metrics]
            fs_vals = [variants[vn].get(k, {}).get(metric, 0) for k in emotion_keys if k in variants[vn]]
            if el_vals and fs_vals:
                el_avg = sum(el_vals) / len(el_vals)
                fs_avg = sum(fs_vals) / len(fs_vals)
                abs_pct = abs((fs_avg - el_avg) / el_avg * 100) if el_avg else 0
                total_diff += abs_pct
                count += 1
                grade = "A+" if abs_pct < 5 else "A" if abs_pct < 10 else "B" if abs_pct < 25 else "C" if abs_pct < 50 else "D"
                print(f"    {metric:<25} diff: {abs_pct:>6.1f}%  grade: {grade}")

        if count:
            avg_diff = total_diff / count
            overall = "A+" if avg_diff < 5 else "A" if avg_diff < 10 else "B" if avg_diff < 25 else "C" if avg_diff < 50 else "D"
            print(f"    {'OVERALL':<25} diff: {avg_diff:>6.1f}%  grade: {overall}")


def main():
    from audiosmith.tts_postprocessor import PostProcessConfig

    reuse = "--reuse" in sys.argv

    base_dir = Path("/mnt/g/Self_Projects/production/AudioSmith/output")
    el_dir = base_dir / "elevenlabs_tests"
    fish_dir = base_dir / "fish_tests"
    fish_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: ElevenLabs reference
    print("Analyzing ElevenLabs reference files...")
    el_metrics = analyze_existing_elevenlabs(el_dir)
    print(f"  Found {len(el_metrics)} ElevenLabs samples")

    # Step 2: Fish Speech raw
    print("\nFish Speech raw samples...")
    fish_raw = generate_fish_tests(fish_dir, reuse=reuse)
    print(f"  {len(fish_raw)} samples ready")

    # Step 3: Post-processing variants
    # ElevenLabs emotion averages: RMS ~0.14, spectral centroid ~2564 Hz
    configs = {
        "PP-dynamics": PostProcessConfig(
            enable_silence=False, enable_dynamics=True,
            enable_breath=False, enable_warmth=False,
            global_intensity=0.7,
        ),
        "Fish-opt": PostProcessConfig(
            enable_silence=False, enable_dynamics=True,
            enable_breath=False, enable_warmth=False,
            enable_normalize=True, target_rms=0.14,
            spectral_tilt=-0.6,
            global_intensity=0.7,
        ),
        "Fish-opt-v2": PostProcessConfig(
            enable_silence=False, enable_dynamics=True,
            enable_breath=True, enable_warmth=False,
            enable_normalize=True, target_rms=0.14,
            spectral_tilt=-0.8,
            global_intensity=0.7,
        ),
        "Fish-opt-v3": PostProcessConfig(
            enable_silence=False, enable_dynamics=True,
            enable_breath=True, enable_warmth=False,
            enable_normalize=True, target_rms=0.14,
            spectral_tilt=-1.0,
            global_intensity=0.8,
        ),
        "Fish-v5e": PostProcessConfig(
            enable_silence=False, enable_dynamics=True, enable_breath=True,
            enable_warmth=False, enable_spectral_matching=True,
            enable_micro_dynamics=False, enable_normalize=True,
            enable_silence_trim=True, max_silence_ms=100,
            target_rms_adaptive=True, spectral_intensity=0.5,
            global_intensity=0.7,
        ),
    }

    variants = {"Raw": fish_raw}

    for config_name, pp_config in configs.items():
        print(f"\nApplying {config_name}...")
        pp_dir = base_dir / f"fish_tests_{config_name.lower().replace('+', '_')}"
        pp_dir.mkdir(parents=True, exist_ok=True)
        variants[config_name] = apply_postprocessing(fish_dir, pp_dir, config_name, pp_config)
        print(f"  Processed {len(variants[config_name])} files")

    # Step 4: Multi-comparison
    print_multi_comparison(el_metrics, variants)

    # Step 5: Save metrics
    all_metrics = {"elevenlabs": el_metrics}
    all_metrics.update(variants)
    metrics_path = base_dir / "fish_vs_elevenlabs_comparison.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nFull metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
