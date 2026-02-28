"""Generate local TTS samples (with/without post-processing) and compare to ElevenLabs.

Generates matching test files for Piper engine, then runs audio analysis
to compare quality metrics (dynamic range, silence %, spectral centroid, RMS).
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Test sentences — same as ElevenLabs tests
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
}

HELLO_TEXT = "Hello world, this is a test of the text to speech engine."

def analyze_audio(wav: np.ndarray, sr: int) -> dict:
    """Compute quality metrics for an audio array."""
    if len(wav) == 0:
        return {}

    duration = len(wav) / sr
    rms = float(np.sqrt(np.mean(wav ** 2)))
    peak = float(np.max(np.abs(wav)))

    # Dynamic range (peak-to-RMS in dB)
    if rms > 1e-8:
        dynamic_range_db = 20 * np.log10(peak / rms)
    else:
        dynamic_range_db = 0.0

    # Silence percentage (frames with RMS < 0.01)
    frame_size = 512
    n_frames = len(wav) // frame_size
    silent_frames = 0
    for i in range(n_frames):
        frame = wav[i * frame_size : (i + 1) * frame_size]
        if np.sqrt(np.mean(frame ** 2)) < 0.01:
            silent_frames += 1
    silence_pct = (silent_frames / max(n_frames, 1)) * 100

    # Spectral centroid (via FFT)
    fft = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), d=1.0/sr)
    if fft.sum() > 1e-8:
        spectral_centroid = float(np.sum(freqs * fft) / np.sum(fft))
    else:
        spectral_centroid = 0.0

    # Energy variance (measure of expressiveness)
    frame_energies = []
    for i in range(n_frames):
        frame = wav[i * frame_size : (i + 1) * frame_size]
        frame_energies.append(np.mean(frame ** 2))
    energy_var = float(np.var(frame_energies)) if frame_energies else 0.0

    # Zero crossing rate
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
        metrics = analyze_audio(wav, sr)
        results[wav_file.stem] = metrics
    return results


def generate_piper_tests(output_dir: Path) -> dict:
    """Generate Piper TTS samples with and without post-processing."""
    from audiosmith.piper_tts import PiperTTS
    from audiosmith.tts_postprocessor import TTSPostProcessor

    results = {}
    pp = TTSPostProcessor()

    # English voice — needs model_path
    model_path = Path("/mnt/g/Self_Projects/production/AudioSmith/models/piper/en_US-lessac-medium.onnx")
    print(f"  Loading Piper ({model_path.stem})...")
    piper = PiperTTS(voice="en_US-lessac-medium", model_path=model_path)
    sr = piper.sample_rate

    # Emotion texts
    for emotion, text in EMOTION_TEXTS.items():
        print(f"  Generating emotion_{emotion}...")
        wav_raw = piper.synthesize(text)
        wav_pp = pp.process(wav_raw.copy(), sr, text=text,
                           emotion={"primary": emotion, "intensity": 0.8})

        # Save both versions
        sf.write(str(output_dir / f"emotion_{emotion}_raw.wav"), wav_raw, sr)
        sf.write(str(output_dir / f"emotion_{emotion}_pp.wav"), wav_pp, sr)

        results[f"emotion_{emotion}_raw"] = analyze_audio(wav_raw, sr)
        results[f"emotion_{emotion}_pp"] = analyze_audio(wav_pp, sr)

    # Hello world
    print("  Generating hello_world...")
    wav_raw = piper.synthesize(HELLO_TEXT)
    wav_pp = pp.process(wav_raw.copy(), sr, text=HELLO_TEXT)
    sf.write(str(output_dir / "hello_raw.wav"), wav_raw, sr)
    sf.write(str(output_dir / "hello_pp.wav"), wav_pp, sr)
    results["hello_raw"] = analyze_audio(wav_raw, sr)
    results["hello_pp"] = analyze_audio(wav_pp, sr)

    # Polish voice — no model available, skip
    print("  Polish voice skipped (no ONNX model available)")

    piper.cleanup()
    return results


def print_comparison(el_metrics: dict, local_metrics: dict, engine_name: str):
    """Print a side-by-side comparison table."""
    print(f"\n{'='*90}")
    print(f" COMPARISON: ElevenLabs vs {engine_name}")
    print(f"{'='*90}")

    # Group by test type
    groups = {
        "Emotions": ["emotion_neutral", "emotion_angry", "emotion_sad", "emotion_excited", "emotion_whisper"],
    }

    for group_name, keys in groups.items():
        print(f"\n--- {group_name} ---")
        print(f"{'Test':<25} {'Metric':<20} {'ElevenLabs':>12} {f'{engine_name} Raw':>14} {f'{engine_name} PP':>14} {'Gap':>8}")
        print("-" * 95)

        for key in keys:
            el_data = el_metrics.get(key, {})
            raw_data = local_metrics.get(f"{key}_raw", {})
            pp_data = local_metrics.get(f"{key}_pp", {})

            if not el_data:
                continue

            for metric in ["dynamic_range_db", "silence_pct", "spectral_centroid_hz", "rms"]:
                el_val = el_data.get(metric, 0)
                raw_val = raw_data.get(metric, 0)
                pp_val = pp_data.get(metric, 0)

                # Calculate improvement
                if el_val != 0:
                    gap_raw = ((raw_val - el_val) / el_val) * 100
                    gap_pp = ((pp_val - el_val) / el_val) * 100
                else:
                    gap_raw = gap_pp = 0

                label = key.replace("emotion_", "") if metric == "dynamic_range_db" else ""
                print(f"{label:<25} {metric:<20} {el_val:>12} {raw_val:>14} {pp_val:>14} {gap_pp:>+7.0f}%")
            print()

    # Summary averages
    print(f"\n--- AVERAGES (Emotion Tests) ---")
    print(f"{'Metric':<25} {'ElevenLabs':>12} {f'{engine_name} Raw':>14} {f'{engine_name} PP':>14} {'Improvement':>12}")
    print("-" * 80)

    for metric in ["dynamic_range_db", "silence_pct", "spectral_centroid_hz", "rms", "duration_s"]:
        el_vals = [el_metrics.get(k, {}).get(metric, 0) for k in groups["Emotions"] if k in el_metrics]
        raw_vals = [local_metrics.get(f"{k}_raw", {}).get(metric, 0) for k in groups["Emotions"] if f"{k}_raw" in local_metrics]
        pp_vals = [local_metrics.get(f"{k}_pp", {}).get(metric, 0) for k in groups["Emotions"] if f"{k}_pp" in local_metrics]

        if el_vals and raw_vals and pp_vals:
            el_avg = sum(el_vals) / len(el_vals)
            raw_avg = sum(raw_vals) / len(raw_vals)
            pp_avg = sum(pp_vals) / len(pp_vals)

            # How much did PP close the gap?
            if abs(el_avg - raw_avg) > 0.01:
                gap_closed = abs(pp_avg - raw_avg) / abs(el_avg - raw_avg) * 100
                direction = "closer" if abs(pp_avg - el_avg) < abs(raw_avg - el_avg) else "further"
            else:
                gap_closed = 0
                direction = "same"

            print(f"{metric:<25} {el_avg:>12.2f} {raw_avg:>14.2f} {pp_avg:>14.2f} {gap_closed:>8.0f}% {direction}")


def main():
    base_dir = Path("/mnt/g/Self_Projects/production/AudioSmith/output")
    el_dir = base_dir / "elevenlabs_tests"
    piper_dir = base_dir / "piper_tests"

    piper_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Analyze existing ElevenLabs files
    print("Analyzing ElevenLabs reference files...")
    el_metrics = analyze_existing_elevenlabs(el_dir)
    print(f"  Found {len(el_metrics)} ElevenLabs samples")

    # Step 2: Generate Piper tests
    print("\nGenerating Piper tests (raw + post-processed)...")
    piper_metrics = generate_piper_tests(piper_dir)
    print(f"  Generated {len(piper_metrics)} Piper samples")

    # Step 3: Compare
    print_comparison(el_metrics, piper_metrics, "Piper")

    # Step 4: Save full metrics as JSON
    all_metrics = {
        "elevenlabs": el_metrics,
        "piper": piper_metrics,
    }
    metrics_path = base_dir / "quality_comparison.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nFull metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
