"""Polish TTS Quality Comparison: Local engines vs ElevenLabs cloud.

Generates Polish speech from Fish Speech + ElevenLabs, applies Polish-specific
post-processing, measures spectral metrics, and compares quality gap.

Usage:
    python3 test_polish_quality.py              # full run (generate + compare)
    python3 test_polish_quality.py --reuse       # reuse existing WAV files
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

# Polish test sentences — varied complexity and phonetics
POLISH_TEXTS = {
    "greeting": "Dzień dobry, jak się masz? Mam nadzieję, że wszystko w porządku.",
    "sibilant": "Szczęśliwy szef wszedł przez drzwi i zaczął mówić o przyszłości.",
    "question": "Czy możesz mi powiedzieć, gdzie jest najbliższy sklep?",
    "emotional": "To niesamowite! Nigdy w życiu nie widziałem czegoś tak pięknego!",
    "narrative": "Stary zamek stał na wzgórzu, otoczony gęstym lasem sosnowym.",
    "formal": "Szanowni Państwo, pragnę przedstawić wyniki naszych badań naukowych.",
}

POLISH_EMOTIONS = {
    "greeting": {"primary": "neutral", "intensity": 0.7},
    "sibilant": {"primary": "happy", "intensity": 0.6},
    "question": {"primary": "neutral", "intensity": 0.5},
    "emotional": {"primary": "excited", "intensity": 0.9},
    "narrative": {"primary": "neutral", "intensity": 0.7},
    "formal": {"primary": "neutral", "intensity": 0.5},
}

# Fish Speech emotion markers for Polish
FISH_EMOTIONS = {
    "greeting": None,
    "sibilant": "happy",
    "question": None,
    "emotional": "excited",
    "narrative": None,
    "formal": None,
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

    # Spectral centroid (capped at 12kHz for fair comparison)
    fft_mag = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)
    cap = min(12000.0, sr / 2.0)
    cap_mask = freqs <= cap
    capped_mag = fft_mag.copy()
    capped_mag[~cap_mask] = 0.0

    if capped_mag.sum() > 1e-8:
        spectral_centroid = float(np.sum(freqs * capped_mag) / np.sum(capped_mag))
    else:
        spectral_centroid = 0.0

    # Brightness (energy above 2kHz / total)
    total_e = np.sum(capped_mag ** 2)
    bright_mask = (freqs >= 2000) & cap_mask
    brightness = float(np.sum(capped_mag[bright_mask] ** 2) / total_e) if total_e > 1e-8 else 0.0

    zcr = float(np.mean(np.abs(np.diff(np.sign(wav))) > 0))

    # Per-band energy (9 bands matching spectral_profiles.py)
    band_edges = [0, 150, 350, 700, 1400, 2800, 4000, 6000, 8000, cap]
    bands_db = []
    band_energies = []
    for i in range(len(band_edges) - 1):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        be = np.sum(capped_mag[mask] ** 2) if np.any(mask) else 1e-16
        band_energies.append(be)
    max_be = max(band_energies) if band_energies else 1e-16
    bands_db = [round(10 * np.log10(e / max_be), 1) if e > 1e-16 else -60.0 for e in band_energies]

    return {
        "duration_s": round(duration, 2),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "silence_pct": round(silence_pct, 1),
        "spectral_centroid_hz": round(spectral_centroid, 0),
        "brightness": round(brightness, 4),
        "energy_variance": round(energy_var, 6),
        "zcr": round(zcr, 4),
        "bands_db": bands_db,
    }


def generate_elevenlabs_polish(output_dir: Path, reuse: bool = False) -> dict:
    """Generate Polish ElevenLabs reference samples."""
    results = {}

    if reuse:
        for wav_file in sorted(output_dir.glob("*.wav")):
            wav, sr = sf.read(str(wav_file), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            results[wav_file.stem] = analyze_audio(wav, sr)
        return results

    from audiosmith.elevenlabs_tts import ElevenLabsTTS
    el = ElevenLabsTTS()

    for key, text in POLISH_TEXTS.items():
        print(f"  EL: {key}...")
        try:
            audio, sr = el.synthesize(text)
            sf.write(str(output_dir / f"{key}.wav"), audio, sr)
            results[key] = analyze_audio(audio, sr)
            time.sleep(0.5)
        except Exception as e:
            print(f"    FAILED: {e}")

    el.cleanup()
    return results


def generate_fish_polish(output_dir: Path, reuse: bool = False) -> dict:
    """Generate Polish Fish Speech samples (with emotion markers)."""
    results = {}

    if reuse:
        for wav_file in sorted(output_dir.glob("*.wav")):
            wav, sr = sf.read(str(wav_file), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            results[wav_file.stem] = analyze_audio(wav, sr)
        return results

    from audiosmith.fish_speech_tts import FishSpeechTTS
    fish = FishSpeechTTS()

    for key, text in POLISH_TEXTS.items():
        emotion = FISH_EMOTIONS.get(key)
        label = f"{key} (emotion={emotion})" if emotion else key
        print(f"  Fish: {label}...")
        try:
            audio, sr = fish.synthesize(text, language="pl", emotion=emotion)
            sf.write(str(output_dir / f"{key}.wav"), audio, sr)
            results[key] = analyze_audio(audio, sr)
            time.sleep(0.5)
        except Exception as e:
            print(f"    FAILED: {e}")

    fish.cleanup()
    return results


def apply_postprocessing(
    source_dir: Path, output_dir: Path, pp_config, language: str = "pl",
) -> dict:
    """Apply post-processing to WAV files and return metrics."""
    from audiosmith.tts_postprocessor import TTSPostProcessor
    pp = TTSPostProcessor(config=pp_config)
    results = {}

    for wav_file in sorted(source_dir.glob("*.wav")):
        wav, sr = sf.read(str(wav_file), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        key = wav_file.stem
        text = POLISH_TEXTS.get(key, "")
        emotion = POLISH_EMOTIONS.get(key)

        try:
            processed = pp.process(wav, sr, text=text, emotion=emotion, language=language)
            sf.write(str(output_dir / f"{key}.wav"), processed, sr)
            results[key] = analyze_audio(processed, sr)
        except Exception as e:
            print(f"    PP failed for {key}: {e}")
            results[key] = analyze_audio(wav, sr)

    return results


def compute_quality_score(engine_metrics: dict, ref_metrics: dict) -> Dict[str, Any]:
    """Compute per-metric and overall quality scores vs reference."""
    score_metrics = ["dynamic_range_db", "spectral_centroid_hz", "rms", "brightness", "zcr"]
    common_keys = set(engine_metrics.keys()) & set(ref_metrics.keys())

    if not common_keys:
        return {"overall_gap_pct": 100, "grade": "F", "per_metric": {}}

    per_metric = {}
    for metric in score_metrics:
        ref_vals = [ref_metrics[k].get(metric, 0) for k in common_keys]
        eng_vals = [engine_metrics[k].get(metric, 0) for k in common_keys]
        ref_avg = sum(ref_vals) / len(ref_vals) if ref_vals else 0
        eng_avg = sum(eng_vals) / len(eng_vals) if eng_vals else 0

        if ref_avg != 0:
            abs_gap = abs((eng_avg - ref_avg) / ref_avg * 100)
        else:
            abs_gap = 0

        grade = "A+" if abs_gap < 5 else "A" if abs_gap < 10 else "B" if abs_gap < 25 else "C" if abs_gap < 50 else "D"
        per_metric[metric] = {
            "ref_avg": round(ref_avg, 4),
            "eng_avg": round(eng_avg, 4),
            "gap_pct": round(abs_gap, 1),
            "grade": grade,
        }

    overall_gap = sum(m["gap_pct"] for m in per_metric.values()) / len(per_metric)
    overall_grade = "A+" if overall_gap < 5 else "A" if overall_gap < 10 else "B" if overall_gap < 25 else "C" if overall_gap < 50 else "D"

    return {
        "overall_gap_pct": round(overall_gap, 1),
        "grade": overall_grade,
        "per_metric": per_metric,
    }


def print_comparison(variants: Dict[str, dict], ref_metrics: dict):
    """Print detailed comparison table."""
    common_keys = sorted(set.intersection(*[set(v.keys()) for v in variants.values()] + [set(ref_metrics.keys())]))

    print(f"\n{'='*130}")
    print(f" POLISH TTS QUALITY COMPARISON vs ElevenLabs Reference")
    print(f"{'='*130}")

    variant_names = list(variants.keys())

    # Per-sample detail
    for key in common_keys:
        ref = ref_metrics.get(key, {})
        if not ref:
            continue
        text = POLISH_TEXTS.get(key, "")[:50]
        print(f"\n  [{key}] \"{text}...\"")

        header = f"    {'Metric':<25} {'ElevenLabs':>12}"
        for vn in variant_names:
            header += f"  {vn:>18}"
        print(header)
        print("    " + "-" * (37 + 20 * len(variant_names)))

        for metric in ["spectral_centroid_hz", "brightness", "dynamic_range_db", "rms", "zcr", "silence_pct"]:
            ref_val = ref.get(metric, 0)
            row = f"    {metric:<25} {ref_val:>12.4f}"
            for vn in variant_names:
                eng_val = variants[vn].get(key, {}).get(metric, 0)
                if ref_val != 0:
                    gap = ((eng_val - ref_val) / ref_val * 100)
                else:
                    gap = 0
                row += f"  {eng_val:>10.4f}({gap:>+5.0f}%)"
            print(row)

    # Band-level comparison for first sample
    first_key = common_keys[0] if common_keys else None
    if first_key:
        band_names = [
            "0-150Hz sub-bass", "150-350Hz bass", "350-700Hz low-mid",
            "700-1.4kHz mid", "1.4-2.8kHz upper-mid", "2.8-4kHz presence",
            "4-6kHz brilliance", "6-8kHz air", "8kHz+ sparkle",
        ]
        print(f"\n  BAND ANALYSIS [{first_key}]")
        header = f"    {'Band':<25} {'ElevenLabs':>12}"
        for vn in variant_names:
            header += f"  {vn:>18}"
        print(header)
        print("    " + "-" * (37 + 20 * len(variant_names)))

        ref_bands = ref_metrics[first_key].get("bands_db", [])
        for i, name in enumerate(band_names):
            if i >= len(ref_bands):
                break
            ref_val = ref_bands[i]
            row = f"    {name:<25} {ref_val:>12.1f} dB"
            for vn in variant_names:
                eng_bands = variants[vn].get(first_key, {}).get("bands_db", [])
                eng_val = eng_bands[i] if i < len(eng_bands) else -60
                diff = eng_val - ref_val
                row += f"  {eng_val:>10.1f}({diff:>+5.1f}dB)"
            print(row)

    # Scorecard
    print(f"\n{'='*130}")
    print(f" QUALITY SCORECARD")
    print(f"{'='*130}")

    for vn in variant_names:
        score = compute_quality_score(variants[vn], ref_metrics)
        print(f"\n  [{vn}] Overall: {score['overall_gap_pct']:.1f}% gap — Grade: {score['grade']}")
        for metric, data in score["per_metric"].items():
            print(f"    {metric:<25} ref={data['ref_avg']:.4f}  eng={data['eng_avg']:.4f}  gap={data['gap_pct']:.1f}%  {data['grade']}")


def generate_chatterbox_polish(output_dir: Path, reuse: bool = False) -> dict:
    """Generate Polish Chatterbox samples (local GPU)."""
    results = {}

    if reuse:
        wav_files = sorted(output_dir.glob("*.wav"))
        if wav_files:
            for wav_file in wav_files:
                wav, sr = sf.read(str(wav_file), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                results[wav_file.stem] = analyze_audio(wav, sr)
            return results
        print("  No existing WAV files, generating fresh...")

    from audiosmith.tts import ChatterboxTTS
    cb = ChatterboxTTS(device="cuda")
    cb.load_model()
    sr = cb.sample_rate

    for key, text in POLISH_TEXTS.items():
        print(f"  CB: {key}...")
        try:
            audio = cb.synthesize(text, language="pl")
            sf.write(str(output_dir / f"{key}.wav"), audio, sr)
            results[key] = analyze_audio(audio, sr)
        except Exception as e:
            print(f"    FAILED: {e}")

    cb.cleanup()
    return results


def main():
    from audiosmith.tts_postprocessor import PostProcessConfig

    reuse = "--reuse" in sys.argv
    skip_cloud = "--local-only" in sys.argv
    base_dir = Path("/mnt/g/Self_Projects/production/AudioSmith/output")

    el_polish_dir = base_dir / "polish_elevenlabs"
    fish_polish_dir = base_dir / "polish_fish_raw"
    fish_pp_en_dir = base_dir / "polish_fish_pp_english"
    fish_pp_pl_dir = base_dir / "polish_fish_pp_polish"
    cb_polish_dir = base_dir / "polish_chatterbox_raw"
    cb_pp_en_dir = base_dir / "polish_chatterbox_pp_english"
    cb_pp_pl_dir = base_dir / "polish_chatterbox_pp_polish"

    for d in [el_polish_dir, fish_polish_dir, fish_pp_en_dir, fish_pp_pl_dir,
              cb_polish_dir, cb_pp_en_dir, cb_pp_pl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: ElevenLabs Polish reference
    print("Step 1: ElevenLabs Polish reference...")
    el_metrics = generate_elevenlabs_polish(el_polish_dir, reuse=reuse)
    print(f"  {len(el_metrics)} samples")

    # Step 2: Fish Speech Polish (raw, with emotion markers)
    if not skip_cloud:
        print("\nStep 2: Fish Speech Polish (raw + emotion markers)...")
        fish_raw = generate_fish_polish(fish_polish_dir, reuse=reuse)
        print(f"  {len(fish_raw)} samples")
    else:
        print("\nStep 2: Fish Speech (skipped, --local-only)...")
        fish_raw = {}
        if any(fish_polish_dir.glob("*.wav")):
            fish_raw = generate_fish_polish(fish_polish_dir, reuse=True)

    # Step 3: Fish + English-calibrated post-processing
    pp_english = PostProcessConfig(
        enable_silence=False, enable_dynamics=True, enable_breath=True,
        enable_warmth=False, enable_spectral_matching=True,
        enable_micro_dynamics=False, enable_normalize=True,
        enable_silence_trim=True, max_silence_ms=100,
        target_rms_adaptive=True, spectral_intensity=0.5,
        global_intensity=0.7,
    )
    # Polish-enhanced PP (shared across engines)
    pp_polish = PostProcessConfig(
        enable_silence=False, enable_dynamics=True, enable_breath=False,
        enable_warmth=False, enable_spectral_matching=True,
        enable_micro_dynamics=False, enable_normalize=True,
        enable_silence_trim=True, max_silence_ms=100,
        target_rms_adaptive=False, target_rms=0.13,
        spectral_intensity=0.3,
        global_intensity=0.7,
        language="pl",
    )
    # Chatterbox-specific PP (engine preset: warmth, micro-dynamics, no silence trim)
    pp_chatterbox_en = PostProcessConfig(
        enable_silence=True, enable_dynamics=True, enable_breath=True,
        enable_warmth=True, enable_spectral_matching=True,
        enable_micro_dynamics=True, spectral_intensity=0.6,
        global_intensity=0.7,
    )
    pp_chatterbox_pl = PostProcessConfig(
        enable_silence=True, enable_dynamics=True, enable_breath=False,
        enable_warmth=False, enable_spectral_matching=True,
        enable_micro_dynamics=True, enable_normalize=True,
        target_rms_adaptive=False, target_rms=0.13,
        spectral_intensity=0.3,
        global_intensity=0.7,
        language="pl",
    )

    if fish_raw:
        print("\nStep 3: Fish + English-only post-processing...")
        fish_pp_en = apply_postprocessing(fish_polish_dir, fish_pp_en_dir, pp_english, language=None)
        print(f"  {len(fish_pp_en)} samples")

        print("\nStep 4: Fish + Polish-enhanced post-processing...")
        fish_pp_pl = apply_postprocessing(fish_polish_dir, fish_pp_pl_dir, pp_polish, language="pl")
        print(f"  {len(fish_pp_pl)} samples")
    else:
        fish_pp_en, fish_pp_pl = {}, {}

    # Step 5: Chatterbox Polish (local GPU)
    print("\nStep 5: Chatterbox Polish (local GPU)...")
    cb_raw = generate_chatterbox_polish(cb_polish_dir, reuse=reuse)
    print(f"  {len(cb_raw)} samples")

    if cb_raw:
        print("\nStep 6: Chatterbox + English PP...")
        cb_pp_en = apply_postprocessing(cb_polish_dir, cb_pp_en_dir, pp_chatterbox_en, language=None)
        print(f"  {len(cb_pp_en)} samples")

        print("\nStep 7: Chatterbox + Polish PP...")
        cb_pp_pl = apply_postprocessing(cb_polish_dir, cb_pp_pl_dir, pp_chatterbox_pl, language="pl")
        print(f"  {len(cb_pp_pl)} samples")
    else:
        cb_pp_en, cb_pp_pl = {}, {}

    # Step 8: Compare all variants
    variants = {}
    if fish_raw:
        variants["Fish Raw"] = fish_raw
        variants["Fish+EN-PP"] = fish_pp_en
        variants["Fish+PL-PP"] = fish_pp_pl
    if cb_raw:
        variants["CB Raw"] = cb_raw
        variants["CB+EN-PP"] = cb_pp_en
        variants["CB+PL-PP"] = cb_pp_pl
    print_comparison(variants, el_metrics)

    # Step 9: Save full metrics
    all_data = {
        "elevenlabs": el_metrics,
        "scores": {
            vn: compute_quality_score(v, el_metrics) for vn, v in variants.items()
        },
    }
    if fish_raw:
        all_data.update({"fish_raw": fish_raw, "fish_pp_english": fish_pp_en, "fish_pp_polish": fish_pp_pl})
    if cb_raw:
        all_data.update({"chatterbox_raw": cb_raw, "chatterbox_pp_english": cb_pp_en, "chatterbox_pp_polish": cb_pp_pl})

    metrics_path = base_dir / "polish_quality_comparison.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            return super().default(obj)

    with open(metrics_path, "w") as f:
        json.dump(all_data, f, indent=2, cls=NumpyEncoder)
    print(f"\nFull metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
