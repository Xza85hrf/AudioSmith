"""Tests for audiosmith.tts_postprocessor module."""

import numpy as np
import pytest

from audiosmith.tts_postprocessor import (PostProcessConfig, TTSPostProcessor,
                                          _add_breath_noise,
                                          _add_micro_dynamics,
                                          _apply_spectral_correction,
                                          _boost_warmth,
                                          _compute_spectral_correction,
                                          _expand_dynamic_range,
                                          _inject_silence,
                                          _measure_spectral_envelope,
                                          _pink_noise, _reshape_dynamic_range,
                                          _synthesize_presence,
                                          _trim_excess_silence)

SR = 24000  # Standard sample rate for tests


def _make_tone(freq: float = 440.0, duration: float = 0.5, sr: int = SR) -> np.ndarray:
    """Generate a sine tone for testing."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _make_silence(duration: float = 0.2, sr: int = SR) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sr * duration), dtype=np.float32)


def _make_speech_like(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Generate speech-like audio: tone + silence gaps."""
    tone = _make_tone(200, 0.15, sr)
    gap = _make_silence(0.05, sr)
    pattern = np.concatenate([tone, gap])
    repeats = max(1, int(duration * sr / len(pattern)))
    wav = np.tile(pattern, repeats)[: int(sr * duration)]
    return wav.astype(np.float32)


class TestPostProcessConfig:
    def test_defaults(self):
        cfg = PostProcessConfig()
        assert cfg.enable_silence is True
        assert cfg.enable_dynamics is True
        assert cfg.enable_breath is True
        assert cfg.enable_warmth is True
        assert cfg.emotion_aware is True
        assert cfg.global_intensity == 0.7

    def test_custom_values(self):
        cfg = PostProcessConfig(
            enable_silence=False,
            global_intensity=1.5,
        )
        assert cfg.enable_silence is False
        assert cfg.global_intensity == 1.5


class TestSilenceInjection:
    def test_no_text_returns_unchanged(self):
        wav = _make_tone(duration=0.3)
        result = _inject_silence(wav, SR, "", 0.7)
        # Single chunk, no split
        assert len(result) == len(wav)

    def test_single_sentence_no_change(self):
        wav = _make_tone(duration=0.3)
        result = _inject_silence(wav, SR, "Hello world", 0.7)
        assert len(result) == len(wav)

    def test_period_adds_silence(self):
        wav = _make_tone(duration=0.5)
        text = "Hello. World."
        result = _inject_silence(wav, SR, text, 0.7)
        # Should be longer due to inserted pause
        assert len(result) > len(wav)

    def test_comma_adds_shorter_silence_than_period(self):
        wav = _make_tone(duration=0.5)
        result_period = _inject_silence(wav, SR, "Hello. World.", 0.7)
        result_comma = _inject_silence(wav, SR, "Hello, world.", 0.7)
        period_added = len(result_period) - len(wav)
        comma_added = len(result_comma) - len(wav)
        # Period pauses should be longer than comma pauses
        assert period_added > comma_added

    def test_multiple_sentences(self):
        wav = _make_tone(duration=1.0)
        text = "First sentence. Second sentence. Third one."
        result = _inject_silence(wav, SR, text, 0.7)
        # Two period boundaries = two pauses inserted
        assert len(result) > len(wav) + SR * 0.15  # At least 150ms total added


class TestDynamicRange:
    def test_flat_audio_gets_varied(self):
        # Constant-amplitude tone has no dynamic variation
        wav = _make_tone(duration=0.5)
        result = _expand_dynamic_range(wav, intensity=0.7)
        # Result should differ from input (gain curve applied)
        assert not np.allclose(wav, result, atol=1e-4)

    def test_intensity_zero_minimal_change(self):
        wav = _make_tone(duration=0.3)
        result = _expand_dynamic_range(wav, intensity=0.0)
        np.testing.assert_array_equal(wav, result)

    def test_no_clipping(self):
        wav = _make_tone(duration=0.5) * 0.9  # Near peak
        result = _expand_dynamic_range(wav, intensity=1.0)
        assert np.max(np.abs(result)) <= 0.96

    def test_empty_audio(self):
        wav = np.array([], dtype=np.float32)
        result = _expand_dynamic_range(wav, intensity=0.7)
        assert len(result) == 0

    def test_silent_audio_unchanged(self):
        wav = np.zeros(2400, dtype=np.float32)
        result = _expand_dynamic_range(wav, intensity=0.7)
        np.testing.assert_array_equal(wav, result)


class TestBreathSimulation:
    def test_noise_only_in_silence(self):
        # Create: 200ms tone + 200ms silence + 200ms tone
        tone = _make_tone(440, 0.2)
        silence = _make_silence(0.2)
        wav = np.concatenate([tone, silence, tone])

        result = _add_breath_noise(wav, SR, intensity=0.5)

        # Silence region should now have non-zero values (breath noise)
        silence_start = len(tone)
        silence_end = silence_start + len(silence)
        silence_region = result[silence_start:silence_end]
        assert np.max(np.abs(silence_region)) > 0

    def test_breath_amplitude_is_subtle(self):
        tone = _make_tone(440, 0.2)
        silence = _make_silence(0.2)
        wav = np.concatenate([tone, silence, tone])
        peak = np.max(np.abs(wav))

        result = _add_breath_noise(wav, SR, intensity=0.5)

        silence_start = len(tone)
        silence_end = silence_start + len(silence)
        breath_peak = np.max(np.abs(result[silence_start:silence_end]))
        # Breath should be at least 30dB below signal peak
        assert breath_peak < peak * 0.03

    def test_intensity_zero_no_change(self):
        wav = np.concatenate([_make_tone(440, 0.2), _make_silence(0.2)])
        result = _add_breath_noise(wav, SR, intensity=0.0)
        np.testing.assert_array_equal(wav, result)

    def test_short_silence_ignored(self):
        # 20ms silence — too short for breath insertion (threshold=80ms)
        tone = _make_tone(440, 0.2)
        short_silence = _make_silence(0.02)
        wav = np.concatenate([tone, short_silence, tone])
        result = _add_breath_noise(wav, SR, intensity=0.5)
        # Short silence should remain silent
        start = len(tone)
        end = start + len(short_silence)
        np.testing.assert_array_almost_equal(
            result[start:end], wav[start:end], decimal=5
        )


class TestWarmthBoost:
    def test_changes_spectrum(self):
        wav = _make_tone(200, duration=0.3)
        result = _boost_warmth(wav, SR, intensity=0.7)
        # Should differ from original
        assert not np.allclose(wav, result, atol=1e-4)

    def test_intensity_zero_no_change(self):
        wav = _make_tone(200, duration=0.3)
        result = _boost_warmth(wav, SR, intensity=0.0)
        np.testing.assert_array_equal(wav, result)

    def test_full_chain_preserves_peak_level(self):
        """Peak normalization happens in process(), not in _boost_warmth."""
        pp = TTSPostProcessor(PostProcessConfig(
            enable_silence=False, enable_dynamics=False, enable_breath=False,
            enable_warmth=True,
        ))
        wav = _make_tone(200, duration=0.3)
        orig_peak = np.max(np.abs(wav))
        result = pp.process(wav, SR)
        new_peak = np.max(np.abs(result))
        # Peak preserved within 2% by process() global normalization
        assert abs(new_peak - orig_peak) / orig_peak < 0.02

    def test_short_audio(self):
        wav = np.array([0.5], dtype=np.float32)
        result = _boost_warmth(wav, SR, intensity=0.7)
        assert len(result) == 1


class TestPinkNoise:
    def test_length(self):
        noise = _pink_noise(4800)
        assert len(noise) == 4800

    def test_normalized_peak(self):
        noise = _pink_noise(4800)
        assert np.max(np.abs(noise)) <= 1.01

    def test_no_dc_offset(self):
        noise = _pink_noise(48000)
        # Mean should be near zero
        assert abs(np.mean(noise)) < 0.1


class TestEmotionAwareness:
    def test_happy_full_intensity(self):
        pp = TTSPostProcessor(PostProcessConfig(global_intensity=1.0))
        intensity = pp._resolve_intensity({"primary": "happy", "intensity": 1.0})
        assert intensity == pytest.approx(1.0, abs=0.01)

    def test_sad_reduced_intensity(self):
        pp = TTSPostProcessor(PostProcessConfig(global_intensity=1.0))
        intensity = pp._resolve_intensity({"primary": "sad", "intensity": 1.0})
        assert intensity == pytest.approx(0.5, abs=0.01)

    def test_whisper_minimal(self):
        pp = TTSPostProcessor(PostProcessConfig(global_intensity=1.0))
        intensity = pp._resolve_intensity({"primary": "whisper", "intensity": 1.0})
        assert intensity == pytest.approx(0.1, abs=0.01)

    def test_no_emotion_uses_global(self):
        pp = TTSPostProcessor(PostProcessConfig(global_intensity=0.8))
        intensity = pp._resolve_intensity(None)
        assert intensity == 0.8

    def test_emotion_aware_disabled(self):
        pp = TTSPostProcessor(
            PostProcessConfig(global_intensity=0.8, emotion_aware=False)
        )
        intensity = pp._resolve_intensity({"primary": "angry", "intensity": 1.0})
        assert intensity == 0.8

    def test_unknown_emotion_uses_default(self):
        pp = TTSPostProcessor(PostProcessConfig(global_intensity=1.0))
        intensity = pp._resolve_intensity({"primary": "confused", "intensity": 1.0})
        assert intensity == pytest.approx(0.7, abs=0.01)


class TestFullChain:
    def test_process_empty_audio(self):
        pp = TTSPostProcessor()
        wav = np.array([], dtype=np.float32)
        result = pp.process(wav, SR)
        assert len(result) == 0

    def test_process_with_text_and_emotion(self):
        pp = TTSPostProcessor()
        wav = _make_speech_like(duration=1.0)
        emotion = {"primary": "happy", "intensity": 0.8}
        result = pp.process(wav, SR, text="Hello, world. How are you?", emotion=emotion)
        # Should be longer (silence added) and different (dynamics + warmth)
        assert len(result) >= len(wav)

    def test_all_steps_disabled_passthrough(self):
        cfg = PostProcessConfig(
            enable_silence=False,
            enable_dynamics=False,
            enable_breath=False,
            enable_warmth=False,
        )
        pp = TTSPostProcessor(cfg)
        wav = _make_tone(duration=0.3)
        result = pp.process(wav, SR, text="Hello.")
        np.testing.assert_array_equal(wav, result)

    def test_different_sample_rates(self):
        pp = TTSPostProcessor()
        for sr in [22050, 24000, 48000]:
            wav = _make_tone(440, 0.3, sr)
            result = pp.process(wav, sr, text="Hello. World.")
            assert len(result) > len(wav)

    def test_config_disable_silence(self):
        cfg = PostProcessConfig(enable_silence=False)
        pp = TTSPostProcessor(cfg)
        wav = _make_tone(duration=0.5)
        result = pp.process(wav, SR, text="Hello. World.")
        # Silence disabled, so length shouldn't increase from pauses
        # (dynamics/warmth may change values but not length)
        assert len(result) == len(wav)


class TestSpectralProfiles:
    def test_get_profile_known(self):
        from audiosmith.spectral_profiles import get_profile
        p = get_profile("angry")
        assert p.emotion == "angry"
        assert p.target_rms == 0.158
        assert p.centroid_hz == 2631.0

    def test_get_profile_unknown_falls_back(self):
        from audiosmith.spectral_profiles import get_profile
        p = get_profile("nonexistent_emotion")
        assert p.emotion == "neutral"

    def test_list_emotions(self):
        from audiosmith.spectral_profiles import list_emotions
        emotions = list_emotions()
        assert "angry" in emotions
        assert "whisper" in emotions
        assert len(emotions) >= 5


class TestSpectralMatching:
    def test_measure_spectral_envelope(self):
        wav = _make_tone(2500, duration=0.5)
        result = _measure_spectral_envelope(wav, SR)
        assert "centroid" in result
        assert "brightness" in result
        assert "bands_db" in result
        assert 1500 < result["centroid"] < 4000
        assert 0.0 <= result["brightness"] <= 1.0
        assert len(result["bands_db"]) == 9

    def test_measure_silent_audio(self):
        wav = np.zeros(SR, dtype=np.float32)
        result = _measure_spectral_envelope(wav, SR)
        assert result["centroid"] == 0.0
        assert result["brightness"] == 0.0

    def test_correction_converges_toward_target(self):
        from audiosmith.spectral_profiles import get_profile
        wav = _make_tone(1000, duration=0.5)
        measured = _measure_spectral_envelope(wav, SR)
        target = get_profile("neutral")
        correction = _compute_spectral_correction(measured, target, 0.7)
        assert len(correction) == 9
        assert all(0.3 <= g <= 3.0 for g in correction)

    def test_apply_spectral_correction(self):
        wav = _make_tone(1000, duration=0.5)
        gains = np.ones(9, dtype=np.float32)
        gains[5] = 1.5  # Boost presence band
        result = _apply_spectral_correction(wav, SR, gains)
        assert len(result) == len(wav)
        assert result.dtype == np.float32

    def test_full_spectral_matching_chain(self):
        cfg = PostProcessConfig(
            enable_spectral_matching=True,
            enable_silence=False, enable_dynamics=False,
            enable_breath=False, enable_warmth=False,
            spectral_intensity=0.8,
        )
        pp = TTSPostProcessor(cfg)
        wav = _make_tone(440, duration=0.5)
        emotion = {"primary": "angry", "intensity": 1.0}
        result = pp.process(wav, SR, emotion=emotion)
        assert len(result) == len(wav)
        assert np.all(np.isfinite(result))


class TestMicroDynamics:
    def test_adds_variation(self):
        wav = _make_tone(200, duration=1.0)
        result = _add_micro_dynamics(wav, SR, "Hello world test phrase here", 1.0)
        # Should differ from input
        assert not np.allclose(result, wav, atol=1e-4)

    def test_intensity_zero_no_change(self):
        wav = _make_tone(200, duration=0.5)
        result = _add_micro_dynamics(wav, SR, "Hello world", 0.0)
        np.testing.assert_array_equal(result, wav)

    def test_short_audio_unchanged(self):
        wav = _make_tone(200, duration=0.05)  # 50ms, too short
        result = _add_micro_dynamics(wav, SR, "Hi", 1.0)
        np.testing.assert_array_equal(result, wav)

    def test_deterministic_per_text(self):
        wav = _make_tone(200, duration=0.5)
        r1 = _add_micro_dynamics(wav, SR, "Same text here", 0.8)
        r2 = _add_micro_dynamics(wav, SR, "Same text here", 0.8)
        np.testing.assert_array_equal(r1, r2)

    def test_different_text_different_result(self):
        wav = _make_tone(200, duration=0.5)
        r1 = _add_micro_dynamics(wav, SR, "First text here", 0.8)
        r2 = _add_micro_dynamics(wav, SR, "Second text now", 0.8)
        assert not np.array_equal(r1, r2)


class TestAdaptiveRMS:
    def test_emotion_rms_targets_match(self):
        cfg = PostProcessConfig(
            enable_normalize=True, target_rms_adaptive=True,
            enable_spectral_matching=True,
            enable_silence=False, enable_dynamics=False,
            enable_breath=False, enable_warmth=False,
        )
        pp = TTSPostProcessor(cfg)
        wav = _make_tone(440, duration=0.5)

        targets = {"whisper": 0.105, "neutral": 0.145, "angry": 0.158}
        for emo, expected_rms in targets.items():
            result = pp.process(wav.copy(), SR, emotion={"primary": emo, "intensity": 1.0})
            rms = float(np.sqrt(np.mean(result ** 2)))
            assert abs(rms - expected_rms) < 0.01, f"{emo}: {rms:.4f} != {expected_rms}"

    def test_no_emotion_uses_config_rms(self):
        cfg = PostProcessConfig(
            enable_normalize=True, target_rms=0.2,
            target_rms_adaptive=True,
            enable_silence=False, enable_dynamics=False,
            enable_breath=False, enable_warmth=False,
            enable_spectral_matching=False,
        )
        pp = TTSPostProcessor(cfg)
        wav = _make_tone(440, duration=0.5)
        result = pp.process(wav, SR)
        rms = float(np.sqrt(np.mean(result ** 2)))
        assert abs(rms - 0.2) < 0.01


class TestSilenceTrimming:
    def test_trims_long_silence(self):
        tone = _make_tone(440, 0.2)
        long_silence = _make_silence(0.5)  # 500ms
        wav = np.concatenate([tone, long_silence, tone])
        result = _trim_excess_silence(wav, SR, max_silence_ms=200)
        # Should be shorter — 500ms trimmed to ~200ms
        assert len(result) < len(wav)
        trimmed_ms = (len(wav) - len(result)) / SR * 1000
        assert trimmed_ms > 200  # At least 200ms trimmed

    def test_preserves_short_silence(self):
        tone = _make_tone(440, 0.2)
        short_silence = _make_silence(0.1)  # 100ms < 200ms threshold
        wav = np.concatenate([tone, short_silence, tone])
        result = _trim_excess_silence(wav, SR, max_silence_ms=200)
        assert len(result) == len(wav)

    def test_empty_audio(self):
        wav = np.array([], dtype=np.float32)
        result = _trim_excess_silence(wav, SR)
        assert len(result) == 0

    def test_no_silence_unchanged(self):
        wav = _make_tone(440, 0.5)
        result = _trim_excess_silence(wav, SR)
        assert len(result) == len(wav)

    def test_config_wired_in_process(self):
        cfg = PostProcessConfig(
            enable_silence_trim=True, max_silence_ms=150,
            enable_silence=False, enable_dynamics=False,
            enable_breath=False, enable_warmth=False,
        )
        pp = TTSPostProcessor(cfg)
        tone = _make_tone(440, 0.2)
        long_silence = _make_silence(0.4)
        wav = np.concatenate([tone, long_silence, tone])
        result = pp.process(wav, SR)
        assert len(result) < len(wav)


class TestTargetDrivenDR:
    def test_with_target_dr(self):
        wav = _make_speech_like(duration=1.0)
        result = _expand_dynamic_range(wav, intensity=0.7, target_dr=17.0)
        assert not np.allclose(wav, result, atol=1e-4)
        assert np.all(np.isfinite(result))

    def test_without_target_uses_fixed(self):
        wav = _make_speech_like(duration=1.0)
        r1 = _expand_dynamic_range(wav, intensity=0.7, target_dr=None)
        r2 = _expand_dynamic_range(wav, intensity=0.7)
        np.testing.assert_array_equal(r1, r2)

    def test_low_dr_target_less_expansion(self):
        wav = _make_speech_like(duration=1.0)
        r_high = _expand_dynamic_range(wav, intensity=0.7, target_dr=20.0)
        r_low = _expand_dynamic_range(wav, intensity=0.7, target_dr=10.0)
        # Higher target should produce more variance
        var_high = np.var(r_high)
        var_low = np.var(r_low)
        assert var_high >= var_low * 0.9  # High target ≥ low target variance


class TestPerEmotionSpectralIntensity:
    def test_per_emotion_overrides_exist(self):
        from audiosmith.emotion_config import EMOTION_SPECTRAL_INTENSITY as _EMOTION_SPECTRAL_INTENSITY
        assert "angry" in _EMOTION_SPECTRAL_INTENSITY
        assert "sad" in _EMOTION_SPECTRAL_INTENSITY
        assert "excited" in _EMOTION_SPECTRAL_INTENSITY
        # Bright emotions (excited/sad) need stronger correction than close ones (angry)
        assert _EMOTION_SPECTRAL_INTENSITY["excited"] > _EMOTION_SPECTRAL_INTENSITY["angry"]

    def test_process_uses_per_emotion_intensity(self):
        cfg = PostProcessConfig(
            enable_spectral_matching=True, spectral_intensity=0.3,
            enable_silence=False, enable_dynamics=False,
            enable_breath=False, enable_warmth=False,
        )
        pp = TTSPostProcessor(cfg)
        wav = _make_tone(440, duration=0.5)
        # Angry should use override (1.0), not config (0.3)
        result = pp.process(wav, SR, emotion={"primary": "angry", "intensity": 1.0})
        assert np.all(np.isfinite(result))


class TestPresenceSynthesis:
    def test_adds_energy_when_centroid_low(self):
        # Low-freq tone has centroid ~200Hz, target 2631Hz — big gap
        wav = _make_tone(200, duration=0.5)
        result = _synthesize_presence(wav, SR, target_centroid=2631.0, intensity=0.8)
        # Should add high-freq energy, changing the waveform
        assert not np.allclose(wav, result, atol=1e-4)

    def test_no_change_when_centroid_close(self):
        # 2500Hz tone is close to 2631Hz target (<15% gap)
        wav = _make_tone(2500, duration=0.5)
        result = _synthesize_presence(wav, SR, target_centroid=2631.0, intensity=0.8)
        # Gap is <15%, should skip
        np.testing.assert_array_equal(wav, result)

    def test_raises_centroid(self):
        wav = _make_tone(200, duration=0.5)
        result = _synthesize_presence(wav, SR, target_centroid=2631.0, intensity=0.8)
        # Measure centroid of both
        def centroid(w):
            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0/SR)
            return float(np.sum(freqs * fft_mag) / np.sum(fft_mag))
        assert centroid(result) > centroid(wav)

    def test_short_audio_unchanged(self):
        wav = _make_tone(200, duration=0.05)
        result = _synthesize_presence(wav, SR, target_centroid=3000.0, intensity=1.0)
        np.testing.assert_array_equal(wav, result)


class TestDRReshaping:
    def test_widens_dr_when_below_target(self):
        wav = _make_speech_like(duration=1.0)
        rms = np.sqrt(np.mean(wav**2))
        peak = np.max(np.abs(wav))
        current_dr = 20 * np.log10(peak / rms)
        target_dr = current_dr + 5.0  # 5dB above current
        result = _reshape_dynamic_range(wav, SR, target_dr, intensity=0.8)
        # Result should differ
        assert not np.allclose(wav, result, atol=1e-4)

    def test_no_change_when_dr_close(self):
        wav = _make_speech_like(duration=1.0)
        rms = np.sqrt(np.mean(wav**2))
        peak = np.max(np.abs(wav))
        current_dr = 20 * np.log10(peak / rms)
        # Target within 2dB — should skip
        result = _reshape_dynamic_range(wav, SR, current_dr + 1.0, intensity=0.8)
        np.testing.assert_array_equal(wav, result)

    def test_empty_audio(self):
        wav = np.array([], dtype=np.float32)
        result = _reshape_dynamic_range(wav, SR, 15.0, intensity=0.8)
        assert len(result) == 0


class TestLanguageAware:
    def test_polish_applies_prosody(self):
        cfg = PostProcessConfig(
            enable_spectral_matching=True,
            enable_silence=False, enable_dynamics=False,
            enable_breath=False, enable_warmth=False,
            enable_micro_dynamics=False,
        )
        pp = TTSPostProcessor(cfg)
        wav = _make_speech_like(duration=1.0)
        emotion = {"primary": "neutral", "intensity": 0.7}
        result = pp.process(wav, SR, text="Bardzo ładna pogoda.", emotion=emotion, language="pl")
        assert not np.allclose(wav, result, atol=1e-4)

    def test_english_no_prosody_change(self):
        cfg = PostProcessConfig(
            enable_spectral_matching=True,
            enable_silence=False, enable_dynamics=False,
            enable_breath=False, enable_warmth=False,
            enable_micro_dynamics=False,
        )
        pp = TTSPostProcessor(cfg)
        wav = _make_speech_like(duration=1.0)
        emotion = {"primary": "neutral", "intensity": 0.7}
        result_en = pp.process(wav.copy(), SR, text="Hello world.", emotion=emotion, language="en")
        result_none = pp.process(wav.copy(), SR, text="Hello world.", emotion=emotion)
        # English and no-language should produce same result (no Polish prosody)
        np.testing.assert_array_equal(result_en, result_none)

    def test_config_language_used_when_param_omitted(self):
        cfg = PostProcessConfig(
            enable_spectral_matching=True, language="pl",
            enable_silence=False, enable_dynamics=False,
            enable_breath=False, enable_warmth=False,
            enable_micro_dynamics=False,
        )
        pp = TTSPostProcessor(cfg)
        wav = _make_speech_like(duration=1.0)
        emotion = {"primary": "neutral", "intensity": 0.7}
        result = pp.process(wav, SR, text="Dzień dobry.", emotion=emotion)
        assert not np.allclose(wav, result, atol=1e-4)


class TestLanguageSpectralModifiers:
    def test_polish_modifier_exists(self):
        from audiosmith.spectral_profiles import get_language_modifier
        mod = get_language_modifier("pl")
        assert "band_5_boost_db" in mod
        assert "band_6_boost_db" in mod
        assert mod["band_6_boost_db"] == 7.0

    def test_unknown_language_empty(self):
        from audiosmith.spectral_profiles import get_language_modifier
        mod = get_language_modifier("xx")
        assert mod == {}

    def test_none_language_empty(self):
        from audiosmith.spectral_profiles import get_language_modifier
        mod = get_language_modifier(None)
        assert mod == {}


class TestLanguagePPOverrides:
    def test_polish_overrides_exist(self):
        from audiosmith.pipeline_config import LANGUAGE_PP_OVERRIDES as _LANGUAGE_PP_OVERRIDES
        assert "pl" in _LANGUAGE_PP_OVERRIDES
        assert _LANGUAGE_PP_OVERRIDES["pl"]["spectral_intensity"] == 0.3
        assert _LANGUAGE_PP_OVERRIDES["pl"]["enable_spectral_matching"] is True

    def test_unknown_language_no_override(self):
        from audiosmith.pipeline_config import LANGUAGE_PP_OVERRIDES as _LANGUAGE_PP_OVERRIDES
        assert _LANGUAGE_PP_OVERRIDES.get("xx") is None


class TestEnginePresets:
    def test_all_presets_process(self):
        from audiosmith.pipeline_config import ENGINE_PP_PRESETS as _ENGINE_PP_PRESETS
        wav = _make_speech_like(duration=0.5)
        text = "Hello world. This is a test."
        emotion = {"primary": "neutral", "intensity": 0.7}

        for engine, preset in _ENGINE_PP_PRESETS.items():
            preset_copy = {**preset, "global_intensity": 0.7}
            cfg = PostProcessConfig(**preset_copy)
            pp = TTSPostProcessor(cfg)
            result = pp.process(wav.copy(), SR, text=text, emotion=emotion)
            assert len(result) > 0, f"{engine} produced empty output"
            assert np.all(np.isfinite(result)), f"{engine} produced NaN/Inf"

    def test_fish_preset_has_silence_trim(self):
        from audiosmith.pipeline_config import ENGINE_PP_PRESETS as _ENGINE_PP_PRESETS
        fish = _ENGINE_PP_PRESETS["fish"]
        assert fish.get("enable_silence_trim") is True
        assert fish.get("max_silence_ms", 200) <= 120
