"""Tests for audiosmith.tts_postprocessor module."""

import numpy as np
import pytest

from audiosmith.exceptions import TTSError
from audiosmith.tts_postprocessor import (
    PostProcessConfig,
    TTSPostProcessor,
    _add_breath_noise,
    _boost_warmth,
    _expand_dynamic_range,
    _inject_silence,
    _pink_noise,
)

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
        assert len(result) > len(wav) + SR * 0.4  # At least 400ms total added


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
