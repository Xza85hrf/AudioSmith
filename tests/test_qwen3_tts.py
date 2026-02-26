"""Tests for audiosmith.qwen3_tts module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from audiosmith.qwen3_tts import (
    Qwen3TTS, VoiceProfile, PREMIUM_VOICES, SUPPORTED_LANGUAGES,
    MODEL_VARIANTS, detect_language, estimate_synthesis_duration,
    _normalize_language, _LANGUAGE_MAP,
)
from audiosmith.exceptions import TTSError


class TestConstants:
    def test_premium_voices_count(self):
        assert len(PREMIUM_VOICES) == 9

    def test_premium_voices_has_ryan(self):
        assert "Ryan" in PREMIUM_VOICES
        assert PREMIUM_VOICES["Ryan"]["language"] == "English"

    def test_supported_languages_count(self):
        assert len(SUPPORTED_LANGUAGES) == 10

    def test_supported_languages_contains_key_langs(self):
        for lang in ["English", "Chinese", "Japanese", "Korean"]:
            assert lang in SUPPORTED_LANGUAGES

    def test_model_variants_count(self):
        assert len(MODEL_VARIANTS) == 3
        assert "base" in MODEL_VARIANTS
        assert "voice_design" in MODEL_VARIANTS
        assert "custom_voice" in MODEL_VARIANTS


class TestVoiceProfile:
    def test_defaults(self):
        p = VoiceProfile(name="test", voice_type="cloned")
        assert p.name == "test"
        assert p.voice_type == "cloned"
        assert p.language == "English"
        assert p.description == ""
        assert p.ref_audio_path is None
        assert p.ref_text is None
        assert p.voice_clone_prompt is None
        assert p.instruct is None
        assert isinstance(p.created_at, float)
        assert isinstance(p.last_used, float)

    def test_custom_values(self):
        p = VoiceProfile(
            name="my_voice", voice_type="designed",
            language="Chinese", description="A warm voice",
            instruct="Male, 30 years old",
        )
        assert p.language == "Chinese"
        assert p.instruct == "Male, 30 years old"


class TestQwen3TTS:
    def test_init_defaults(self):
        tts = Qwen3TTS()
        assert tts.device_str == "auto"
        assert tts.use_flash_attention is True
        assert tts.dtype_str == "bfloat16"
        assert tts.sample_rate == 24000
        assert tts.initialized is False
        assert tts.max_voice_cache == 10
        assert tts._base_model is None
        assert tts._active_model_type is None

    def test_init_custom(self):
        tts = Qwen3TTS(device="cpu", dtype="float32", max_voice_cache=5)
        assert tts.device_str == "cpu"
        assert tts.dtype_str == "float32"
        assert tts.max_voice_cache == 5

    def test_add_voice_profile_lru_eviction(self):
        tts = Qwen3TTS(max_voice_cache=2)
        p1 = VoiceProfile(name="v1", voice_type="cloned")
        p2 = VoiceProfile(name="v2", voice_type="cloned")
        p3 = VoiceProfile(name="v3", voice_type="cloned")
        tts._add_voice_profile(p1)
        tts._add_voice_profile(p2)
        tts._add_voice_profile(p3)
        assert "v1" not in tts._voice_profiles
        assert "v2" in tts._voice_profiles
        assert "v3" in tts._voice_profiles

    def test_add_voice_profile_update_existing(self):
        tts = Qwen3TTS(max_voice_cache=3)
        p1 = VoiceProfile(name="v1", voice_type="cloned")
        tts._add_voice_profile(p1)
        p1_updated = VoiceProfile(name="v1", voice_type="designed")
        tts._add_voice_profile(p1_updated)
        assert tts._voice_profiles["v1"].voice_type == "designed"
        assert len(tts._voice_profiles) == 1

    def test_get_voice_profile_updates_lru(self):
        tts = Qwen3TTS(max_voice_cache=2)
        p1 = VoiceProfile(name="v1", voice_type="cloned")
        p2 = VoiceProfile(name="v2", voice_type="cloned")
        tts._add_voice_profile(p1)
        tts._add_voice_profile(p2)
        # Access v1 to move it to end (most recent)
        result = tts._get_voice_profile("v1")
        assert result is not None
        assert result.name == "v1"
        # Now add v3 — v2 should be evicted (oldest), not v1
        p3 = VoiceProfile(name="v3", voice_type="cloned")
        tts._add_voice_profile(p3)
        assert "v1" in tts._voice_profiles
        assert "v2" not in tts._voice_profiles

    def test_get_voice_profile_missing(self):
        tts = Qwen3TTS()
        assert tts._get_voice_profile("nonexistent") is None

    def test_make_cache_key_consistent(self):
        tts = Qwen3TTS()
        k1 = tts._make_cache_key("Hello", "Ryan", "English", None)
        k2 = tts._make_cache_key("Hello", "Ryan", "English", None)
        assert k1 == k2

    def test_make_cache_key_differs(self):
        tts = Qwen3TTS()
        k1 = tts._make_cache_key("Hello", "Ryan", "English", None)
        k2 = tts._make_cache_key("World", "Ryan", "English", None)
        assert k1 != k2

    def test_cache_audio_eviction(self):
        tts = Qwen3TTS()
        for i in range(55):
            tts._cache_audio(f"key_{i}", np.array([float(i)]), 12000)
        assert len(tts._audio_cache) == 50
        assert "key_0" not in tts._audio_cache
        assert "key_54" in tts._audio_cache

    def test_cleanup(self):
        tts = Qwen3TTS()
        tts._base_model = MagicMock()
        tts._design_model = MagicMock()
        tts._custom_model = MagicMock()
        tts._active_model_type = "base"
        tts.initialized = True
        p = VoiceProfile(name="v1", voice_type="cloned")
        tts._add_voice_profile(p)
        tts._audio_cache["k"] = (np.array([1.0]), 12000)

        tts.cleanup()

        assert tts._base_model is None
        assert tts._design_model is None
        assert tts._custom_model is None
        assert tts._active_model_type is None
        assert len(tts._voice_profiles) == 0
        assert len(tts._audio_cache) == 0
        assert tts.initialized is False

    def test_context_manager(self):
        with patch.object(Qwen3TTS, "cleanup") as mock_cleanup:
            with Qwen3TTS() as tts:
                assert isinstance(tts, Qwen3TTS)
            mock_cleanup.assert_called_once()

    def test_get_available_voices_premium(self):
        tts = Qwen3TTS()
        voices = tts.get_available_voices()
        assert "Ryan" in voices
        assert voices["Ryan"]["type"] == "premium"
        assert len(voices) == len(PREMIUM_VOICES)

    def test_get_available_voices_with_custom(self):
        tts = Qwen3TTS()
        p = VoiceProfile(name="my_clone", voice_type="cloned", language="English")
        tts._add_voice_profile(p)
        voices = tts.get_available_voices()
        assert "my_clone" in voices
        assert voices["my_clone"]["type"] == "cloned"
        assert len(voices) == len(PREMIUM_VOICES) + 1

    def test_load_model_unknown_type(self):
        tts = Qwen3TTS()
        with pytest.raises(TTSError, match="Unknown model type"):
            tts.load_model("nonexistent")

    def test_synthesize_unknown_voice(self):
        tts = Qwen3TTS()
        with pytest.raises(TTSError, match="Unknown voice"):
            tts.synthesize("Hello", voice="FakeVoice")

    def test_synthesize_cache_hit(self):
        tts = Qwen3TTS()
        cached_audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        key = tts._make_cache_key("Hi", "Ryan", None, None)
        tts._audio_cache[key] = (cached_audio, 12000)
        audio, sr = tts.synthesize("Hi", voice="Ryan")
        np.testing.assert_array_equal(audio, cached_audio)
        assert sr == 12000


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("Hello world") == "English"

    def test_chinese(self):
        assert detect_language("你好世界") == "Chinese"

    def test_japanese(self):
        assert detect_language("こんにちは世界") == "Japanese"

    def test_korean(self):
        assert detect_language("안녕하세요") == "Korean"

    def test_russian(self):
        assert detect_language("Привет мир") == "Russian"

    def test_empty(self):
        assert detect_language("") == "English"

    def test_mixed_defaults_english(self):
        assert detect_language("Hello world test sentence") == "English"


class TestNormalizeLanguage:
    def test_iso_code_en(self):
        assert _normalize_language("en") == "english"

    def test_iso_code_zh(self):
        assert _normalize_language("zh") == "chinese"

    def test_full_name_lowercase(self):
        assert _normalize_language("english") == "english"

    def test_full_name_capitalized(self):
        assert _normalize_language("English") == "english"

    def test_full_name_uppercase(self):
        assert _normalize_language("JAPANESE") == "japanese"

    def test_iso_polish(self):
        assert _normalize_language("pl") == "auto"

    def test_auto(self):
        assert _normalize_language("auto") == "auto"

    def test_unsupported_raises(self):
        with pytest.raises(TTSError, match="Unsupported language"):
            _normalize_language("xx")


class TestEstimateSynthesisDuration:
    def test_basic(self):
        assert estimate_synthesis_duration("one two three four five") == 2.0

    def test_custom_wps(self):
        assert estimate_synthesis_duration("one two three", words_per_second=3.0) == 1.0

    def test_empty(self):
        assert estimate_synthesis_duration("") == 0.0

    def test_single_word(self):
        assert estimate_synthesis_duration("hello") == pytest.approx(0.4)
