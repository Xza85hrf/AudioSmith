"""Tests for audiosmith.language_data module."""

import pytest

from audiosmith.language_data import LanguageConfig, LANGUAGES, get_language


class TestLanguageConfig:
    """Test LanguageConfig dataclass."""

    def test_frozen_immutable(self):
        """Test that LanguageConfig is frozen and immutable."""
        config = LanguageConfig(code="test")
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            config.code = "modified"

    def test_default_question_starters(self):
        """Test default question_starters is empty frozenset."""
        config = LanguageConfig(code="xx")
        assert config.question_starters == frozenset()
        assert isinstance(config.question_starters, frozenset)

    def test_default_vowels(self):
        """Test default vowels is empty frozenset."""
        config = LanguageConfig(code="xx")
        assert config.vowels == frozenset()
        assert isinstance(config.vowels, frozenset)

    def test_default_stress_position(self):
        """Test default stress_position is 'variable'."""
        config = LanguageConfig(code="xx")
        assert config.stress_position == "variable"

    def test_default_syllable_timed(self):
        """Test default syllable_timed is False."""
        config = LanguageConfig(code="xx")
        assert config.syllable_timed is False

    def test_default_has_tech_corrections(self):
        """Test default has_tech_corrections is False."""
        config = LanguageConfig(code="xx")
        assert config.has_tech_corrections is False

    def test_custom_values(self):
        """Test LanguageConfig with custom values."""
        config = LanguageConfig(
            code="test",
            question_starters=frozenset({"q1", "q2"}),
            vowels=frozenset("aeiou"),
            stress_position="penultimate",
            syllable_timed=True,
            has_tech_corrections=True,
        )
        assert config.code == "test"
        assert config.question_starters == frozenset({"q1", "q2"})
        assert config.vowels == frozenset("aeiou")
        assert config.stress_position == "penultimate"
        assert config.syllable_timed is True
        assert config.has_tech_corrections is True

    def test_vowels_is_frozenset(self):
        """Test that vowels property is always frozenset."""
        config = LanguageConfig(code="xx", vowels=frozenset("aeiou"))
        assert isinstance(config.vowels, frozenset)

    def test_question_starters_is_frozenset(self):
        """Test that question_starters property is always frozenset."""
        config = LanguageConfig(code="xx", question_starters=frozenset({"q"}))
        assert isinstance(config.question_starters, frozenset)


class TestLanguages:
    """Test LANGUAGES dictionary and language configurations."""

    def test_languages_dict_exists(self):
        """Test that LANGUAGES dict is defined."""
        assert LANGUAGES is not None
        assert isinstance(LANGUAGES, dict)

    def test_languages_not_empty(self):
        """Test that LANGUAGES contains entries."""
        assert len(LANGUAGES) > 0

    def test_all_languages_have_code(self):
        """Test that all language configs have matching code key."""
        for code, config in LANGUAGES.items():
            assert config.code == code

    def test_pl_vowels(self):
        """Test Polish language has correct vowels including ą, ę, ó."""
        pl_config = LANGUAGES["pl"]
        expected_vowels = frozenset("aeiouyóąę")
        assert pl_config.vowels == expected_vowels
        assert "ó" in pl_config.vowels
        assert "ą" in pl_config.vowels
        assert "ę" in pl_config.vowels

    def test_pl_stress_position(self):
        """Test Polish has penultimate stress."""
        pl_config = LANGUAGES["pl"]
        assert pl_config.stress_position == "penultimate"

    def test_pl_syllable_timed(self):
        """Test Polish is syllable-timed."""
        pl_config = LANGUAGES["pl"]
        assert pl_config.syllable_timed is True

    def test_pl_has_tech_corrections(self):
        """Test Polish has tech corrections enabled."""
        pl_config = LANGUAGES["pl"]
        assert pl_config.has_tech_corrections is True

    def test_pl_question_starters(self):
        """Test Polish question starters include expected words."""
        pl_config = LANGUAGES["pl"]
        expected = {"czy", "jak", "gdzie"}
        assert expected.issubset(pl_config.question_starters)

    def test_en_vowels(self):
        """Test English language has 6 vowels."""
        en_config = LANGUAGES["en"]
        expected_vowels = frozenset("aeiouy")
        assert en_config.vowels == expected_vowels
        assert len(en_config.vowels) == 6

    def test_en_stress_position(self):
        """Test English has variable stress."""
        en_config = LANGUAGES["en"]
        assert en_config.stress_position == "variable"

    def test_en_syllable_timed(self):
        """Test English is not syllable-timed."""
        en_config = LANGUAGES["en"]
        assert en_config.syllable_timed is False

    def test_en_no_tech_corrections(self):
        """Test English has tech corrections disabled."""
        en_config = LANGUAGES["en"]
        assert en_config.has_tech_corrections is False

    def test_en_question_starters(self):
        """Test English question starters include expected words."""
        en_config = LANGUAGES["en"]
        expected = {"what", "when", "how"}
        assert expected.issubset(en_config.question_starters)

    def test_fr_stress_position(self):
        """Test French has ultimate stress."""
        fr_config = LANGUAGES["fr"]
        assert fr_config.stress_position == "ultimate"

    def test_fr_syllable_timed(self):
        """Test French is syllable-timed."""
        fr_config = LANGUAGES["fr"]
        assert fr_config.syllable_timed is True

    def test_es_syllable_timed(self):
        """Test Spanish is syllable-timed."""
        es_config = LANGUAGES["es"]
        assert es_config.syllable_timed is True

    def test_all_languages_have_non_empty_vowels(self):
        """Test that all languages have at least one vowel defined."""
        for code, config in LANGUAGES.items():
            assert len(config.vowels) > 0, f"{code} has no vowels"

    def test_all_languages_have_non_empty_question_starters(self):
        """Test that all languages have question starters defined."""
        for code, config in LANGUAGES.items():
            assert len(config.question_starters) > 0, f"{code} has no question starters"


class TestGetLanguage:
    """Test get_language accessor function."""

    def test_get_language_polish(self):
        """Test get_language returns Polish config for 'pl'."""
        config = get_language("pl")
        assert config.code == "pl"
        assert "czy" in config.question_starters

    def test_get_language_english(self):
        """Test get_language returns English config for 'en'."""
        config = get_language("en")
        assert config.code == "en"
        assert "what" in config.question_starters

    def test_get_language_spanish(self):
        """Test get_language returns Spanish config for 'es'."""
        config = get_language("es")
        assert config.code == "es"
        assert "qué" in config.question_starters

    def test_get_language_french(self):
        """Test get_language returns French config for 'fr'."""
        config = get_language("fr")
        assert config.code == "fr"
        assert "qui" in config.question_starters

    def test_get_language_german(self):
        """Test get_language returns German config for 'de'."""
        config = get_language("de")
        assert config.code == "de"
        assert "wer" in config.question_starters

    def test_get_language_unknown_returns_fallback(self):
        """Test get_language returns fallback for unknown code."""
        config = get_language("unknown_code")
        assert config is not None
        assert isinstance(config, LanguageConfig)

    def test_get_language_fallback_has_unknown_code(self):
        """Test fallback config has code='unknown'."""
        config = get_language("unknown_code")
        assert config.code == "unknown"

    def test_get_language_fallback_has_basic_vowels(self):
        """Test fallback config has basic vowels."""
        config = get_language("unknown_code")
        assert len(config.vowels) > 0
        # Fallback should have at least basic vowels
        expected_basic = frozenset("aeiou")
        assert expected_basic.issubset(config.vowels)

    def test_get_language_case_sensitive(self):
        """Test that get_language is case-sensitive."""
        pl_lower = get_language("pl")
        pl_upper = get_language("PL")
        # PL (uppercase) should not match, returns fallback
        assert pl_lower.code == "pl"
        assert pl_upper.code == "unknown"

    def test_get_language_all_defined_languages(self):
        """Test get_language returns correct config for all defined languages."""
        for code in LANGUAGES.keys():
            config = get_language(code)
            assert config.code == code
            assert config is LANGUAGES[code]

    def test_get_language_returns_correct_object(self):
        """Test that get_language returns the exact config object from LANGUAGES."""
        pl_direct = LANGUAGES["pl"]
        pl_via_accessor = get_language("pl")
        assert pl_direct is pl_via_accessor
