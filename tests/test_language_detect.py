"""Tests for audiosmith.language_detect module."""

import pytest
from audiosmith.language_detect import LanguageDetector


class TestLanguageDetector:
    @pytest.fixture
    def detector(self):
        return LanguageDetector()

    def test_detect_polish(self, detector):
        assert detector.detect('Cześć, to jest tekst testowy w języku polskim.') == 'pl'

    def test_detect_spanish(self, detector):
        text = 'el gato que es la de un en una casa grande'
        assert detector.detect(text) == 'es'

    def test_detect_empty(self, detector):
        assert detector.detect('') == 'auto'

    def test_detect_short(self, detector):
        assert detector.detect('hi') == 'auto'

    def test_detect_with_confidence(self, detector):
        lang, conf = detector.detect_with_confidence('Cześć, to jest tekst testowy.')
        assert lang == 'pl'
        assert 0.0 <= conf <= 1.0

    def test_detect_with_confidence_low(self, detector):
        lang, conf = detector.detect_with_confidence('123 456 789 000 111 222')
        assert lang == 'auto'
        assert conf < 0.5

    def test_is_supported_language(self, detector):
        assert detector.is_supported_language('en') is True
        assert detector.is_supported_language('pl') is True
        assert detector.is_supported_language('xx') is False

    def test_get_supported_languages(self, detector):
        languages = detector.get_supported_languages()
        assert isinstance(languages, set)
        assert len(languages) == 23
        assert 'en' in languages

    def test_detect_japanese(self):
        detector = LanguageDetector(min_text_length=3)
        assert detector.detect('こんにちは世界のテスト') == 'ja'

    def test_detect_chinese(self):
        detector = LanguageDetector(min_text_length=3)
        assert detector.detect('你好世界测试文本') == 'zh'

    def test_detect_arabic(self):
        detector = LanguageDetector(min_text_length=3)
        assert detector.detect('مرحبا بالعالم') == 'ar'

    def test_detect_german(self, detector):
        text = 'der die und das ist sich den mit von zu'
        assert detector.detect(text) == 'de'
