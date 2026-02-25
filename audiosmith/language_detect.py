"""Language detection using character-set heuristics and word-frequency scoring."""

import logging
import re
from typing import Set, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: Set[str] = {
    'en', 'pl', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi', 'nl',
    'sv', 'da', 'fi', 'el', 'he', 'ms', 'nb', 'sw', 'tr',
}

_RANGE_PATTERNS = {
    'ja': re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),
    'zh': re.compile(r'[\u4E00-\u9FFF]'),
    'ko': re.compile(r'[\uAC00-\uD7AF]'),
    'ar': re.compile(r'[\u0600-\u06FF]'),
    'hi': re.compile(r'[\u0900-\u097F]'),
    'ru': re.compile(r'[\u0400-\u04FF]'),
    'el': re.compile(r'[\u0370-\u03FF]'),
    'he': re.compile(r'[\u0590-\u05FF]'),
}

_PL_DIACRITICS = re.compile(r'[ąćęłńóśźż]', re.IGNORECASE)

_COMMON_WORDS = {
    'pl': {'i', 'w', 'z', 'na', 'do', 'nie', 'to', 'się', 'jest', 'jak'},
    'es': {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'una'},
    'fr': {'le', 'de', 'et', 'un', 'il', 'ne', 'je', 'son', 'que', 'les'},
    'de': {'der', 'die', 'und', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'ist'},
    'it': {'il', 'di', 'che', 'la', 'un', 'per', 'in', 'è', 'sono', 'una'},
    'pt': {'de', 'da', 'em', 'um', 'para', 'é', 'que', 'do', 'uma', 'não'},
    'nl': {'de', 'het', 'een', 'van', 'ik', 'te', 'dat', 'die', 'op', 'zijn'},
    'sv': {'och', 'att', 'det', 'en', 'jag', 'av', 'för', 'på', 'är', 'som'},
    'da': {'og', 'det', 'at', 'en', 'er', 'der', 'til', 'af', 'ikke', 'jeg'},
    'fi': {'ja', 'on', 'ei', 'ole', 'se', 'että', 'hän', 'minä', 'sinä', 'me'},
    'tr': {'ve', 'bu', 'bir', 'da', 'için', 'ama', 'ben', 'sen', 'var', 'çok'},
    'nb': {'og', 'det', 'at', 'en', 'er', 'som', 'til', 'på', 'av', 'har'},
    'ms': {'dan', 'di', 'kepada', 'untuk', 'yang', 'adalah', 'ini', 'itu', 'dengan', 'saya'},
    'sw': {'na', 'ya', 'wa', 'kwa', 'ku', 'ni', 'cha', 'wote', 'kama', 'la'},
}


class LanguageDetector:
    """Detect language from text using character-set and word-frequency heuristics."""

    def __init__(self, min_text_length: int = 20, confidence_threshold: float = 0.5) -> None:
        self.min_text_length = min_text_length
        self.confidence_threshold = confidence_threshold

    def detect(self, text: str) -> str:
        lang, conf = self.detect_with_confidence(text)
        return lang

    def detect_with_confidence(self, text: str) -> Tuple[str, float]:
        if not text or len(text) < self.min_text_length:
            return 'auto', 0.0

        # Check character-range patterns (CJK, Arabic, Cyrillic, etc.)
        for lang, pattern in _RANGE_PATTERNS.items():
            if pattern.search(text):
                return lang, 1.0

        # Polish diacritics are a strong signal
        if _PL_DIACRITICS.search(text):
            return 'pl', 0.95

        # Word-frequency scoring
        words = set(re.findall(r'\b\w+\b', text.lower()))
        if not words:
            return 'auto', 0.0

        scores = {}
        for lang, markers in _COMMON_WORDS.items():
            matches = len(words & markers)
            scores[lang] = matches

        best_lang = max(scores, key=scores.get) if scores else 'en'
        best_score = scores.get(best_lang, 0)

        if best_score == 0:
            return 'auto', 0.0

        confidence = min(1.0, best_score / max(len(words) * 0.1, 1.0))
        if confidence >= self.confidence_threshold:
            return best_lang, confidence

        return 'auto', confidence

    def is_supported_language(self, code: str) -> bool:
        return code in SUPPORTED_LANGUAGES

    def get_supported_languages(self) -> Set[str]:
        return SUPPORTED_LANGUAGES.copy()
