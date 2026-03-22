"""Language-specific configuration for text processing and prosody.

Centralizes all language-dependent data (vowels, stress rules, question words)
so individual modules can be parameterized by language code instead of
hardcoding Polish or English assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True)
class LanguageConfig:
    """Immutable language configuration for text and audio processing."""

    code: str
    question_starters: FrozenSet[str] = field(default_factory=frozenset)
    vowels: FrozenSet[str] = field(default_factory=frozenset)
    stress_position: str = "variable"  # "penultimate", "ultimate", "variable"
    syllable_timed: bool = False
    has_tech_corrections: bool = False


LANGUAGES: dict[str, LanguageConfig] = {
    "pl": LanguageConfig(
        code="pl",
        question_starters=frozenset({
            "czy", "jak", "gdzie", "kiedy", "kto", "co", "dlaczego",
            "ile", "jaki", "jaka", "jakie", "która", "który", "które",
            "czemu", "skąd", "dokąd",
        }),
        vowels=frozenset("aeiouyóąę"),
        stress_position="penultimate",
        syllable_timed=True,
        has_tech_corrections=True,
    ),
    "en": LanguageConfig(
        code="en",
        question_starters=frozenset({
            "what", "when", "where", "who", "why", "how",
            "is", "are", "was", "were",
            "do", "does", "did",
            "can", "could", "will", "would", "should",
            "have", "has",
        }),
        vowels=frozenset("aeiouy"),
        stress_position="variable",
        syllable_timed=False,
        has_tech_corrections=False,
    ),
    "es": LanguageConfig(
        code="es",
        question_starters=frozenset({
            "qué", "cuál", "cuándo", "dónde", "quién", "cómo",
            "cuánto", "por qué",
        }),
        vowels=frozenset("aeiou"),
        stress_position="variable",
        syllable_timed=True,
        has_tech_corrections=True,
    ),
    "fr": LanguageConfig(
        code="fr",
        question_starters=frozenset({
            "qui", "que", "quoi", "où", "quand", "comment",
            "pourquoi", "combien", "quel", "quelle",
            "est-ce",
        }),
        vowels=frozenset("aeiouyàâéèêëïîôùûü"),
        stress_position="ultimate",
        syllable_timed=True,
        has_tech_corrections=False,
    ),
    "de": LanguageConfig(
        code="de",
        question_starters=frozenset({
            "wer", "was", "wann", "wo", "warum", "wie",
            "welch", "wessen", "wohin", "woher",
        }),
        vowels=frozenset("aeiouäöü"),
        stress_position="variable",
        syllable_timed=False,
        has_tech_corrections=True,
    ),
}

# Default fallback for unknown languages
_FALLBACK = LanguageConfig(code="unknown", vowels=frozenset("aeiou"))


def get_language(code: str) -> LanguageConfig:
    """Get language configuration by ISO 639-1 code.

    Returns English config as fallback for unknown languages.
    """
    return LANGUAGES.get(code, _FALLBACK)
