"""Punctuation restoration for transcribed text — adds missing periods, question marks, capitalization."""

import re
from typing import List


QUESTION_STARTERS = {
    'what', 'when', 'where', 'who', 'why', 'how',
    'is', 'are', 'was', 'were',
    'do', 'does', 'did',
    'can', 'could', 'will', 'would', 'should',
    'have', 'has',
}


class PunctuationRestorer:
    """Restore punctuation and capitalization to raw transcribed text."""

    def restore(self, text: str) -> str:
        if not text or not text.strip():
            return text

        sentences = self._split_sentences(text)
        return ' '.join(self._restore_sentence(s) for s in sentences)

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+|\n+', text)
        return [p.strip() for p in parts if p.strip()]

    def _restore_sentence(self, sentence: str) -> str:
        sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
        if sentence[-1] in '.!?;:':
            return sentence
        return sentence + self._determine_ending(sentence)

    def _determine_ending(self, sentence: str) -> str:
        words = sentence.lower().split()
        if words and words[0] in QUESTION_STARTERS:
            return '?'
        return '.'
