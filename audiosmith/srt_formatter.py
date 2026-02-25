"""Professional SRT subtitle formatter — splits Whisper segments into broadcast-quality subtitles."""

import re
import logging
from typing import Any, Dict, List

from audiosmith.srt import SRTEntry, seconds_to_timestamp

logger = logging.getLogger(__name__)

MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CHARS_TOTAL = 84  # 42 * 2
MIN_DURATION = 1.0
MAX_DURATION = 7.0
MIN_GAP = 0.04  # 40ms gap between subtitles

STRONG_BREAKS = {'.', '!', '?', '\u2026'}
WEAK_BREAKS = {',', ';', ':', '\u2013', '-'}


class SRTFormatter:
    """Converts Whisper transcription segments into professional-quality SRT subtitles.

    Enforces: max 42 chars/line, max 2 lines per subtitle, 1-7s duration.
    Uses word-level timestamps when available.
    """

    def __init__(
        self,
        max_chars_per_line: int = MAX_CHARS_PER_LINE,
        max_lines: int = MAX_LINES,
        max_duration: float = MAX_DURATION,
        min_duration: float = MIN_DURATION,
    ) -> None:
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_chars_total = max_chars_per_line * max_lines

    def format_segments(self, segments: List[Dict[str, Any]]) -> List[SRTEntry]:
        """Convert Whisper segments to professional SRT entries."""
        raw_entries: List[Dict[str, Any]] = []

        for segment in segments:
            words = segment.get('words', [])
            if words and len(words) > 0:
                raw_entries.extend(self._split_with_words(segment))
            else:
                raw_entries.extend(self._split_by_text(segment))

        processed = self._post_process(raw_entries)

        return [
            SRTEntry(
                index=i,
                start_time=seconds_to_timestamp(e['start']),
                end_time=seconds_to_timestamp(e['end']),
                text=e['text'],
            )
            for i, e in enumerate(processed, 1)
        ]

    def _split_with_words(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split using word-level timestamps for precise timing."""
        words = segment.get('words', [])
        if not words:
            return self._split_by_text(segment)

        entries: List[Dict[str, Any]] = []
        current_words: List[Dict[str, Any]] = []
        current_text = ''
        current_start = None

        for word in words:
            word_text = word.get('text', '').strip()
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)

            if not word_text:
                continue

            if current_start is None:
                current_start = word_start

            test_text = (current_text + ' ' + word_text).strip()
            test_duration = word_end - current_start

            should_split = False
            if len(test_text) > self.max_chars_total:
                should_split = True
            if test_duration > self.max_duration and current_text:
                should_split = True
            if current_text and current_text[-1] in STRONG_BREAKS and test_duration > self.min_duration:
                should_split = True

            if should_split and current_text:
                last_word = current_words[-1] if current_words else word
                entries.append({
                    'start': current_start,
                    'end': last_word.get('end', word_start),
                    'text': self._format_text(current_text),
                })
                current_words = []
                current_text = ''
                current_start = word_start

            current_words.append(word)
            current_text = (current_text + ' ' + word_text).strip()

        if current_text:
            last_word = current_words[-1] if current_words else {'end': segment.get('end', 0)}
            entries.append({
                'start': current_start or segment.get('start', 0),
                'end': last_word.get('end', segment.get('end', 0)),
                'text': self._format_text(current_text),
            })

        return entries

    def _split_by_text(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback: split by text with proportional timing."""
        text = segment.get('text', '').strip()
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        duration = end - start

        if not text:
            return []

        if len(text) <= self.max_chars_total and duration <= self.max_duration:
            return [{'start': start, 'end': end, 'text': self._format_text(text)}]

        chunks = self._split_text_into_chunks(text)
        entries: List[Dict[str, Any]] = []
        total_chars = sum(len(c) for c in chunks)
        current_time = start

        for i, chunk in enumerate(chunks):
            chunk_duration = (len(chunk) / max(total_chars, 1)) * duration
            chunk_duration = max(self.min_duration, min(chunk_duration, self.max_duration))
            chunk_end = min(current_time + chunk_duration, end)
            if i == len(chunks) - 1:
                chunk_end = end
            entries.append({
                'start': current_time,
                'end': chunk_end,
                'text': self._format_text(chunk),
            })
            current_time = chunk_end + MIN_GAP

        return entries

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text at sentence/clause boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current = ''

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self.max_chars_total:
                current = (current + ' ' + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                if len(sentence) > self.max_chars_total:
                    sub = self._split_long_text(sentence)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ''
                else:
                    current = sentence

        if current:
            chunks.append(current)

        return chunks if chunks else [text[:self.max_chars_total]]

    def _split_long_text(self, text: str) -> List[str]:
        """Split long text at word boundaries."""
        words = text.split()
        chunks: List[str] = []
        current = ''

        for word in words:
            if len(current) + len(word) + 1 <= self.max_chars_total:
                current = (current + ' ' + word).strip()
            else:
                if current:
                    chunks.append(current)
                current = word

        if current:
            chunks.append(current)

        return chunks

    def _format_text(self, text: str) -> str:
        """Wrap text into max 2 lines at natural midpoint."""
        text = text.strip()
        if len(text) <= self.max_chars_per_line:
            return text

        mid = len(text) // 2
        best = mid

        for offset in range(min(20, mid)):
            if mid + offset < len(text) and text[mid + offset] == ' ':
                best = mid + offset
                break
            if mid - offset >= 0 and text[mid - offset] == ' ':
                best = mid - offset
                break

        line1 = text[:best].strip()
        line2 = text[best:].strip()
        return f'{line1}\n{line2}'

    def _post_process(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure minimum gaps, remove empties, fix durations."""
        processed: List[Dict[str, Any]] = []

        for e in entries:
            if not e['text'].strip():
                continue
            if processed:
                prev = processed[-1]
                if e['start'] < prev['end'] + MIN_GAP:
                    e['start'] = prev['end'] + MIN_GAP
            if e['end'] <= e['start']:
                e['end'] = e['start'] + self.min_duration
            processed.append(e)

        return processed
