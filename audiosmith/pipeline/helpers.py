"""Utility functions for the dubbing pipeline."""

import re
from pathlib import Path
from typing import Any, Dict, List

from audiosmith.emotion_config import EMOTION_TTS_MAP
from audiosmith.models import DubbingSegment
from audiosmith.srt import write_srt
from audiosmith.srt_formatter import SRTFormatter


def _emotion_to_tts_params(emotion: str, intensity: float = 0.5) -> Dict[str, float]:
    """Convert emotion label + intensity to Chatterbox TTS parameters."""
    params = EMOTION_TTS_MAP.get(emotion, {'exaggeration': 0.5, 'cfg_weight': 0.5})
    # Scale towards defaults at low intensity, towards full values at high intensity
    return {
        'exaggeration': 0.5 + (params['exaggeration'] - 0.5) * intensity,
        'cfg_weight': 0.5 + (params['cfg_weight'] - 0.5) * intensity,
    }


def _segments_to_dicts(segments: List[DubbingSegment]) -> List[Dict[str, Any]]:
    """Convert DubbingSegment objects to dictionaries for JSON serialization."""
    result = []
    for s in segments:
        result.append({
            'index': s.index,
            'start_time': s.start_time,
            'end_time': s.end_time,
            'original_text': s.original_text,
            'translated_text': s.translated_text,
            'speaker_id': s.speaker_id,
            'metadata': s.metadata,
            'tts_audio_path': str(s.tts_audio_path) if s.tts_audio_path else None,
            'tts_duration_ms': s.tts_duration_ms,
        })
    return result


def _dicts_to_segments(dicts: List[Dict[str, Any]]) -> List[DubbingSegment]:
    """Convert dictionaries back to DubbingSegment objects."""
    segments = []
    for d in dicts:
        seg = DubbingSegment(
            index=d['index'],
            start_time=d['start_time'],
            end_time=d['end_time'],
            original_text=d['original_text'],
            translated_text=d.get('translated_text', ''),
        )
        seg.speaker_id = d.get('speaker_id')
        seg.metadata = d.get('metadata', {})
        if d.get('tts_audio_path'):
            seg.tts_audio_path = Path(d['tts_audio_path'])
        seg.tts_duration_ms = d.get('tts_duration_ms')
        segments.append(seg)
    return segments


def _write_srt(segments: List[DubbingSegment], path: Path) -> None:
    """Write segments to an SRT file."""
    formatter = SRTFormatter()
    raw_segments = [
        {
            'text': seg.translated_text or seg.original_text,
            'start': seg.start_time,
            'end': seg.end_time,
            'words': [],
        }
        for seg in segments
    ]
    entries = formatter.format_segments(raw_segments)
    write_srt(entries, path)


def _write_srt_from_schedule(scheduled: List, path: Path) -> None:
    """Write SRT using the mixer's actual placement times, not original segment times."""
    formatter = SRTFormatter()
    raw_segments = [
        {
            'text': item.segment.translated_text or item.segment.original_text,
            'start': item.place_at_ms / 1000.0,
            'end': (item.place_at_ms + item.actual_duration_ms) / 1000.0,
            'words': [],
        }
        for item in scheduled
    ]
    entries = formatter.format_segments(raw_segments)
    write_srt(entries, path)


def _dedup_repeated_words(text: str, max_repeats: int = 2) -> str:
    """Collapse runs of 3+ identical consecutive words to max_repeats."""
    words = text.split()
    if len(words) < 3:
        return text
    result = [words[0]]
    count = 1
    for w in words[1:]:
        if w.lower() == result[-1].lower():
            count += 1
            if count <= max_repeats:
                result.append(w)
        else:
            result.append(w)
            count = 1
    return ' '.join(result)


def _is_fish_skippable(text: str, seg_start: float, seg_end: float,
                       total_duration: float = 0.0,
                       prev_end: float = 0.0) -> bool:
    """Detect segments that Fish Speech S2-Pro will hallucinate on.

    Returns True if the segment should be skipped for Fish Speech:
    - Text is too short (< 3 words after cleaning)
    - Segment follows a large silence gap (> 10s) with short text
    - Segment is in the last 30s of the timeline with very short text
    """
    words = text.split()
    word_count = len(words)

    # Too few words — Fish hallucinates filler on <3 word inputs
    if word_count < 3:
        return True

    # Large silence gap before this segment + short text = hallucination risk
    gap_before = seg_start - prev_end
    if gap_before > 10.0 and word_count < 6:
        return True

    # Last 30s of timeline with short text — end-of-stream hallucination
    if total_duration > 0 and (total_duration - seg_end) < 30.0 and word_count < 5:
        return True

    return False


def _validate_tts_duration(audio_samples: int, sample_rate: int,
                           word_count: int, language: str = 'en') -> bool:
    """Check if TTS output duration is reasonable for the input text.

    Returns True if duration is plausible, False if likely hallucinated.
    """
    if audio_samples <= 0 or word_count <= 0:
        return False

    duration_s = audio_samples / sample_rate
    # Speaking rates vary by language (words per second)
    wps_slow = 1.5  # Very slow speaker
    wps_fast = 5.0  # Very fast / compressed

    min_expected = word_count / wps_fast
    max_expected = word_count / wps_slow

    # Allow 2x margin on both sides
    return (duration_s >= min_expected * 0.5) and (duration_s <= max_expected * 2.0)


def _clean_tts_text(text: str) -> str:
    """Strip non-speakable content from SRT text before TTS synthesis.

    Removes: [stage directions], (parenthetical notes), music lyrics,
    speaker tags like [Marty], and leading dialogue em-dashes.
    """
    if not text:
        return text
    # Remove bracketed content: [Muzyka], [chrząkanie], [Marty], etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Remove parenthetical directions: (laughing), (whispering)
    text = re.sub(r'\(.*?\)', '', text)
    # Remove music/lyrics lines: anything with music note symbols
    text = re.sub(r'♪[^♪]*♪', '', text)
    text = re.sub(r'♪.*', '', text)
    # Strip leading dialogue em-dashes (keep the text after)
    text = re.sub(r'^\s*[–—-]\s*', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
