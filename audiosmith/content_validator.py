"""Content validation — validates transcription segments before formatting/processing."""

import re
import logging
from typing import List

from audiosmith.models import DubbingSegment
from audiosmith.exceptions import ValidationError

logger = logging.getLogger(__name__)

ARTIFACT_PATTERNS = [
    r'\[MUSIC\]', r'\[music\]', r'\[silence\]', r'\[SILENCE\]',
    r'\[inaudible\]', r'\[INAUDIBLE\]', r'\[laughter\]', r'\[LAUGHTER\]',
    r'\[background noise\]', r'\[BACKGROUND NOISE\]',
    r'\[crosstalk\]', r'\[CROSSTALK\]', r'\[cough\]', r'\[COUGH\]', r'\[NOISE\]',
]


class ContentValidator:
    """Validate transcription segments for quality and format compliance."""

    def __init__(
        self,
        min_segment_duration: float = 0.2,
        max_segment_duration: float = 15.0,
        min_text_length: int = 1,
        max_text_length: int = 5000,
    ) -> None:
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.artifact_regex = re.compile('|'.join(ARTIFACT_PATTERNS), re.IGNORECASE)

    def validate_segment(
        self,
        segment: DubbingSegment,
        min_duration: float = None,
        max_duration: float = None,
    ) -> bool:
        min_dur = min_duration if min_duration is not None else self.min_segment_duration
        max_dur = max_duration if max_duration is not None else self.max_segment_duration

        duration = segment.end_time - segment.start_time
        if duration < min_dur:
            raise ValidationError(
                f"Segment {segment.index}: duration {duration:.2f}s < {min_dur}s",
                error_code="SEGMENT_TOO_SHORT",
            )
        if duration > max_dur:
            raise ValidationError(
                f"Segment {segment.index}: duration {duration:.2f}s > {max_dur}s",
                error_code="SEGMENT_TOO_LONG",
            )

        self.validate_text_length(segment.original_text)

        if self.has_artifacts(segment.original_text):
            logger.warning("Segment %d contains transcription artifacts", segment.index)

        return True

    def validate_segments(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        valid = []
        for seg in segments:
            try:
                self.validate_segment(seg)
                valid.append(seg)
            except ValidationError as e:
                logger.warning("Skipping invalid segment %d: %s", seg.index, e.message)
        return valid

    def validate_text_length(self, text: str, min_length: int = None, max_length: int = None) -> bool:
        min_len = min_length if min_length is not None else self.min_text_length
        max_len = max_length if max_length is not None else self.max_text_length

        if not text or len(text) < min_len:
            raise ValidationError(f"Text too short: {len(text) if text else 0} < {min_len}")
        if len(text) > max_len:
            raise ValidationError(f"Text too long: {len(text)} > {max_len}")
        return True

    def has_artifacts(self, text: str) -> bool:
        return bool(self.artifact_regex.search(text))

    def remove_artifacts(self, text: str) -> str:
        return self.artifact_regex.sub('', text).strip()
