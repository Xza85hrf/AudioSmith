"""Transcription post-processor — 4-stage chain: hallucination filter, splitter, punctuation, labeler."""

import logging
import re
from typing import List, Union

from audiosmith.models import DubbingSegment
from audiosmith.punctuation_restorer import PunctuationRestorer
from audiosmith.tech_corrections import TechTermCorrections
from audiosmith.content_validator import ContentValidator

logger = logging.getLogger(__name__)

FILLER_PATTERNS = [
    r'\buh\b', r'\bum\b', r'\bahh?\b', r'\bhmm+\b',
    r'\byou know\b', r'\blike\s+like\b', r'\bso\s+so\b',
]

NON_SPEECH_PATTERNS = [
    r'\[MUSIC\]', r'\[music\]', r'\[silence\]', r'\[SILENCE\]',
    r'\[inaudible\]', r'\[INAUDIBLE\]', r'\[laughter\]', r'\[LAUGHTER\]',
    r'\[background noise\]', r'\[BACKGROUND NOISE\]',
    r'\[crosstalk\]', r'\[CROSSTALK\]', r'\[cough\]', r'\[COUGH\]', r'\[NOISE\]',
]

MAX_SEGMENT_LENGTH = 500


class TranscriptionPostProcessor:
    """4-stage pipeline for transcription post-processing.

    1. Hallucination Filter — remove filler words and artifacts
    2. Segment Splitter — split long segments at sentence boundaries
    3. Punctuation Restorer — add punctuation and apply tech corrections
    4. Non-Speech Labeler — mark music/silence/noise segments
    """

    def __init__(self) -> None:
        self.punctuation_restorer = PunctuationRestorer()
        self.tech_corrections = TechTermCorrections()
        self.content_validator = ContentValidator()
        self.filler_regex = re.compile('|'.join(FILLER_PATTERNS), re.IGNORECASE)
        self.non_speech_regex = re.compile('|'.join(NON_SPEECH_PATTERNS), re.IGNORECASE)
        self.enabled_stages = {
            'hallucination_filter': True,
            'segment_splitter': True,
            'punctuation_restorer': True,
            'non_speech_labeler': True,
        }

    def process(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        if not segments:
            return []

        result = segments
        if self.enabled_stages['hallucination_filter']:
            result = self.stage1_hallucination_filter(result)
        if self.enabled_stages['segment_splitter']:
            result = self.stage2_segment_splitter(result)
        if self.enabled_stages['punctuation_restorer']:
            result = self.stage3_punctuation_restorer(result)
        if self.enabled_stages['non_speech_labeler']:
            result = self.stage4_non_speech_labeler(result)
        return result

    def stage1_hallucination_filter(
        self, input_data: Union[List[DubbingSegment], List[str]]
    ) -> Union[List[DubbingSegment], List[str]]:
        if not input_data:
            return input_data

        # Handle List[str] for standalone use
        if isinstance(input_data[0], str):
            return [
                re.sub(r'\s+', ' ', self.filler_regex.sub('', t)).strip()
                for t in input_data
                if re.sub(r'\s+', ' ', self.filler_regex.sub('', t)).strip()
            ]

        # Handle List[DubbingSegment]
        result = []
        for seg in input_data:
            cleaned = re.sub(r'\s+', ' ', self.filler_regex.sub('', seg.original_text)).strip()
            if not cleaned:
                continue
            result.append(DubbingSegment(
                index=seg.index,
                start_time=seg.start_time,
                end_time=seg.end_time,
                original_text=cleaned,
                speaker_id=seg.speaker_id,
                is_speech=seg.is_speech,
            ))
        return result

    def stage2_segment_splitter(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        result = []
        idx = 0
        for seg in segments:
            if len(seg.original_text) <= MAX_SEGMENT_LENGTH:
                seg.index = idx
                result.append(seg)
                idx += 1
                continue

            sentences = re.split(r'(?<=[.!?])\s+', seg.original_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                seg.index = idx
                result.append(seg)
                idx += 1
                continue

            total_chars = sum(len(s) for s in sentences)
            duration = seg.end_time - seg.start_time
            offset = seg.start_time

            for sentence in sentences:
                proportion = len(sentence) / total_chars if total_chars > 0 else 1.0 / len(sentences)
                chunk_dur = duration * proportion
                result.append(DubbingSegment(
                    index=idx,
                    start_time=offset,
                    end_time=offset + chunk_dur,
                    original_text=sentence,
                    speaker_id=seg.speaker_id,
                    is_speech=seg.is_speech,
                ))
                offset += chunk_dur
                idx += 1

        return result

    def stage3_punctuation_restorer(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        for seg in segments:
            text = self.punctuation_restorer.restore(seg.original_text)
            seg.original_text = self.tech_corrections.correct(text)
        return segments

    def stage4_non_speech_labeler(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        for seg in segments:
            seg.is_speech = not bool(self.non_speech_regex.search(seg.original_text))
        return segments

    def enable_stage(self, stage_name: str, enabled: bool) -> None:
        if stage_name in self.enabled_stages:
            self.enabled_stages[stage_name] = enabled
