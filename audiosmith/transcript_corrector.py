"""Context-aware correction of Whisper transcription errors."""

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from audiosmith.models import DubbingSegment

logger = logging.getLogger(__name__)

_COMMON_CORRECTIONS: Dict[str, Dict[str, Any]] = {
    "gonna": {"target": "going to"},
    "wanna": {"target": "want to"},
    "gotta": {"target": "got to"},
    "kinda": {"target": "kind of"},
    "dunno": {"target": "don't know"},
    "recieve": {"target": "receive"},
    "occured": {"target": "occurred"},
    "definately": {"target": "definitely"},
    "seperate": {"target": "separate"},
    "accomodate": {"target": "accommodate"},
    "wierd": {"target": "weird"},
    "thier": {"target": "their"},
    "untill": {"target": "until"},
    "basicly": {"target": "basically"},
    "goverment": {"target": "government"},
    "truely": {"target": "truly"},
    "enviroment": {"target": "environment"},
    "neccessary": {"target": "necessary"},
}


class TranscriptCorrector:
    """Context-aware correction of Whisper transcription errors."""

    def __init__(self, domain_dict_path: Optional[Path] = None) -> None:
        self._corrections: Dict[str, Dict[str, Any]] = {
            k: {**v, "context": v.get("context", [])}
            for k, v in _COMMON_CORRECTIONS.items()
        }
        self._correction_count = 0
        if domain_dict_path is not None:
            for wrong, correct in self.load_domain_dict(domain_dict_path).items():
                self._corrections[wrong.lower()] = {"target": correct, "context": []}

    def correct(self, segments: List[DubbingSegment]) -> List[DubbingSegment]:
        """Apply corrections to original_text using +/-2 adjacent segments as context."""
        self._correction_count = 0
        for i, seg in enumerate(segments):
            context_parts = []
            for j in range(max(0, i - 2), min(len(segments), i + 3)):
                if j != i and segments[j].original_text:
                    context_parts.append(segments[j].original_text)
            context = " ".join(context_parts)
            seg.original_text = self._apply_corrections(seg.original_text, context)
        logger.info("Applied %d corrections across %d segments", self._correction_count, len(segments))
        return segments

    def add_correction(
        self, wrong: str, correct: str, context_words: Optional[List[str]] = None,
    ) -> None:
        """Register a correction rule with optional context trigger words."""
        self._corrections[wrong.lower()] = {"target": correct, "context": context_words or []}

    @staticmethod
    def load_domain_dict(path: Path) -> Dict[str, str]:
        """Load domain corrections from a JSON file {wrong: correct}."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Could not load domain dict %s: %s", path, e)
            return {}

    def _apply_corrections(self, text: str, context: str) -> str:
        if not text:
            return text
        result = text
        for wrong, rule in self._corrections.items():
            required_context = rule.get("context", [])
            if required_context:
                ctx_lower = context.lower()
                if not any(w in ctx_lower for w in required_context):
                    continue
            pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)

            def _replace(match: re.Match, target: str = rule["target"]) -> str:
                self._correction_count += 1
                matched = match.group(0)
                if matched.isupper():
                    return target.upper()
                if matched[0].isupper():
                    return target.capitalize()
                return target

            result = pattern.sub(_replace, result)
        return result

    def _is_phonetically_similar(self, a: str, b: str, threshold: float = 0.7) -> bool:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

    def get_correction_count(self) -> int:
        return self._correction_count
