"""Polish text corpus preparation for TTS training data generation.

Downloads Polish Wikipedia plaintext, extracts sentences, validates
Polish diacritics, and produces a clean corpus file for TTS synthesis.
"""

import logging
import re
import random
from pathlib import Path
from typing import List, Optional, Set

from audiosmith.exceptions import TrainingError

logger = logging.getLogger(__name__)

# Polish diacritics that distinguish Polish from other Latin-script languages
POLISH_DIACRITICS: Set[str] = set("ąęćśźżłńó")
POLISH_DIACRITICS_UPPER: Set[str] = set("ĄĘĆŚŹŻŁŃÓ")
ALL_POLISH_CHARS: Set[str] = POLISH_DIACRITICS | POLISH_DIACRITICS_UPPER

# Common Polish abbreviations to expand
_ABBREVIATIONS = {
    "ul.": "ulica",
    "al.": "aleja",
    "pl.": "plac",
    "nr": "numer",
    "tel.": "telefon",
    "wg": "według",
    "np.": "na przykład",
    "tzn.": "to znaczy",
    "tzw.": "tak zwany",
    "m.in.": "między innymi",
    "ok.": "około",
    "tj.": "to jest",
    "dr": "doktor",
    "prof.": "profesor",
    "mgr": "magister",
    "inż.": "inżynier",
    "r.": "roku",
    "w.": "wiek",
    "tys.": "tysięcy",
    "mln": "milionów",
    "mld": "miliardów",
    "godz.": "godzina",
    "min.": "minut",
    "sek.": "sekund",
    "im.": "imienia",
    "św.": "święty",
    "woj.": "województwo",
    "pow.": "powiat",
    "gm.": "gmina",
}

# Regex to detect Wikipedia markup remnants
_WIKI_MARKUP = re.compile(r"\[\[|\]\]|\{\{|\}\}|<[^>]+>|&[a-z]+;|\|")
# Regex for multiple spaces
_MULTI_SPACE = re.compile(r"\s+")
# Sentence boundary (period/question/exclamation followed by space and uppercase)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-ZĄĘĆŚŹŻŁŃÓ])')


class PolishCorpusManager:
    """Manages Polish text corpus for TTS training data generation."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or Path.home() / ".cache" / "audiosmith" / "corpus"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def download_wikipedia(self, output_path: Optional[Path] = None) -> Path:
        """Download and extract Polish Wikipedia plaintext.

        Uses the Wikimedia cirrussearch dump (pre-extracted text) for efficiency.
        Falls back to a smaller sample if the full dump is unavailable.

        Returns path to the downloaded plaintext file.
        """
        import urllib.request
        import gzip
        import json

        dump_path = output_path or self._cache_dir / "plwiki-text.txt"
        if dump_path.exists() and dump_path.stat().st_size > 1_000_000:
            logger.info("Using cached Wikipedia dump: %s", dump_path)
            return dump_path

        # CirrusSearch dump — dated directories with JSON lines
        # Format: alternating index + content lines
        dump_url = self._find_latest_dump_url()
        gz_path = self._cache_dir / "plwiki-cirrussearch.json.gz"

        logger.info("Downloading Polish Wikipedia dump from %s", dump_url)
        try:
            urllib.request.urlretrieve(dump_url, str(gz_path))
        except Exception as e:
            raise TrainingError(
                f"Failed to download Wikipedia dump: {e}",
                error_code="TRAIN_WIKI_DL",
                original_error=e,
            )

        logger.info("Extracting text from dump (~5.6 GB compressed)...")
        line_count = 0
        with gzip.open(str(gz_path), "rt", encoding="utf-8") as gz, \
             open(str(dump_path), "w", encoding="utf-8") as out:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    # CirrusSearch alternates index lines and content lines.
                    # Content lines have a "text" field; index lines have "index".
                    text = doc.get("text", "")
                    if text and len(text) > 50:
                        out.write(text + "\n\n")
                        line_count += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info("Extracted %d articles to %s", line_count, dump_path)
        return dump_path

    @staticmethod
    def _find_latest_dump_url() -> str:
        """Find the latest CirrusSearch dump URL for Polish Wikipedia."""
        import urllib.request
        import re as _re

        index_url = "https://dumps.wikimedia.org/other/cirrussearch/"
        try:
            with urllib.request.urlopen(index_url) as resp:
                html = resp.read().decode("utf-8")
            dates = _re.findall(r'href="(\d{8})/"', html)
            if not dates:
                raise TrainingError(
                    "No CirrusSearch dump dates found",
                    error_code="TRAIN_WIKI_DL",
                )
            latest = sorted(dates)[-1]
            return (
                f"https://dumps.wikimedia.org/other/cirrussearch/{latest}/"
                f"plwiki-{latest}-cirrussearch-content.json.gz"
            )
        except TrainingError:
            raise
        except Exception as e:
            raise TrainingError(
                f"Failed to discover dump URL: {e}",
                error_code="TRAIN_WIKI_DL",
                original_error=e,
            )

    def extract_sentences(
        self,
        text_path: Path,
        min_len: int = 10,
        max_len: int = 200,
        max_sentences: int = 50_000,
    ) -> List[str]:
        """Extract and filter Polish sentences from plaintext.

        Args:
            text_path: Path to plaintext file (one article per paragraph).
            min_len: Minimum sentence length in characters.
            max_len: Maximum sentence length in characters.
            max_sentences: Stop after collecting this many unique sentences.

        Returns:
            List of unique, cleaned Polish sentences.
        """
        if not text_path.exists():
            raise TrainingError(
                f"Text file not found: {text_path}",
                error_code="TRAIN_CORPUS_MISSING",
            )

        seen: Set[str] = set()
        sentences: List[str] = []

        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or len(line) < min_len:
                    continue

                # Split into sentences
                for sent in _SENTENCE_SPLIT.split(line):
                    cleaned = self._clean_sentence(sent)
                    if not cleaned:
                        continue

                    if len(cleaned) < min_len or len(cleaned) > max_len:
                        continue

                    if not self.validate_polish(cleaned):
                        continue

                    # Deduplicate
                    key = cleaned.lower().strip()
                    if key in seen:
                        continue
                    seen.add(key)

                    sentences.append(cleaned)
                    if len(sentences) >= max_sentences:
                        logger.info("Reached target of %d sentences", max_sentences)
                        return sentences

        logger.info("Extracted %d unique Polish sentences", len(sentences))
        return sentences

    def validate_polish(self, text: str) -> bool:
        """Check that text contains Polish diacritics (not pure ASCII).

        Returns True if the text has at least one Polish-specific character.
        """
        return bool(set(text) & ALL_POLISH_CHARS)

    def diversify(
        self,
        sentences: List[str],
        target_count: int = 10_000,
        question_ratio: float = 0.3,
    ) -> List[str]:
        """Balance corpus for punctuation diversity and length variety.

        Aims for ~30% questions, varied sentence lengths, mixed punctuation.

        Args:
            sentences: Input sentence pool.
            target_count: Desired output count.
            question_ratio: Target ratio of questions (sentences ending with '?').

        Returns:
            Balanced subset of sentences.
        """
        if len(sentences) <= target_count:
            return sentences

        questions = [s for s in sentences if s.rstrip().endswith("?")]
        exclamations = [s for s in sentences if s.rstrip().endswith("!")]
        statements = [s for s in sentences if s.rstrip().endswith(".")]

        result: List[str] = []
        target_questions = int(target_count * question_ratio)
        target_excl = int(target_count * 0.05)

        # Sample questions
        random.shuffle(questions)
        result.extend(questions[:target_questions])

        # Sample exclamations
        random.shuffle(exclamations)
        result.extend(exclamations[:target_excl])

        # Fill rest with statements
        remaining = target_count - len(result)
        random.shuffle(statements)
        result.extend(statements[:remaining])

        # If still short, add any remaining
        if len(result) < target_count:
            used = set(result)
            extras = [s for s in sentences if s not in used]
            random.shuffle(extras)
            result.extend(extras[:target_count - len(result)])

        random.shuffle(result)
        logger.info(
            "Diversified corpus: %d sentences (%.0f%% questions, %.0f%% excl)",
            len(result),
            sum(1 for s in result if s.endswith("?")) / max(len(result), 1) * 100,
            sum(1 for s in result if s.endswith("!")) / max(len(result), 1) * 100,
        )
        return result

    def save_corpus(self, sentences: List[str], path: Path) -> None:
        """Save corpus to a text file (one sentence per line, UTF-8)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sent in sentences:
                f.write(sent + "\n")
        logger.info("Saved %d sentences to %s", len(sentences), path)

    def load_corpus(self, path: Path) -> List[str]:
        """Load corpus from a text file (one sentence per line)."""
        if not path.exists():
            raise TrainingError(
                f"Corpus file not found: {path}",
                error_code="TRAIN_CORPUS_MISSING",
            )
        with open(path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d sentences from %s", len(sentences), path)
        return sentences

    def _clean_sentence(self, text: str) -> str:
        """Clean a sentence: remove markup, expand abbreviations, normalize."""
        # Remove Wikipedia markup remnants
        text = _WIKI_MARKUP.sub("", text)

        # Expand abbreviations
        for abbr, expansion in _ABBREVIATIONS.items():
            text = text.replace(abbr, expansion)

        # Normalize whitespace
        text = _MULTI_SPACE.sub(" ", text).strip()

        # Must end with sentence-ending punctuation
        if text and text[-1] not in ".!?":
            return ""

        # Reject if contains digits (reduces TTS errors on numbers)
        if re.search(r"\d", text):
            return ""

        return text
