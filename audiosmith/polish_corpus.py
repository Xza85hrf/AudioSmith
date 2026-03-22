"""Polish text corpus — re-exported from aiml_training.

The canonical implementation lives in aiml_training.training.polish_corpus.
This module provides backward-compatible imports for AudioSmith code.
"""

import warnings

warnings.warn(
    "Import from aiml_training.training.polish_corpus directly. "
    "This re-export will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from aiml_training.training.polish_corpus import (  # noqa: F401
        _ABBREVIATIONS, _MULTI_SPACE, _SENTENCE_SPLIT, _WIKI_MARKUP,
        ALL_POLISH_CHARS, POLISH_DIACRITICS, POLISH_DIACRITICS_UPPER,
        PolishCorpusManager)
except ImportError:
    pass
