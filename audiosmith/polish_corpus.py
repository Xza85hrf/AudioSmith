"""Polish text corpus — re-exported from aiml_training.

The canonical implementation lives in aiml_training.training.polish_corpus.
This module provides backward-compatible imports for AudioSmith code.
"""

from aiml_training.training.polish_corpus import (_ABBREVIATIONS,  # noqa: F401
                                                  _MULTI_SPACE,
                                                  _SENTENCE_SPLIT,
                                                  _WIKI_MARKUP,
                                                  ALL_POLISH_CHARS,
                                                  POLISH_DIACRITICS,
                                                  POLISH_DIACRITICS_UPPER,
                                                  PolishCorpusManager)
