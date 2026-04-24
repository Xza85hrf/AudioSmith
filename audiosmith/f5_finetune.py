"""F5-TTS fine-tuning — re-exported from aiml_training.

The canonical implementation lives in aiml_training.training.f5_finetune.
This module provides backward-compatible imports for AudioSmith code.
"""

import warnings

warnings.warn(
    "Import from aiml_training.training.f5_finetune directly. "
    "This re-export will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from aiml_training.training.f5_finetune import (  # noqa: F401
        F5_MEL_CONFIG, F5_MODEL_CONFIG, POLISH_CHARS, F5FineTuneConfig,
        F5FineTuneTrainer)
except ImportError:
    pass
