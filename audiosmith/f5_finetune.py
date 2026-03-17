"""F5-TTS fine-tuning — re-exported from aiml_training.

The canonical implementation lives in aiml_training.training.f5_finetune.
This module provides backward-compatible imports for AudioSmith code.
"""

from aiml_training.training.f5_finetune import (F5_MEL_CONFIG,  # noqa: F401
                                                F5_MODEL_CONFIG, POLISH_CHARS,
                                                F5FineTuneConfig,
                                                F5FineTuneTrainer)
