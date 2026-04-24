"""Qwen3-TTS LoRA fine-tuning — re-exported from aiml_training.

The canonical implementation lives in aiml_training.training.qwen3_finetune.
This module provides backward-compatible imports for AudioSmith code.
"""

import warnings

warnings.warn(
    "Import from aiml_training.training.qwen3_finetune directly. "
    "This re-export will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from aiml_training.training.qwen3_finetune import (  # noqa: F401
        Qwen3LoRAConfig, Qwen3LoRATrainer, TTSFineTuneDataset)
except ImportError:
    pass
