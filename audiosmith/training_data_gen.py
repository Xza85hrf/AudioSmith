"""Training data generation — re-exported from aiml_training.

The canonical implementation lives in aiml_training.training.training_data_gen.
This module provides backward-compatible imports for AudioSmith code.
"""

try:
    from aiml_training.training.training_data_gen import (  # noqa: F401
        Checkpoint, TrainingDataConfig, TrainingDataGenerator)
except ImportError:
    pass

# Re-export EMOTION_TTS_MAP from emotion_config for backward compatibility
from audiosmith.emotion_config import EMOTION_TTS_MAP as _EMOTION_TTS_MAP  # noqa: F401
