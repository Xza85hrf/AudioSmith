"""Training data generation — re-exported from aiml_training.

The canonical implementation lives in aiml_training.training.training_data_gen.
This module provides backward-compatible imports for AudioSmith code.
"""

try:
    from aiml_training.training.training_data_gen import (  # noqa: F401
        _EMOTION_TTS_MAP, Checkpoint, TrainingDataConfig, TrainingDataGenerator)
except ImportError:
    pass
