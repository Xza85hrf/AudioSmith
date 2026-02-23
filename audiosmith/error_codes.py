"""Error codes for the AudioSmith processing pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


@dataclass
class ErrorContext:
    """Detailed context for errors."""
    timestamp: datetime
    component: str
    operation: str
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "details": self.details,
            "stack_trace": self.stack_trace,
        }


class ErrorCategory(Enum):
    SYSTEM = "System"
    FILE = "File Operation"
    AUDIO = "Audio Processing"
    TRANSLATION = "Translation"
    PIPELINE = "Pipeline"
    MODEL = "ML Model"


class ErrorCode(Enum):
    # System Errors (1000-1009)
    SYSTEM_ERROR = 1000
    INSUFFICIENT_RESOURCES = 1001
    IO_ERROR = 1002
    PERMISSION_ERROR = 1003
    MEMORY_ERROR = 1004
    GPU_ERROR = 1006
    TIMEOUT_ERROR = 1007

    # File Errors (2000-2006)
    FILE_NOT_FOUND = 2000
    FILE_ACCESS_DENIED = 2001
    FILE_CORRUPTED = 2002
    INVALID_FILE_FORMAT = 2003
    FILE_TOO_LARGE = 2004
    FILE_WRITE_ERROR = 2005
    FILE_READ_ERROR = 2006

    # Audio Errors (4000-4007)
    AUDIO_EXTRACTION_ERROR = 4000
    TRANSCRIPTION_ERROR = 4001
    SAMPLE_RATE_ERROR = 4004
    TTS_ERROR = 4007

    # Translation Errors (5000-5002)
    TRANSLATION_ERROR = 5000
    LANGUAGE_NOT_SUPPORTED = 5002

    # Pipeline Errors (6000-6005)
    PIPELINE_INITIALIZATION_ERROR = 6000
    STEP_EXECUTION_ERROR = 6001
    INVALID_CONFIGURATION = 6002
    PIPELINE_TIMEOUT = 6003
    STATE_ERROR = 6005

    # Model Errors (7000-7002)
    MODEL_LOAD_ERROR = 7000
    MODEL_INFERENCE_ERROR = 7001
    MODEL_NOT_FOUND = 7002

    # Dubbing Pipeline Errors (9000-9006)
    DUBBING_EXTRACTION_ERROR = 9000
    DUBBING_TRANSCRIPTION_ERROR = 9001
    DUBBING_TRANSLATION_ERROR = 9002
    DUBBING_TTS_ERROR = 9003
    DUBBING_MIX_ERROR = 9004
    DUBBING_ENCODE_ERROR = 9005
    DUBBING_PIPELINE_ERROR = 9006

    @classmethod
    def get_category(cls, code) -> ErrorCategory:
        code_value = code.value if isinstance(code, cls) else code
        ranges = {
            (1000, 1999): ErrorCategory.SYSTEM,
            (2000, 2999): ErrorCategory.FILE,
            (4000, 4999): ErrorCategory.AUDIO,
            (5000, 5999): ErrorCategory.TRANSLATION,
            (6000, 6999): ErrorCategory.PIPELINE,
            (7000, 7999): ErrorCategory.MODEL,
            (9000, 9999): ErrorCategory.PIPELINE,
        }
        for (lo, hi), cat in ranges.items():
            if lo <= code_value <= hi:
                return cat
        return ErrorCategory.SYSTEM

    @classmethod
    def get_description(cls, code) -> str:
        descriptions = {
            cls.SYSTEM_ERROR: "General system error",
            cls.INSUFFICIENT_RESOURCES: "Insufficient system resources",
            cls.IO_ERROR: "Input/output operation failed",
            cls.PERMISSION_ERROR: "Permission denied",
            cls.MEMORY_ERROR: "Memory allocation error",
            cls.GPU_ERROR: "GPU operation error",
            cls.TIMEOUT_ERROR: "Operation timed out",
            cls.FILE_NOT_FOUND: "File not found",
            cls.FILE_ACCESS_DENIED: "File access denied",
            cls.FILE_CORRUPTED: "File is corrupted",
            cls.INVALID_FILE_FORMAT: "Invalid file format",
            cls.FILE_TOO_LARGE: "File size exceeds limit",
            cls.FILE_WRITE_ERROR: "Failed to write file",
            cls.FILE_READ_ERROR: "Failed to read file",
            cls.AUDIO_EXTRACTION_ERROR: "Audio extraction failed",
            cls.TRANSCRIPTION_ERROR: "Transcription failed",
            cls.SAMPLE_RATE_ERROR: "Invalid sample rate",
            cls.TTS_ERROR: "Text-to-speech synthesis failed",
            cls.TRANSLATION_ERROR: "Translation failed",
            cls.LANGUAGE_NOT_SUPPORTED: "Language not supported",
            cls.PIPELINE_INITIALIZATION_ERROR: "Pipeline initialization failed",
            cls.STEP_EXECUTION_ERROR: "Pipeline step execution failed",
            cls.INVALID_CONFIGURATION: "Invalid configuration",
            cls.PIPELINE_TIMEOUT: "Pipeline execution timed out",
            cls.STATE_ERROR: "Invalid pipeline state",
            cls.MODEL_LOAD_ERROR: "Failed to load ML model",
            cls.MODEL_INFERENCE_ERROR: "Model inference failed",
            cls.MODEL_NOT_FOUND: "ML model not found",
            cls.DUBBING_EXTRACTION_ERROR: "Audio extraction failed during dubbing",
            cls.DUBBING_TRANSCRIPTION_ERROR: "Transcription failed during dubbing",
            cls.DUBBING_TRANSLATION_ERROR: "Translation failed during dubbing",
            cls.DUBBING_TTS_ERROR: "Voice synthesis failed during dubbing",
            cls.DUBBING_MIX_ERROR: "Audio mixing failed during dubbing",
            cls.DUBBING_ENCODE_ERROR: "Video encoding failed during dubbing",
            cls.DUBBING_PIPELINE_ERROR: "Generic dubbing pipeline failure",
        }
        return descriptions.get(code, "Unknown error")
