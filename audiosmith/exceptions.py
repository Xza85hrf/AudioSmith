"""Custom exception hierarchy for AudioSmith."""


class AudioSmithError(Exception):
    """Base exception for all AudioSmith errors."""

    def __init__(self, message, error_code=None, details=None, original_error=None):
        self.message = message
        self.error_code = error_code or "UNKNOWN"
        self.details = details or {}
        self.original_error = original_error

        full_message = f"[{self.error_code}] {message}"
        if details:
            full_message += f"\nDetails: {details}"
        if original_error:
            full_message += f"\nCaused by: {original_error}"

        super().__init__(full_message)

    def to_dict(self):
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None,
        }


class ProcessingError(AudioSmithError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "PROC_ERR", details, original_error)


class ConfigError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "CFG_ERR", details, original_error)


class InputError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "INP_ERR", details, original_error)


class ConversionError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "CONV_ERR", details, original_error)


class TranscriptionError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "TRANS_ERR", details, original_error)


class TranslationError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "TRAN_ERR", details, original_error)


class TTSError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "TTS_ERR", details, original_error)


class DubbingError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "DUB_ERR", details, original_error)


class DiarizationError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "DIAR_ERR", details, original_error)


class VocalIsolationError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "VOCAL_ERR", details, original_error)


class ValidationError(ProcessingError):
    def __init__(self, message, error_code=None, details=None, original_error=None):
        super().__init__(message, error_code or "VALID_ERR", details, original_error)
