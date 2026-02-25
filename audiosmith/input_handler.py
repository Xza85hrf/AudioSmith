"""Input validation and normalization — pre-flight checks for files and parameters."""

import logging
from pathlib import Path
from typing import Optional

from audiosmith.exceptions import InputError, ValidationError
from audiosmith.language_detect import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma'}
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}


class InputHandler:
    """Validate and normalize input files and parameters."""

    def validate_audio_file(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            raise InputError(f"Audio file not found: {path}")
        if not path.is_file():
            raise InputError(f"Path is not a file: {path}")
        return True

    def validate_audio_format(self, path: Path) -> bool:
        suffix = Path(path).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_FORMATS:
            raise ValidationError(f"Unsupported audio format: {suffix}")
        return True

    def validate_video_file(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            raise InputError(f"Video file not found: {path}")
        if not path.is_file():
            raise InputError(f"Path is not a file: {path}")
        return True

    def validate_video_format(self, path: Path) -> bool:
        suffix = Path(path).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_FORMATS:
            raise ValidationError(f"Unsupported video format: {suffix}")
        return True

    def validate_output_dir(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            raise InputError(f"Output directory does not exist: {path}")
        if not path.is_dir():
            raise InputError(f"Path is not a directory: {path}")
        return True

    def validate_language_code(self, language: str) -> bool:
        if language not in SUPPORTED_LANGUAGES:
            raise ValidationError(f"Unsupported language code: {language}")
        return True

    def normalize_path(self, path) -> Path:
        return Path(path).resolve()

    def validate_all(
        self,
        audio_file: Optional[Path] = None,
        video_file: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        source_language: str = 'en',
        target_language: str = 'pl',
    ) -> bool:
        if audio_file:
            self.validate_audio_file(audio_file)
            self.validate_audio_format(audio_file)
        if video_file:
            self.validate_video_file(video_file)
            self.validate_video_format(video_file)
        if output_dir:
            self.validate_output_dir(output_dir)
        self.validate_language_code(source_language)
        self.validate_language_code(target_language)
        return True
