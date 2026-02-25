"""Tests for audiosmith.input_handler module."""

import pytest
from pathlib import Path
from audiosmith.input_handler import InputHandler
from audiosmith.exceptions import InputError, ValidationError


class TestInputHandler:
    @pytest.fixture
    def handler(self):
        return InputHandler()

    def test_validate_audio_file_exists(self, handler, tmp_path):
        f = tmp_path / 'test.wav'
        f.touch()
        assert handler.validate_audio_file(f) is True

    def test_validate_audio_file_not_found(self, handler):
        with pytest.raises(InputError):
            handler.validate_audio_file(Path('/nonexistent.wav'))

    def test_validate_audio_format_supported(self, handler, tmp_path):
        f = tmp_path / 'test.wav'
        f.touch()
        assert handler.validate_audio_format(f) is True

    def test_validate_audio_format_unsupported(self, handler):
        with pytest.raises(ValidationError):
            handler.validate_audio_format(Path('file.xyz'))

    def test_validate_output_dir(self, handler, tmp_path):
        assert handler.validate_output_dir(tmp_path) is True

    def test_validate_output_dir_not_found(self, handler):
        with pytest.raises(InputError):
            handler.validate_output_dir(Path('/nonexistent_dir'))

    def test_validate_language_code_valid(self, handler):
        assert handler.validate_language_code('en') is True

    def test_validate_language_code_invalid(self, handler):
        with pytest.raises(ValidationError):
            handler.validate_language_code('xx')

    def test_normalize_path(self, handler):
        result = handler.normalize_path('.')
        assert result.is_absolute()

    def test_validate_all(self, handler, tmp_path):
        f = tmp_path / 'test.wav'
        f.touch()
        assert handler.validate_all(
            audio_file=f,
            output_dir=tmp_path,
            source_language='en',
            target_language='pl',
        ) is True

    def test_validate_video_format_supported(self, handler, tmp_path):
        f = tmp_path / 'test.mp4'
        f.touch()
        assert handler.validate_video_format(f) is True

    def test_validate_video_format_unsupported(self, handler):
        with pytest.raises(ValidationError):
            handler.validate_video_format(Path('file.abc'))
