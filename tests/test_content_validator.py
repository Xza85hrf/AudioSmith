"""Tests for audiosmith.content_validator module."""

import pytest
from audiosmith.content_validator import ContentValidator
from audiosmith.models import DubbingSegment
from audiosmith.exceptions import ValidationError


class TestContentValidator:
    @pytest.fixture
    def validator(self):
        return ContentValidator()

    def test_validate_valid_segment(self, validator):
        seg = DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='Hello world')
        assert validator.validate_segment(seg) is True

    def test_validate_empty_text(self, validator):
        seg = DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='')
        with pytest.raises(ValidationError):
            validator.validate_segment(seg)

    def test_validate_too_short_duration(self, validator):
        seg = DubbingSegment(index=0, start_time=0.0, end_time=0.05, original_text='Hello')
        with pytest.raises(ValidationError):
            validator.validate_segment(seg, min_duration=0.5)

    def test_validate_too_long_duration(self, validator):
        seg = DubbingSegment(index=0, start_time=0.0, end_time=20.0, original_text='Hello')
        with pytest.raises(ValidationError):
            validator.validate_segment(seg, max_duration=10.0)

    def test_validate_segments_batch(self, validator):
        segments = [
            DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='Hello'),
            DubbingSegment(index=1, start_time=2.0, end_time=4.0, original_text='World'),
        ]
        result = validator.validate_segments(segments)
        assert len(result) == 2

    def test_validate_segments_filters_invalid(self, validator):
        segments = [
            DubbingSegment(index=0, start_time=0.0, end_time=2.0, original_text='Hello'),
            DubbingSegment(index=1, start_time=2.0, end_time=4.0, original_text=''),
        ]
        result = validator.validate_segments(segments)
        assert len(result) == 1

    def test_validate_text_length_ok(self, validator):
        assert validator.validate_text_length('Hello world', min_length=1, max_length=100) is True

    def test_validate_text_too_short(self, validator):
        with pytest.raises(ValidationError):
            validator.validate_text_length('Hi', min_length=10)

    def test_validate_text_too_long(self, validator):
        with pytest.raises(ValidationError):
            validator.validate_text_length('a' * 1000, max_length=100)

    def test_has_artifacts_true(self, validator):
        assert validator.has_artifacts('Hello [MUSIC] world') is True

    def test_has_artifacts_false(self, validator):
        assert validator.has_artifacts('Hello world') is False

    def test_remove_artifacts(self, validator):
        result = validator.remove_artifacts('Hello [MUSIC] world')
        assert '[MUSIC]' not in result
