"""Tests for audiosmith.error_codes module."""

from audiosmith.error_codes import ErrorCode, ErrorCategory, ErrorContext


class TestErrorCode:
    def test_values_are_integers(self):
        for code in ErrorCode:
            assert isinstance(code.value, int)

    def test_get_description(self):
        desc = ErrorCode.get_description(ErrorCode.MODEL_LOAD_ERROR)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_category(self):
        cat = ErrorCode.get_category(ErrorCode.TRANSLATION_ERROR)
        assert cat == ErrorCategory.TRANSLATION

    def test_dubbing_codes_exist(self):
        assert ErrorCode.DUBBING_EXTRACTION_ERROR.value == 9000
        assert ErrorCode.DUBBING_PIPELINE_ERROR.value == 9006


class TestErrorContext:
    def test_creation(self):
        from datetime import datetime
        ctx = ErrorContext(
            timestamp=datetime.now(),
            component='ffmpeg',
            operation='extract_audio',
            details={'file': 'test.mp4'},
        )
        assert ctx.component == 'ffmpeg'
        assert ctx.operation == 'extract_audio'
        d = ctx.to_dict()
        assert 'timestamp' in d
