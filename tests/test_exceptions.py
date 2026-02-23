"""Tests for audiosmith.exceptions module."""

from audiosmith.exceptions import (
    AudioSmithError, ProcessingError, TranscriptionError,
    TranslationError, TTSError, DubbingError,
)


class TestExceptionHierarchy:
    def test_base_error(self):
        e = AudioSmithError('test', error_code='E001')
        assert e.message == 'test'
        assert e.error_code == 'E001'
        assert '[E001]' in str(e)

    def test_to_dict(self):
        e = AudioSmithError('msg', error_code='X', details={'key': 'val'})
        d = e.to_dict()
        assert d['error_type'] == 'AudioSmithError'
        assert d['error_code'] == 'X'
        assert d['details'] == {'key': 'val'}

    def test_inheritance(self):
        assert issubclass(DubbingError, ProcessingError)
        assert issubclass(ProcessingError, AudioSmithError)
        assert issubclass(TranscriptionError, ProcessingError)
        assert issubclass(TranslationError, ProcessingError)
        assert issubclass(TTSError, ProcessingError)

    def test_default_codes(self):
        assert DubbingError('x').error_code == 'DUB_ERR'
        assert TTSError('x').error_code == 'TTS_ERR'
        assert TranslationError('x').error_code == 'TRAN_ERR'

    def test_original_error(self):
        orig = ValueError('bad')
        e = AudioSmithError('wrap', original_error=orig)
        assert e.original_error is orig
        assert 'bad' in str(e)
