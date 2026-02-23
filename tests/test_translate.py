"""Tests for audiosmith.translate module."""

from unittest.mock import patch

import pytest

from audiosmith.translate import translate, translate_batch
from audiosmith.exceptions import TranslationError


class TestTranslate:
    @patch('audiosmith.translate.translate_argos')
    def test_argos_backend(self, mock_argos):
        mock_argos.return_value = 'Hola mundo'
        result = translate('Hello world', 'en', 'es', backend='argos')
        assert result == 'Hola mundo'
        mock_argos.assert_called_once_with('Hello world', 'en', 'es')

    def test_empty_text(self):
        assert translate('', 'en', 'es') == ''
        assert translate('   ', 'en', 'es') == '   '

    @patch('audiosmith.translate.translate_argos', side_effect=ImportError('no argos'))
    @patch('audiosmith.translate.translate_gemma', side_effect=Exception('no gemma'))
    def test_fallback_failure(self, mock_gemma, mock_argos):
        with pytest.raises(TranslationError):
            translate('text', 'en', 'es')

    @patch('audiosmith.translate.translate_argos', side_effect=ImportError('no argos'))
    @patch('audiosmith.translate.translate_gemma', return_value='fallback result')
    def test_argos_to_gemma_fallback(self, mock_gemma, mock_argos):
        result = translate('text', 'en', 'es')
        assert result == 'fallback result'


class TestTranslateBatch:
    @patch('audiosmith.translate.translate_argos')
    def test_batch(self, mock_argos):
        mock_argos.side_effect = ['A', 'B', 'C']
        result = translate_batch(['a', 'b', 'c'], 'en', 'es')
        assert result == ['A', 'B', 'C']
