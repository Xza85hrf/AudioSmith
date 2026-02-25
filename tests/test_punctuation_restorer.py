"""Tests for audiosmith.punctuation_restorer module."""

import pytest
from audiosmith.punctuation_restorer import PunctuationRestorer


class TestPunctuationRestorer:
    @pytest.fixture
    def restorer(self):
        return PunctuationRestorer()

    def test_add_period(self, restorer):
        assert restorer.restore('Hello world').endswith('.')

    def test_add_question_mark(self, restorer):
        assert restorer.restore('What is your name').endswith('?')

    def test_preserve_existing(self, restorer):
        assert restorer.restore('Hello world.') == 'Hello world.'

    def test_capitalize_first(self, restorer):
        result = restorer.restore('hello world')
        assert result[0] == 'H'

    def test_empty_text(self, restorer):
        assert restorer.restore('') == ''

    def test_whitespace_only(self, restorer):
        assert restorer.restore('   ') == '   '

    def test_multiple_sentences(self, restorer):
        result = restorer.restore('First sentence. second sentence')
        assert 'Second' in result

    def test_how_question(self, restorer):
        assert restorer.restore('How are you doing today').endswith('?')

    def test_where_question(self, restorer):
        assert restorer.restore('Where did you go').endswith('?')

    def test_does_question(self, restorer):
        assert restorer.restore('Does this work').endswith('?')

    def test_preserve_question_mark(self, restorer):
        assert restorer.restore('What is this?') == 'What is this?'
