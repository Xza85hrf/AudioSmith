"""Tests for audiosmith.download module."""

from audiosmith.download import (
    is_url, slugify, segments_to_txt, segments_to_vtt, segments_to_json,
)
import json


class TestIsUrl:
    def test_http(self):
        assert is_url('http://example.com')
        assert is_url('https://youtube.com/watch?v=abc')

    def test_not_url(self):
        assert not is_url('/path/to/file.mp4')
        assert not is_url('file.mp4')


class TestSlugify:
    def test_basic(self):
        assert slugify('Hello World') == 'hello-world'

    def test_special_chars(self):
        assert slugify('Hello! @World#') == 'hello-world'

    def test_dashes(self):
        assert slugify('a--b') == 'a-b'

    def test_strip(self):
        assert slugify('  spaces  ') == 'spaces'


class TestSegmentFormatters:
    def setup_method(self):
        self.segments = [
            {'text': 'Hello', 'start': 0.0, 'end': 1.5},
            {'text': 'World', 'start': 2.0, 'end': 3.0},
        ]

    def test_txt(self):
        result = segments_to_txt(self.segments)
        assert 'Hello' in result
        assert 'World' in result
        assert result == 'Hello\nWorld'

    def test_vtt(self):
        result = segments_to_vtt(self.segments)
        assert result.startswith('WEBVTT')
        assert '-->' in result

    def test_json(self):
        result = segments_to_json(self.segments)
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]['text'] == 'Hello'
        assert data[1]['start'] == 2.0
