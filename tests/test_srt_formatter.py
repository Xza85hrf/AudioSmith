"""Tests for audiosmith.srt_formatter module."""

import re

import pytest

from audiosmith.srt_formatter import SRTFormatter, MIN_GAP
from audiosmith.srt import SRTEntry


class TestFormatSegments:
    @pytest.fixture
    def formatter(self):
        return SRTFormatter()

    def test_simple_segment(self, formatter):
        result = formatter.format_segments([
            {'text': 'Hello world', 'start': 0.0, 'end': 2.0, 'words': []},
        ])
        assert len(result) == 1
        assert result[0].text == 'Hello world'
        assert isinstance(result[0], SRTEntry)

    def test_with_word_timestamps(self, formatter):
        seg = {
            'text': 'Hello world this is a test',
            'start': 0.0, 'end': 3.0,
            'words': [
                {'text': 'Hello', 'start': 0.0, 'end': 0.5},
                {'text': 'world', 'start': 0.5, 'end': 1.0},
                {'text': 'this', 'start': 1.0, 'end': 1.5},
                {'text': 'is', 'start': 1.5, 'end': 1.8},
                {'text': 'a', 'start': 1.8, 'end': 2.0},
                {'text': 'test', 'start': 2.0, 'end': 3.0},
            ],
        }
        result = formatter.format_segments([seg])
        assert len(result) >= 1
        assert all(isinstance(e, SRTEntry) for e in result)

    def test_long_text_splits(self, formatter):
        long_text = ' '.join(['word'] * 30)  # ~150 chars
        seg = {'text': long_text, 'start': 0.0, 'end': 5.0, 'words': []}
        result = formatter.format_segments([seg])
        assert len(result) > 1

    def test_empty_segment(self, formatter):
        result = formatter.format_segments([
            {'text': '', 'start': 0.0, 'end': 1.0, 'words': []},
        ])
        assert len(result) == 0

    def test_multiple_segments(self, formatter):
        segs = [
            {'text': 'First', 'start': 0.0, 'end': 1.0, 'words': []},
            {'text': 'Second', 'start': 1.5, 'end': 2.5, 'words': []},
            {'text': 'Third', 'start': 3.0, 'end': 4.0, 'words': []},
        ]
        result = formatter.format_segments(segs)
        assert [e.index for e in result] == [1, 2, 3]


class TestLineWrapping:
    @pytest.fixture
    def formatter(self):
        return SRTFormatter()

    def test_short_text_no_wrap(self, formatter):
        result = formatter._format_text('Short text')
        assert '\n' not in result

    def test_long_text_wraps(self, formatter):
        text = 'This is a much longer text that exceeds forty two characters per line limit'
        result = formatter._format_text(text)
        assert '\n' in result

    def test_wrap_at_word_boundary(self, formatter):
        text = 'This is a very long sentence that should wrap at a word boundary point'
        result = formatter._format_text(text)
        for line in result.split('\n'):
            assert not line.startswith(' ')
            assert not line.endswith(' ')


class TestTextSplitting:
    @pytest.fixture
    def formatter(self):
        return SRTFormatter()

    def test_short_text_single_chunk(self, formatter):
        result = formatter._split_text_into_chunks('Short')
        assert result == ['Short']

    def test_long_text_multiple_chunks(self, formatter):
        text = ' '.join(['word'] * 30)  # ~150 chars
        result = formatter._split_text_into_chunks(text)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 84


class TestPostProcess:
    @pytest.fixture
    def formatter(self):
        return SRTFormatter()

    def test_removes_empty(self, formatter):
        entries = [{'start': 0.0, 'end': 1.0, 'text': ''}]
        result = formatter._post_process(entries)
        assert len(result) == 0

    def test_ensures_gap(self, formatter):
        entries = [
            {'start': 0.0, 'end': 1.0, 'text': 'A'},
            {'start': 1.01, 'end': 2.0, 'text': 'B'},
        ]
        result = formatter._post_process(entries)
        assert result[1]['start'] >= result[0]['end'] + MIN_GAP

    def test_fixes_zero_duration(self, formatter):
        entries = [{'start': 5.0, 'end': 5.0, 'text': 'A'}]
        result = formatter._post_process(entries)
        assert result[0]['end'] > result[0]['start']


class TestDurationLimits:
    def test_long_segment_splits(self):
        formatter = SRTFormatter()
        long_text = ' '.join(['word'] * 30)
        seg = {'text': long_text, 'start': 0.0, 'end': 15.0, 'words': []}
        result = formatter.format_segments([seg])
        assert len(result) > 1


class TestSRTEntryOutput:
    def test_entries_have_timestamps(self):
        formatter = SRTFormatter()
        seg = {'text': 'Hello world', 'start': 1.5, 'end': 3.5, 'words': []}
        result = formatter.format_segments([seg])
        assert len(result) == 1
        assert re.match(r'\d{2}:\d{2}:\d{2},\d{3}', result[0].start_time)
        assert re.match(r'\d{2}:\d{2}:\d{2},\d{3}', result[0].end_time)
