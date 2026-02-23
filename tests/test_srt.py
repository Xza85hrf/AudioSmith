"""Tests for audiosmith.srt module."""

import pytest
from pathlib import Path

from audiosmith.srt import (
    SRTEntry, parse_srt, parse_srt_file, write_srt,
    timestamp_to_seconds, seconds_to_timestamp,
)


class TestTimestamps:
    def test_seconds_to_timestamp(self):
        assert seconds_to_timestamp(0.0) == '00:00:00,000'
        assert seconds_to_timestamp(61.5) == '00:01:01,500'
        assert seconds_to_timestamp(3661.123) == '01:01:01,123'

    def test_timestamp_to_seconds(self):
        assert timestamp_to_seconds('00:00:00,000') == 0.0
        assert timestamp_to_seconds('00:01:01,500') == 61.5
        assert abs(timestamp_to_seconds('01:01:01,123') - 3661.123) < 0.01

    def test_roundtrip(self):
        for val in [0.0, 1.5, 61.123, 3661.999]:
            ts = seconds_to_timestamp(val)
            back = timestamp_to_seconds(ts)
            assert abs(back - val) < 0.002


class TestParseSrt:
    def test_parse_basic(self, sample_srt_file):
        entries = parse_srt_file(sample_srt_file)
        assert len(entries) == 2
        assert entries[0].text == 'Hello, world!'
        assert entries[0].index == 1
        assert entries[1].text == 'This is a test.'

    def test_parse_string(self):
        content = "1\n00:00:00,000 --> 00:00:01,000\nTest\n\n"
        entries = parse_srt(content)
        assert len(entries) == 1
        assert entries[0].text == 'Test'

    def test_parse_empty(self):
        entries = parse_srt('')
        assert entries == []


class TestWriteSrt:
    def test_write_and_read_roundtrip(self, tmp_path):
        entries = [
            SRTEntry(index=1, start_time='00:00:00,000', end_time='00:00:01,500', text='Hello'),
            SRTEntry(index=2, start_time='00:00:02,000', end_time='00:00:03,000', text='World'),
        ]
        path = tmp_path / 'out.srt'
        write_srt(entries, path)

        parsed = parse_srt_file(path)
        assert len(parsed) == 2
        assert parsed[0].text == 'Hello'
        assert parsed[1].text == 'World'
        assert parsed[0].end_time == '00:00:01,500'
