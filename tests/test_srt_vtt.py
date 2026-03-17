"""Tests for SRT and VTT subtitle utilities."""



from audiosmith.srt import (SRTEntry, parse_srt, seconds_to_timestamp,
                            seconds_to_vtt_timestamp, timestamp_to_seconds,
                            write_srt, write_vtt)


class TestTimestampConversion:
    def test_timestamp_to_seconds_basic(self):
        assert timestamp_to_seconds('00:00:01,000') == 1.0

    def test_timestamp_to_seconds_complex(self):
        result = timestamp_to_seconds('01:23:45,678')
        assert abs(result - 5025.678) < 0.001

    def test_timestamp_to_seconds_dot_separator(self):
        assert timestamp_to_seconds('00:00:01.500') == 1.5

    def test_seconds_to_timestamp_basic(self):
        assert seconds_to_timestamp(1.0) == '00:00:01,000'

    def test_seconds_to_timestamp_complex(self):
        assert seconds_to_timestamp(5025.678) == '01:23:45,678'

    def test_seconds_to_vtt_timestamp_basic(self):
        assert seconds_to_vtt_timestamp(1.0) == '00:00:01.000'

    def test_seconds_to_vtt_timestamp_complex(self):
        assert seconds_to_vtt_timestamp(5025.678) == '01:23:45.678'

    def test_srt_vs_vtt_timestamp_format(self):
        """SRT uses comma, VTT uses dot as millisecond separator."""
        srt = seconds_to_timestamp(1.5)
        vtt = seconds_to_vtt_timestamp(1.5)
        assert ',' in srt and '.' not in srt.split(':')[-1]
        assert '.' in vtt and ',' not in vtt


class TestSRTEntry:
    def test_to_srt(self):
        entry = SRTEntry(1, '00:00:01,000', '00:00:03,000', 'Hello world')
        result = entry.to_srt()
        assert '1\n' in result
        assert '00:00:01,000 --> 00:00:03,000' in result
        assert 'Hello world' in result


class TestParseSRT:
    def test_parse_basic(self):
        content = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld"
        entries = parse_srt(content)
        assert len(entries) == 2
        assert entries[0].text == 'Hello'
        assert entries[1].text == 'World'

    def test_parse_multiline_text(self):
        content = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two"
        entries = parse_srt(content)
        assert len(entries) == 1
        assert entries[0].text == 'Line one\nLine two'

    def test_parse_empty(self):
        assert parse_srt('') == []

    def test_parse_malformed_skipped(self):
        content = "bad\n\n1\n00:00:01,000 --> 00:00:03,000\nGood"
        entries = parse_srt(content)
        assert len(entries) == 1
        assert entries[0].text == 'Good'


class TestWriteSRT:
    def test_write_reindexes(self, tmp_path):
        entries = [
            SRTEntry(5, '00:00:01,000', '00:00:03,000', 'First'),
            SRTEntry(9, '00:00:04,000', '00:00:06,000', 'Second'),
        ]
        out = tmp_path / 'test.srt'
        write_srt(entries, out)
        content = out.read_text(encoding='utf-8')
        assert content.startswith('1\n')
        assert '2\n' in content


class TestWriteVTT:
    def test_write_vtt_header(self, tmp_path):
        entries = [SRTEntry(1, '00:00:01,000', '00:00:03,000', 'Hello')]
        out = tmp_path / 'test.vtt'
        write_vtt(entries, out)
        content = out.read_text(encoding='utf-8')
        assert content.startswith('WEBVTT\n')

    def test_write_vtt_dot_timestamps(self, tmp_path):
        entries = [SRTEntry(1, '00:00:01,500', '00:00:03,250', 'Test')]
        out = tmp_path / 'test.vtt'
        write_vtt(entries, out)
        content = out.read_text(encoding='utf-8')
        assert '00:00:01.500 --> 00:00:03.250' in content
        assert ',' not in content.split('WEBVTT')[1]

    def test_write_vtt_multiple_entries(self, tmp_path):
        entries = [
            SRTEntry(1, '00:00:01,000', '00:00:03,000', 'First'),
            SRTEntry(2, '00:00:04,000', '00:00:06,000', 'Second'),
        ]
        out = tmp_path / 'test.vtt'
        write_vtt(entries, out)
        content = out.read_text(encoding='utf-8')
        assert 'First' in content
        assert 'Second' in content

    def test_roundtrip_srt_to_vtt(self, tmp_path):
        """Parse SRT, write as VTT, verify format."""
        srt_content = "1\n00:00:01,000 --> 00:00:03,000\nHello world"
        entries = parse_srt(srt_content)
        vtt_path = tmp_path / 'out.vtt'
        write_vtt(entries, vtt_path)
        content = vtt_path.read_text(encoding='utf-8')
        assert content.startswith('WEBVTT')
        assert '00:00:01.000 --> 00:00:03.000' in content
        assert 'Hello world' in content
