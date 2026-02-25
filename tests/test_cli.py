"""Tests for audiosmith.cli module."""

from unittest.mock import patch, MagicMock
from pathlib import Path
from click.testing import CliRunner
from audiosmith.cli import cli


class TestCLI:
    def test_help(self):
        result = CliRunner().invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'AudioSmith' in result.output

    def test_all_commands_in_help(self):
        result = CliRunner().invoke(cli, ['--help'])
        for cmd in ['dub', 'transcribe', 'translate', 'batch', 'export', 'normalize', 'check', 'tts', 'extract-voices']:
            assert cmd in result.output

    def test_dub_help(self):
        result = CliRunner().invoke(cli, ['dub', '--help'])
        assert result.exit_code == 0
        assert '--target-lang' in result.output

    def test_transcribe_help(self):
        result = CliRunner().invoke(cli, ['transcribe', '--help'])
        assert result.exit_code == 0
        assert '--model' in result.output

    def test_translate_help(self):
        result = CliRunner().invoke(cli, ['translate', '--help'])
        assert result.exit_code == 0
        assert '--target-lang' in result.output

    def test_transcribe_url_help(self):
        result = CliRunner().invoke(cli, ['transcribe-url', '--help'])
        assert result.exit_code == 0

    def test_dub_missing_target_lang(self):
        result = CliRunner().invoke(cli, ['dub', 'nonexistent.mp4'])
        assert result.exit_code != 0


class TestCheckCommand:
    def test_check_pass(self):
        mock = MagicMock()
        mock.run_all_checks.return_value = {'ffmpeg': True, 'torch': True, 'cuda': True,
                                             'faster_whisper': True, 'disk_space': True}
        mock.get_summary.return_value = 'System Pre-Flight Checks:\n  ffmpeg               PASS'
        with patch('audiosmith.system_check.SystemChecker', return_value=mock):
            result = CliRunner().invoke(cli, ['check'])
        assert result.exit_code == 0
        assert 'PASS' in result.output

    def test_check_with_failures(self):
        mock = MagicMock()
        mock.run_all_checks.return_value = {'ffmpeg': False, 'torch': False, 'cuda': False,
                                             'faster_whisper': False, 'disk_space': True}
        mock.get_summary.return_value = 'System Pre-Flight Checks:\n  ffmpeg               FAIL'
        with patch('audiosmith.system_check.SystemChecker', return_value=mock):
            result = CliRunner().invoke(cli, ['check'])
        assert 'Warning' in result.output


class TestNormalizeCommand:
    def test_normalize(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.touch()
        mock = MagicMock()
        mock.analyze.return_value = {'lufs': -20.0, 'peak_db': -3.0}
        with patch('audiosmith.audio_normalizer.AudioNormalizer', return_value=mock):
            result = CliRunner().invoke(cli, ['normalize', str(audio)])
        assert result.exit_code == 0
        assert 'LUFS' in result.output


class TestExportCommand:
    def test_export_txt(self, tmp_path):
        srt = tmp_path / "test.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:05,000\nHello\n")
        mock_entry = MagicMock(index=1, start_time=1.0, end_time=5.0, text='Hello')
        mock_fmt = MagicMock()
        with patch('audiosmith.srt.parse_srt_file', return_value=[mock_entry]), \
             patch('audiosmith.document_formatter.DocumentFormatter', return_value=mock_fmt):
            result = CliRunner().invoke(cli, ['export', str(srt), '-f', 'txt'])
        assert result.exit_code == 0
        assert 'Exported' in result.output


class TestBatchCommand:
    def test_batch_no_files(self):
        result = CliRunner().invoke(cli, ['batch', '--target-lang', 'pl'])
        assert result.exit_code != 0

    def test_batch_help(self):
        result = CliRunner().invoke(cli, ['batch', '--help'])
        assert '--continue-on-error' in result.output


class TestTTSCommand:
    def test_tts_missing_output(self):
        result = CliRunner().invoke(cli, ['tts', 'Hello'])
        assert result.exit_code != 0

    def test_tts_help(self):
        result = CliRunner().invoke(cli, ['tts', '--help'])
        assert '--engine' in result.output
        assert 'piper' in result.output
        assert 'qwen3' in result.output


class TestExtractVoicesCommand:
    def test_help(self):
        result = CliRunner().invoke(cli, ['extract-voices', '--help'])
        assert result.exit_code == 0
        assert '--diarize' in result.output
        assert '--num-samples' in result.output
        assert '--catalog' in result.output

    def test_extract_evenly(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.touch()
        mock_sample = MagicMock(
            speaker_id="voice_00", sample_path=Path("v.wav"), duration=5.0,
            mean_volume_db=-15.0,
        )
        mock_catalog = MagicMock()
        mock_catalog.samples = [mock_sample]
        mock_catalog.get_speakers.return_value = ["voice_00"]
        mock_catalog.get_best_sample.return_value = mock_sample
        mock_extractor = MagicMock()
        mock_extractor.extract_evenly.return_value = mock_catalog
        with patch('audiosmith.voice_extractor.VoiceExtractor', return_value=mock_extractor), \
             patch('audiosmith.voice_extractor.create_voice_profiles', return_value={"voice_00": {}}):
            result = CliRunner().invoke(cli, ['extract-voices', str(audio), '-n', '3'])
        assert result.exit_code == 0
        assert 'Extracted 1 samples' in result.output

    def test_extract_with_diarize(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.touch()
        mock_catalog = MagicMock()
        mock_catalog.samples = []
        mock_catalog.get_speakers.return_value = []
        mock_extractor = MagicMock()
        mock_extractor.extract_with_diarization.return_value = mock_catalog
        with patch('audiosmith.voice_extractor.VoiceExtractor', return_value=mock_extractor), \
             patch('audiosmith.voice_extractor.create_voice_profiles', return_value={}):
            result = CliRunner().invoke(cli, ['extract-voices', str(audio), '--diarize'])
        assert result.exit_code == 0
        assert 'Extracted 0 samples' in result.output
