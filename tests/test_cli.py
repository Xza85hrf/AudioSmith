"""Tests for audiosmith.cli module."""

from click.testing import CliRunner
from audiosmith.cli import cli


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'AudioSmith' in result.output

    def test_dub_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['dub', '--help'])
        assert result.exit_code == 0
        assert '--target-lang' in result.output
        assert '--resume' in result.output

    def test_transcribe_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['transcribe', '--help'])
        assert result.exit_code == 0
        assert '--model' in result.output
        assert '--language' in result.output

    def test_translate_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['translate', '--help'])
        assert result.exit_code == 0
        assert '--target-lang' in result.output
        assert '--backend' in result.output

    def test_transcribe_url_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['transcribe-url', '--help'])
        assert result.exit_code == 0
        assert '--output' in result.output

    def test_dub_missing_target_lang(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['dub', 'nonexistent.mp4'])
        assert result.exit_code != 0
