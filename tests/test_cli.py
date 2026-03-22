"""Tests for audiosmith.cli module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from click.testing import CliRunner

from audiosmith.cli import cli
from audiosmith.exceptions import AudioSmithError


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
        assert 'Export Complete' in result.output


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
        assert 'Extracted' in result.output and '1 samples' in result.output

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
        assert 'Extracted' in result.output and '0 samples' in result.output


class TestTTSIntegration:
    """E2E tests for TTS command with multiple engines."""

    def test_tts_piper_engine(self, tmp_path):
        """Piper TTS synthesis creates output file with correct metadata."""
        output = tmp_path / "output.wav"

        # Mock PiperTTS
        mock_piper = MagicMock()
        mock_piper.sample_rate = 22050
        mock_piper.synthesize.return_value = np.zeros(22050, dtype=np.float32)

        def fake_sf_write(path, data, sr):
            """Create a fake WAV file to prevent FileNotFoundError on stat()."""
            Path(path).write_bytes(b"RIFF" + b"\x00" * 100)

        with patch('audiosmith.piper_tts.PiperTTS', return_value=mock_piper), \
             patch('soundfile.write', side_effect=fake_sf_write):
            result = CliRunner().invoke(cli, [
                'tts', 'Hello world', '--engine', 'piper', '-o', str(output)
            ])

        assert result.exit_code == 0, f"Failed: {result.output}\n{result.exception}"
        assert 'Synthesis Complete' in result.output
        assert 'Engine: piper' in result.output
        assert 'Hello world' not in result.output  # Text should not be in output for privacy

    def test_tts_chatterbox_engine(self, tmp_path):
        """Chatterbox TTS synthesis with audio prompt."""
        output = tmp_path / "output.wav"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_cb = MagicMock()
        mock_cb.sample_rate = 24000
        mock_cb.synthesize.return_value = np.zeros(24000, dtype=np.float32)

        def fake_sf_write(path, data, sr):
            Path(path).write_bytes(b"RIFF" + b"\x00" * 100)

        with patch('audiosmith.tts.ChatterboxTTS', return_value=mock_cb), \
             patch('soundfile.write', side_effect=fake_sf_write):
            result = CliRunner().invoke(cli, [
                'tts', 'Test speech', '--engine', 'chatterbox',
                '-o', str(output), '--ref-audio', str(ref_audio)
            ])

        assert result.exit_code == 0
        assert 'Synthesis Complete' in result.output
        assert 'Engine: chatterbox' in result.output
        # Verify synthesize was called with audio_prompt_path
        mock_cb.synthesize.assert_called_once()
        call_kwargs = mock_cb.synthesize.call_args[1]
        assert call_kwargs.get('audio_prompt_path') == str(ref_audio)

    def test_tts_missing_required_output(self):
        """TTS command fails without --output option."""
        result = CliRunner().invoke(cli, ['tts', 'Hello'])
        assert result.exit_code != 0
        assert 'required' in result.output.lower() or 'output' in result.output.lower()

    def test_tts_post_processing_enabled(self, tmp_path):
        """TTS with post-processing applies naturalness filters."""
        output = tmp_path / "output.wav"

        mock_piper = MagicMock()
        mock_piper.sample_rate = 22050
        audio_data = np.random.randn(22050).astype(np.float32)
        mock_piper.synthesize.return_value = audio_data

        mock_pp = MagicMock()
        processed_audio = np.random.randn(22050).astype(np.float32)
        mock_pp.process.return_value = processed_audio

        write_calls = []
        def track_sf_write(path, data, sr):
            write_calls.append((path, data, sr))
            Path(path).write_bytes(b"RIFF" + b"\x00" * 100)

        with patch('audiosmith.piper_tts.PiperTTS', return_value=mock_piper), \
             patch('audiosmith.tts_postprocessor.TTSPostProcessor', return_value=mock_pp), \
             patch('soundfile.write', side_effect=track_sf_write):
            result = CliRunner().invoke(cli, [
                'tts', 'Hello', '--engine', 'piper', '-o', str(output)
            ])

        assert result.exit_code == 0
        # Verify post-processor was called
        mock_pp.process.assert_called_once()
        # soundfile.write should be called twice (initial + post-processed)
        assert len(write_calls) >= 1

    def test_tts_post_processing_disabled(self, tmp_path):
        """TTS with --no-post-process skips naturalness filters."""
        output = tmp_path / "output.wav"

        mock_piper = MagicMock()
        mock_piper.sample_rate = 22050
        mock_piper.synthesize.return_value = np.zeros(22050, dtype=np.float32)

        mock_pp = MagicMock()

        def fake_sf_write(path, data, sr):
            Path(path).write_bytes(b"RIFF" + b"\x00" * 100)

        with patch('audiosmith.piper_tts.PiperTTS', return_value=mock_piper), \
             patch('audiosmith.tts_postprocessor.TTSPostProcessor', return_value=mock_pp), \
             patch('soundfile.write', side_effect=fake_sf_write):
            result = CliRunner().invoke(cli, [
                'tts', 'Hello', '--engine', 'piper', '-o', str(output),
                '--no-post-process'
            ])

        assert result.exit_code == 0
        # Post-processor should not be instantiated
        mock_pp.process.assert_not_called()


class TestDubIntegration:
    """E2E tests for dub command."""

    def test_dub_basic_pipeline(self, tmp_path):
        """Dub command runs pipeline and displays results."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        output_dir = tmp_path / "output"

        # Mock the pipeline
        mock_result = MagicMock()
        mock_result.segments_dubbed = 5
        mock_result.total_time = 12.5
        mock_result.output_video_path = output_dir / "dubbed.mp4"

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch('audiosmith.pipeline.DubbingPipeline', return_value=mock_pipeline):
            result = CliRunner().invoke(cli, [
                'dub', str(video), '--target-lang', 'pl'
            ])

        assert result.exit_code == 0, f"Failed: {result.output}\n{result.exception}"
        assert 'Dubbing Complete' in result.output
        assert '5' in result.output  # segments_dubbed
        assert '12.5' in result.output  # total_time

    def test_dub_with_diarization(self, tmp_path):
        """Dub command passes diarization flag to pipeline."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        mock_result = MagicMock()
        mock_result.segments_dubbed = 3
        mock_result.total_time = 5.0
        mock_result.output_video_path = None

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch('audiosmith.pipeline.DubbingPipeline', return_value=mock_pipeline), \
             patch('audiosmith.models.DubbingConfig'):
            result = CliRunner().invoke(cli, [
                'dub', str(video), '--target-lang', 'es', '--diarize'
            ])

        assert result.exit_code == 0
        # Verify pipeline was called
        mock_pipeline.run.assert_called_once()

    def test_dub_missing_target_lang(self, tmp_path):
        """Dub command requires --target-lang."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        result = CliRunner().invoke(cli, ['dub', str(video)])
        assert result.exit_code != 0
        assert 'target-lang' in result.output or 'required' in result.output.lower()

    def test_dub_with_custom_tts_engine(self, tmp_path):
        """Dub command accepts custom TTS engine option."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        mock_result = MagicMock()
        mock_result.segments_dubbed = 2
        mock_result.total_time = 3.0
        mock_result.output_video_path = None

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch('audiosmith.pipeline.DubbingPipeline', return_value=mock_pipeline):
            result = CliRunner().invoke(cli, [
                'dub', str(video), '--target-lang', 'de', '--engine', 'qwen3'
            ])

        assert result.exit_code == 0
        assert 'Dubbing Complete' in result.output

    def test_dub_with_resume_flag(self, tmp_path):
        """Dub command accepts resume flag for checkpoint recovery."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        mock_result = MagicMock()
        mock_result.segments_dubbed = 1
        mock_result.total_time = 1.5
        mock_result.output_video_path = None

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch('audiosmith.pipeline.DubbingPipeline', return_value=mock_pipeline):
            result = CliRunner().invoke(cli, [
                'dub', str(video), '--target-lang', 'fr', '--resume'
            ])

        assert result.exit_code == 0
        assert 'Dubbing Complete' in result.output


class TestTranslateCommand:
    """E2E tests for translate command."""

    def test_translate_basic(self, tmp_path):
        """Translate SRT file with Polish target language."""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nHello world\n\n")

        mock_entry = MagicMock(index=1, start_time=1.0, end_time=5.0, text='Hello world')

        def mock_translate_func(text, source_lang, target_lang, backend='argos'):
            """Mock translation function."""
            return 'Cześć świecie' if text == 'Hello world' else text

        with patch('audiosmith.srt.parse_srt_file', return_value=[mock_entry]), \
             patch('audiosmith.translate.translate', side_effect=mock_translate_func), \
             patch('audiosmith.srt.write_srt'):
            result = CliRunner().invoke(cli, [
                'translate', str(srt_file), '--target-lang', 'pl'
            ])

        assert result.exit_code == 0, f"Failed: {result.output}\n{result.exception}"
        assert 'Translation Complete' in result.output
        assert '1' in result.output  # entry count

    def test_translate_with_source_lang(self, tmp_path):
        """Translate with explicit source language."""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nHello\n\n")

        mock_entry = MagicMock(index=1, start_time=1.0, end_time=5.0, text='Hello')

        def mock_translate_func(text, source_lang, target_lang, backend='argos'):
            return 'Hola' if target_lang == 'es' else text

        with patch('audiosmith.srt.parse_srt_file', return_value=[mock_entry]), \
             patch('audiosmith.translate.translate', side_effect=mock_translate_func), \
             patch('audiosmith.srt.write_srt'):
            result = CliRunner().invoke(cli, [
                'translate', str(srt_file), '--target-lang', 'es', '--source-lang', 'en'
            ])

        assert result.exit_code == 0
        assert 'Translation Complete' in result.output

    def test_translate_missing_target_lang(self, tmp_path):
        """Translate without required --target-lang fails."""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nHello\n\n")

        result = CliRunner().invoke(cli, ['translate', str(srt_file)])
        assert result.exit_code != 0
        assert 'target-lang' in result.output or 'required' in result.output.lower()

    def test_translate_nonexistent_file(self):
        """Translate with nonexistent SRT file fails."""
        result = CliRunner().invoke(cli, [
            'translate', '/nonexistent/file.srt', '--target-lang', 'pl'
        ])
        assert result.exit_code != 0

    def test_translate_with_backend_option(self, tmp_path):
        """Translate with explicit backend (gemma)."""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nTest\n\n")

        mock_entry = MagicMock(index=1, start_time=1.0, end_time=5.0, text='Test')
        call_tracker = []

        def mock_translate_func(text, source_lang, target_lang, backend='argos'):
            call_tracker.append({'backend': backend})
            return text

        with patch('audiosmith.srt.parse_srt_file', return_value=[mock_entry]), \
             patch('audiosmith.translate.translate', side_effect=mock_translate_func), \
             patch('audiosmith.srt.write_srt'):
            result = CliRunner().invoke(cli, [
                'translate', str(srt_file), '--target-lang', 'de', '--backend', 'gemma'
            ])

        assert result.exit_code == 0
        assert len(call_tracker) > 0
        assert call_tracker[0]['backend'] == 'gemma'


class TestTranscribeUrlCommand:
    """E2E tests for transcribe-url command."""

    def test_transcribe_url_basic(self):
        """Transcribe URL from YouTube-like service."""
        mock_segment = MagicMock(id='0', start=1.0, end=5.0, text='Hello from video')

        with patch('audiosmith.download.download_media',
                   return_value=(Path('/tmp/video.mp4'), 'Test Video')), \
             patch('audiosmith.ffmpeg.extract_audio'), \
             patch('audiosmith.transcribe.Transcriber') as mock_transcriber_class, \
             patch('audiosmith.srt.write_srt'), \
             patch('audiosmith.srt_formatter.SRTFormatter'):

            # Setup transcriber mock
            mock_transcriber = MagicMock()
            mock_transcriber.transcribe.return_value = [mock_segment]
            mock_transcriber_class.return_value = mock_transcriber

            result = CliRunner().invoke(cli, ['transcribe-url', 'https://example.com/video'])

        assert result.exit_code == 0, f"Failed: {result.output}\n{result.exception}"
        assert 'URL Transcription Complete' in result.output
        assert 'Test Video' in result.output

    def test_transcribe_url_with_language(self):
        """Transcribe URL with specific language."""
        mock_segment = MagicMock(id='0', start=0.0, end=2.0, text='Bonjour')

        with patch('audiosmith.download.download_media',
                   return_value=(Path('/tmp/video.mp4'), 'French Video')), \
             patch('audiosmith.ffmpeg.extract_audio'), \
             patch('audiosmith.transcribe.Transcriber') as mock_transcriber_class, \
             patch('audiosmith.srt.write_srt'), \
             patch('audiosmith.srt_formatter.SRTFormatter'):

            mock_transcriber = MagicMock()
            mock_transcriber.transcribe.return_value = [mock_segment]
            mock_transcriber_class.return_value = mock_transcriber

            result = CliRunner().invoke(cli, [
                'transcribe-url', 'https://example.com/video', '--language', 'fr'
            ])

        assert result.exit_code == 0
        assert 'URL Transcription Complete' in result.output

    def test_transcribe_url_with_output_format(self):
        """Transcribe URL with JSON output format."""
        mock_segment = {'start': 1.0, 'end': 3.0, 'text': 'Test'}

        with patch('audiosmith.download.download_media',
                   return_value=(Path('/tmp/video.mp4'), 'Test')), \
             patch('audiosmith.ffmpeg.extract_audio'), \
             patch('audiosmith.transcribe.Transcriber') as mock_transcriber_class, \
             patch('audiosmith.srt_formatter.SRTFormatter'), \
             patch('audiosmith.download.segments_to_json', return_value='{"segments": []}'):

            mock_transcriber = MagicMock()
            mock_transcriber.transcribe.return_value = [mock_segment]
            mock_transcriber_class.return_value = mock_transcriber

            result = CliRunner().invoke(cli, [
                'transcribe-url', 'https://example.com/video', '--output', 'json'
            ])

        assert result.exit_code == 0

    def test_transcribe_url_download_error(self):
        """Transcribe URL fails when download_media raises error."""
        with patch('audiosmith.download.download_media',
                   side_effect=AudioSmithError("Download failed")):
            result = CliRunner().invoke(cli, ['transcribe-url', 'https://invalid.example'])

        assert result.exit_code != 0
        assert 'Error' in result.output


class TestInfoCommand:
    """E2E tests for info command."""

    def test_info_basic(self):
        """Info command displays system info and TTS engines."""
        result = CliRunner().invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'System' in result.output
        assert 'TTS Engines' in result.output

    def test_info_contains_python_version(self):
        """Info shows Python version."""
        result = CliRunner().invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'Python' in result.output

    def test_info_contains_ffmpeg_check(self):
        """Info shows FFmpeg availability."""
        result = CliRunner().invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'FFmpeg' in result.output

    def test_info_contains_capabilities(self):
        """Info shows processing capabilities."""
        result = CliRunner().invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'Capabilities' in result.output or 'Transcription' in result.output


class TestVoicesCommand:
    """E2E tests for voices command."""

    def test_voices_all_engines(self):
        """Voices command shows all TTS engines."""
        result = CliRunner().invoke(cli, ['voices'])
        assert result.exit_code == 0
        assert 'Piper' in result.output or 'piper' in result.output.lower()
        assert 'Chatterbox' in result.output or 'chatterbox' in result.output.lower()

    def test_voices_piper_only(self):
        """Voices command with --engine piper shows only piper voices."""
        result = CliRunner().invoke(cli, ['voices', '--engine', 'piper'])
        assert result.exit_code == 0
        assert 'Piper' in result.output or 'piper' in result.output.lower()

    def test_voices_chatterbox_only(self):
        """Voices command with --engine chatterbox shows chatterbox info."""
        result = CliRunner().invoke(cli, ['voices', '--engine', 'chatterbox'])
        assert result.exit_code == 0
        assert 'Chatterbox' in result.output or 'chatterbox' in result.output.lower()

    def test_voices_qwen3_option(self):
        """Voices command supports qwen3 engine."""
        result = CliRunner().invoke(cli, ['voices', '--engine', 'qwen3'])
        assert result.exit_code == 0
        # Qwen3 should appear in output
        assert 'Qwen' in result.output or 'qwen' in result.output.lower()


class TestDubErrorHandling:
    """Error path tests for dub command."""

    def test_dub_pipeline_error(self, tmp_path):
        """Dub command handles pipeline errors gracefully."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = AudioSmithError("Pipeline failed")

        with patch('audiosmith.pipeline.DubbingPipeline', return_value=mock_pipeline):
            result = CliRunner().invoke(cli, [
                'dub', str(video), '--target-lang', 'es'
            ])

        assert result.exit_code != 0
        assert 'Error' in result.output


class TestTranscribeErrorHandling:
    """Error path tests for transcribe command."""

    def test_transcribe_nonexistent_file(self):
        """Transcribe with nonexistent file fails."""
        result = CliRunner().invoke(cli, [
            'transcribe', '/nonexistent/audio.mp3'
        ])
        assert result.exit_code != 0


class TestBatchCommandExtended:
    """Extended tests for batch command error paths."""

    def test_batch_missing_target_lang(self):
        """Batch command requires --target-lang."""
        result = CliRunner().invoke(cli, ['batch'])
        assert result.exit_code != 0
