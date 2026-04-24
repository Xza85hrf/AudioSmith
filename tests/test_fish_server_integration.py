"""Integration tests for Fish Speech server manager in the pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from audiosmith.models import DubbingConfig
from audiosmith.pipeline.core import DubbingPipeline


class TestFishServerManagerIntegration:
    """Test Fish Speech server manager integration with pipeline."""

    def test_init_fish_engine_starts_server_when_base_url_set(self, tmp_path):
        """_init_fish_engine starts the server when fish_base_url is configured."""
        config = DubbingConfig(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path / "output",
            target_language="en",
            fish_base_url="http://127.0.0.1:8080",
        )
        pipeline = DubbingPipeline(config)

        with patch("audiosmith.fish_server.FishServerManager") as mock_mgr_class:
            mock_mgr = MagicMock()
            mock_mgr.ensure_running.return_value = True
            mock_mgr_class.return_value = mock_mgr

            with patch("audiosmith.fish_speech_tts.FishSpeechTTS"):
                engine, sr = pipeline._init_fish_engine()

            # Verify manager was created with correct URL
            mock_mgr_class.assert_called_once_with(
                base_url="http://127.0.0.1:8080"
            )
            # Verify ensure_running was called
            mock_mgr.ensure_running.assert_called_once()

    def test_init_fish_engine_stores_manager_on_pipeline(self, tmp_path):
        """_init_fish_engine stores the manager instance on the pipeline."""
        config = DubbingConfig(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path / "output",
            target_language="en",
            fish_base_url="http://127.0.0.1:8080",
        )
        pipeline = DubbingPipeline(config)

        with patch("audiosmith.fish_server.FishServerManager") as mock_mgr_class:
            mock_mgr = MagicMock()
            mock_mgr.ensure_running.return_value = True
            mock_mgr_class.return_value = mock_mgr

            with patch("audiosmith.fish_speech_tts.FishSpeechTTS"):
                pipeline._init_fish_engine()

            # Verify manager is stored on pipeline
            assert hasattr(pipeline, "_fish_server_mgr")
            assert pipeline._fish_server_mgr is mock_mgr

    def test_init_fish_engine_logs_warning_on_startup_failure(self, tmp_path, caplog):
        """_init_fish_engine logs warning when server startup fails."""
        import logging
        caplog.set_level(logging.WARNING)

        config = DubbingConfig(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path / "output",
            target_language="en",
            fish_base_url="http://127.0.0.1:8080",
        )
        pipeline = DubbingPipeline(config)

        with patch("audiosmith.fish_server.FishServerManager") as mock_mgr_class:
            mock_mgr = MagicMock()
            mock_mgr.ensure_running.return_value = False
            mock_mgr_class.return_value = mock_mgr

            with patch("audiosmith.fish_speech_tts.FishSpeechTTS"):
                pipeline._init_fish_engine()

            # Verify warning was logged
            assert "Fish Speech server not available" in caplog.text

    def test_init_fish_engine_skips_manager_when_no_base_url(self, tmp_path):
        """_init_fish_engine skips server startup when no fish_base_url."""
        config = DubbingConfig(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path / "output",
            target_language="en",
            fish_base_url=None,  # No local server
        )
        pipeline = DubbingPipeline(config)

        with patch("audiosmith.fish_server.FishServerManager") as mock_mgr_class:
            with patch("audiosmith.fish_speech_tts.FishSpeechTTS"):
                pipeline._init_fish_engine()

            # Verify manager was NOT created
            mock_mgr_class.assert_not_called()
            # Verify manager is not stored
            assert not hasattr(pipeline, "_fish_server_mgr")
