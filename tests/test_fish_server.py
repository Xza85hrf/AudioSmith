"""Tests for audiosmith.fish_server module."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from audiosmith.fish_server import FishServerManager


class TestFishServerManagerIsHealthy:
    """Test is_healthy method."""

    def test_is_healthy_returns_true_on_200(self):
        """is_healthy returns True when server responds 200."""
        mgr = FishServerManager()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("audiosmith.fish_server.requests.get", return_value=mock_response):
            assert mgr.is_healthy() is True

    def test_is_healthy_returns_false_on_non_200(self):
        """is_healthy returns False when server responds non-200."""
        mgr = FishServerManager()
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("audiosmith.fish_server.requests.get", return_value=mock_response):
            assert mgr.is_healthy() is False

    def test_is_healthy_returns_false_on_connection_error(self):
        """is_healthy returns False on ConnectionError."""
        mgr = FishServerManager()

        with patch(
            "audiosmith.fish_server.requests.get",
            side_effect=Exception("Connection refused")
        ):
            assert mgr.is_healthy() is False

    def test_is_healthy_checks_correct_endpoint(self):
        """is_healthy checks the /v1/models endpoint."""
        mgr = FishServerManager(base_url="http://127.0.0.1:9999")
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("audiosmith.fish_server.requests.get", return_value=mock_response) as mock_get:
            mgr.is_healthy()
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "http://127.0.0.1:9999/v1/models" in str(call_args)


class TestFishServerManagerEnsureRunning:
    """Test ensure_running method."""

    def test_ensure_running_returns_true_if_already_healthy(self):
        """ensure_running returns True and doesn't start if already healthy."""
        mgr = FishServerManager()

        with patch.object(mgr, "is_healthy", return_value=True):
            with patch.object(mgr, "_start") as mock_start:
                result = mgr.ensure_running()
                assert result is True
                mock_start.assert_not_called()

    def test_ensure_running_calls_start_if_not_healthy(self):
        """ensure_running calls _start when not healthy."""
        mgr = FishServerManager()

        with patch.object(mgr, "is_healthy", return_value=False):
            with patch.object(mgr, "_start", return_value=True) as mock_start:
                result = mgr.ensure_running()
                assert result is True
                mock_start.assert_called_once()

    def test_ensure_running_returns_false_on_start_failure(self):
        """ensure_running returns False when _start fails."""
        mgr = FishServerManager()

        with patch.object(mgr, "is_healthy", return_value=False):
            with patch.object(mgr, "_start", return_value=False):
                result = mgr.ensure_running()
                assert result is False


class TestFishServerManagerStart:
    """Test _start method."""

    def test_start_returns_false_if_api_server_missing(self):
        """_start returns False if api_server.py doesn't exist."""
        from pathlib import Path
        mgr = FishServerManager(fish_dir=Path("/nonexistent"))

        result = mgr._start()
        assert result is False

    def test_start_returns_false_if_checkpoint_missing(self, tmp_path):
        """_start returns False if checkpoint doesn't exist."""
        fish_dir = tmp_path / "fish"
        fish_dir.mkdir()
        (fish_dir / "tools").mkdir()
        (fish_dir / "tools" / "api_server.py").write_text("# stub")

        mgr = FishServerManager(fish_dir=fish_dir)
        result = mgr._start()
        assert result is False

    def test_start_launches_subprocess_with_correct_cmd(self, tmp_path):
        """_start launches subprocess with correct command."""
        fish_dir = tmp_path / "fish"
        fish_dir.mkdir()
        (fish_dir / "tools").mkdir()
        (fish_dir / "tools" / "api_server.py").write_text("# stub")
        (fish_dir / "checkpoints" / "s2-pro").mkdir(parents=True)

        mgr = FishServerManager(fish_dir=fish_dir, device="cuda:0")

        with patch("audiosmith.fish_server.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            with patch.object(mgr, "_wait_for_health", return_value=True):
                mgr._start()

            # Verify Popen was called
            assert mock_popen.called
            call_kwargs = mock_popen.call_args[1]
            assert call_kwargs["cwd"] == str(fish_dir)

    def test_start_sets_we_started_flag(self, tmp_path):
        """_start sets _we_started flag to True."""
        fish_dir = tmp_path / "fish"
        fish_dir.mkdir()
        (fish_dir / "tools").mkdir()
        (fish_dir / "tools" / "api_server.py").write_text("# stub")
        (fish_dir / "checkpoints" / "s2-pro").mkdir(parents=True)

        mgr = FishServerManager(fish_dir=fish_dir)
        assert mgr._we_started is False

        with patch("audiosmith.fish_server.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            with patch.object(mgr, "_wait_for_health", return_value=True):
                mgr._start()
                assert mgr._we_started is True


class TestFishServerManagerStop:
    """Test stop method."""

    def test_stop_does_nothing_if_we_didnt_start(self):
        """stop is no-op if _we_started is False."""
        mgr = FishServerManager()
        mgr._we_started = False
        mgr._process = None

        # Should not raise
        mgr.stop()

    def test_stop_terminates_process_if_we_started(self):
        """stop terminates process if we started it."""
        mgr = FishServerManager()
        mgr._we_started = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mgr._process = mock_process

        mgr.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called()

    def test_stop_kills_process_if_terminate_fails(self):
        """stop kills process if terminate doesn't work."""
        mgr = FishServerManager()
        mgr._we_started = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 10)
        mgr._process = mock_process

        mgr.stop()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_stop_clears_process_reference(self):
        """stop sets _process to None."""
        mgr = FishServerManager()
        mgr._we_started = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mgr._process = mock_process

        mgr.stop()

        assert mgr._process is None
        assert mgr._we_started is False

    def test_stop_is_idempotent(self):
        """stop can be called multiple times safely."""
        mgr = FishServerManager()
        mgr._we_started = False
        mgr._process = None

        mgr.stop()
        mgr.stop()  # Should not raise


class TestFishServerManagerContextManager:
    """Test context manager protocol."""

    def test_context_manager_calls_ensure_running(self):
        """__enter__ calls ensure_running."""
        mgr = FishServerManager()

        with patch.object(mgr, "ensure_running") as mock_ensure:
            with mgr:
                mock_ensure.assert_called_once()

    def test_context_manager_returns_self(self):
        """__enter__ returns self."""
        mgr = FishServerManager()

        with patch.object(mgr, "ensure_running"):
            with mgr as result:
                assert result is mgr

    def test_context_manager_calls_stop_on_exit(self):
        """__exit__ calls stop."""
        mgr = FishServerManager()

        with patch.object(mgr, "ensure_running"):
            with patch.object(mgr, "stop") as mock_stop:
                with mgr:
                    pass
                mock_stop.assert_called_once()

    def test_context_manager_calls_stop_on_exception(self):
        """__exit__ calls stop even on exception."""
        mgr = FishServerManager()

        with patch.object(mgr, "ensure_running"):
            with patch.object(mgr, "stop") as mock_stop:
                try:
                    with mgr:
                        raise ValueError("test")
                except ValueError:
                    pass
                mock_stop.assert_called_once()


class TestFishServerManagerInit:
    """Test initialization."""

    def test_default_base_url(self):
        """Default base_url is http://127.0.0.1:8080."""
        mgr = FishServerManager()
        assert mgr.base_url == "http://127.0.0.1:8080"

    def test_custom_base_url(self):
        """Custom base_url is respected."""
        mgr = FishServerManager(base_url="http://localhost:9000")
        assert mgr.base_url == "http://localhost:9000"

    def test_default_checkpoint(self):
        """Default checkpoint is checkpoints/s2-pro."""
        mgr = FishServerManager()
        assert str(mgr.checkpoint) == "checkpoints/s2-pro"

    def test_custom_device(self):
        """Custom device is respected."""
        mgr = FishServerManager(device="cuda:0")
        assert mgr.device == "cuda:0"

    def test_process_starts_as_none(self):
        """_process starts as None."""
        mgr = FishServerManager()
        assert mgr._process is None

    def test_we_started_false_initially(self):
        """_we_started starts as False."""
        mgr = FishServerManager()
        assert mgr._we_started is False


class TestFishServerManagerWaitForHealth:
    """Test _wait_for_health method."""

    def test_wait_for_health_returns_true_on_health_check_success(self):
        """_wait_for_health returns True when health check succeeds."""
        mgr = FishServerManager(startup_timeout=5)
        mgr._process = MagicMock()
        mgr._process.poll.return_value = None

        with patch.object(mgr, "is_healthy", return_value=True):
            result = mgr._wait_for_health()
            assert result is True

    def test_wait_for_health_returns_false_on_process_exit(self):
        """_wait_for_health returns False if process exits."""
        mgr = FishServerManager(startup_timeout=5)
        mgr._process = MagicMock()
        mgr._process.poll.return_value = 1  # Exited with code 1
        mgr._process.stderr = MagicMock()
        mgr._process.stderr.read.return_value = b"error"

        result = mgr._wait_for_health()
        assert result is False

    def test_wait_for_health_returns_false_on_timeout(self):
        """_wait_for_health returns False on timeout."""
        mgr = FishServerManager(startup_timeout=1)
        mgr._process = MagicMock()
        mgr._process.poll.return_value = None

        with patch.object(mgr, "is_healthy", return_value=False):
            with patch("audiosmith.fish_server.time.sleep"):
                result = mgr._wait_for_health()
                assert result is False

    def test_wait_for_health_calls_stop_on_timeout(self):
        """_wait_for_health calls stop on timeout."""
        mgr = FishServerManager(startup_timeout=1)
        mgr._process = MagicMock()
        mgr._process.poll.return_value = None

        with patch.object(mgr, "is_healthy", return_value=False):
            with patch.object(mgr, "stop") as mock_stop:
                with patch("audiosmith.fish_server.time.sleep"):
                    mgr._wait_for_health()
                    mock_stop.assert_called_once()
