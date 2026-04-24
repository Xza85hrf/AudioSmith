"""Fish Speech local server lifecycle management."""

from __future__ import annotations

import atexit
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger("audiosmith.fish_server")

# Default paths
FISH_SPEECH_DIR = Path("/tmp/fish-speech")
FISH_CHECKPOINT = "checkpoints/s2-pro"


class FishServerManager:
    """Manages the local Fish Speech S2-Pro server lifecycle.

    Checks if the server is already running, starts it if needed,
    and provides graceful shutdown.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        fish_dir: Optional[Path] = None,
        checkpoint: str = FISH_CHECKPOINT,
        device: str = "cuda:1",
        startup_timeout: int = 120,
    ) -> None:
        self.base_url = base_url
        self.fish_dir = fish_dir or FISH_SPEECH_DIR
        self.checkpoint = checkpoint
        self.device = device
        self.startup_timeout = startup_timeout
        self._process: Optional[subprocess.Popen] = None
        self._we_started = False

    def is_healthy(self) -> bool:
        """Check if the Fish Speech server is responding."""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout, Exception):
            return False

    def ensure_running(self) -> bool:
        """Ensure the server is running. Start it if needed.

        Returns True if the server is healthy, False if startup failed.
        """
        if self.is_healthy():
            logger.info("Fish Speech server already running at %s", self.base_url)
            return True

        return self._start()

    def _start(self) -> bool:
        """Start the Fish Speech server as a subprocess."""
        api_server = self.fish_dir / "tools" / "api_server.py"
        if not api_server.exists():
            logger.error(
                "Fish Speech not found at %s. Clone the repo first.", self.fish_dir,
            )
            return False

        checkpoint_path = self.fish_dir / self.checkpoint
        if not checkpoint_path.exists():
            logger.error(
                "Fish Speech checkpoint not found at %s. Download the model first.",
                checkpoint_path,
            )
            return False

        # Parse port from base_url
        parsed = urlparse(self.base_url)
        port = parsed.port or 8080

        cmd = [
            "uv", "run", "python", "tools/api_server.py",
            "--llama-checkpoint-path", self.checkpoint,
            "--device", self.device,
            "--listen", f"0.0.0.0:{port}",
            "--compile",
        ]

        logger.info("Starting Fish Speech server: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            cwd=str(self.fish_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._we_started = True

        # Register shutdown hook
        atexit.register(self.stop)

        # Wait for server to become healthy
        return self._wait_for_health()

    def _wait_for_health(self) -> bool:
        """Poll the health endpoint until the server is ready."""
        start = time.monotonic()
        interval = 2.0

        while time.monotonic() - start < self.startup_timeout:
            if self._process and self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                logger.error("Fish Speech server exited during startup: %s", stderr[:500])
                return False

            if self.is_healthy():
                elapsed = time.monotonic() - start
                logger.info("Fish Speech server ready in %.1fs", elapsed)
                return True

            time.sleep(interval)

        logger.error(
            "Fish Speech server failed to start within %ds", self.startup_timeout,
        )
        self.stop()
        return False

    def stop(self) -> None:
        """Stop the server if we started it."""
        if not self._we_started or self._process is None:
            return

        if self._process.poll() is not None:
            self._process = None
            return

        logger.info("Stopping Fish Speech server (pid=%d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Fish Speech server didn't stop gracefully, killing")
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("Fish Speech server failed to terminate after kill")

        self._process = None
        self._we_started = False

    def __enter__(self) -> FishServerManager:
        self.ensure_running()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
