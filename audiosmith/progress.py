"""Progress tracking with rich fallback to logging."""

import logging
import sys
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int, str], None]


class ProgressTracker:
    """Tracks pipeline progress with optional rich console output."""

    def __init__(
        self,
        total_steps: int = 6,
        use_rich: Optional[bool] = None,
        callback: Optional[ProgressCallback] = None,
    ):
        self.total_steps = total_steps
        self.callback = callback
        self._current_step = 0
        self._current_task = None
        self._progress = None
        self._started = False
        self._rich_available = False

        if use_rich is None:
            use_rich = self._is_tty()

        if use_rich:
            try:
                from rich.progress import (
                    Progress, SpinnerColumn, TextColumn,
                    BarColumn, TaskProgressColumn,
                )
                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    transient=True,
                )
                self._rich_available = True
            except ImportError:
                pass

    @staticmethod
    def _is_tty() -> bool:
        """Check if stdout is connected to a terminal."""
        return sys.stdout.isatty()

    @contextmanager
    def step(self, name: str, total: int = 0):
        """Context manager for tracking a pipeline step."""
        self._current_step += 1
        step_label = f"[{self._current_step}/{self.total_steps}] {name}"

        if self._rich_available and self._progress:
            if not self._started:
                self._progress.start()
                self._started = True
            task_id = self._progress.add_task(step_label, total=total or None)
            self._current_task = task_id
            try:
                yield
            finally:
                self._progress.update(task_id, completed=total if total > 0 else 1)
        else:
            logger.info("Starting: %s", step_label)
            try:
                yield
            finally:
                logger.info("Completed: %s", step_label)

        if self.callback:
            self.callback(name, self._current_step, self.total_steps, "complete")

    def update(self, increment: int = 1, message: str = ""):
        """Update progress for the current step."""
        if self._rich_available and self._progress and self._current_task is not None:
            current = self._progress.tasks[self._current_task].completed or 0
            self._progress.update(self._current_task, completed=current + increment)
        if message:
            logger.info("Progress: %s", message)

    def complete(self):
        """Finalize all tracking."""
        if self._rich_available and self._progress and self._started:
            self._progress.stop()
            self._started = False
