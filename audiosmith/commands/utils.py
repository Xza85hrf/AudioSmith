"""Shared CLI utilities for AudioSmith commands."""

import sys

from rich.console import Console
from rich.panel import Panel

from audiosmith.exceptions import AudioSmithError

console = Console()


def handle_error(error: AudioSmithError, exit_code: int = 1) -> None:
    """Display an AudioSmith error and exit.

    Args:
        error: The AudioSmithError to display.
        exit_code: Exit code to use (default: 1).
    """
    console.print(f"[bold red]Error:[/bold red] {error.message}")
    sys.exit(exit_code)


def show_success_panel(title: str, content: str) -> None:
    """Display a success result panel.

    Args:
        title: Panel title.
        content: Panel content (can include rich formatting).
    """
    console.print(Panel(content, title=title, border_style="green"))
