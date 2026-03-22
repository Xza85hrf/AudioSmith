"""Allow running as `python -m audiosmith`."""

from audiosmith.cli import cli

# Import commands package to register all command modules with cli.
# This must happen after cli is defined but before cli() is called.
import audiosmith.commands  # noqa: F401

cli()
