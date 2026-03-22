"""AudioSmith CLI commands subpackage.

This package organizes CLI commands into focused modules while maintaining
backward compatibility. The main cli group is defined in audiosmith.cli and
commands are registered here.

The key pattern: cli group must be defined BEFORE commands are registered.
Since these are separate modules, we import cli from audiosmith.cli, then
manually add command functions to it using cli.add_command().
"""

# Import the cli group from the main cli module
from audiosmith.cli import cli

# Import command modules to get their click.command() objects
from audiosmith.commands.dub import dub
from audiosmith.commands.tts_cmd import tts
from audiosmith.commands.transcribe import transcribe, translate, transcribe_url
from audiosmith.commands.batch import batch

# Register commands with the cli group
cli.add_command(dub)
cli.add_command(tts)
cli.add_command(transcribe)
cli.add_command(translate)
cli.add_command(transcribe_url)
cli.add_command(batch)

# Expose the cli group for external use
__all__ = ['cli', 'dub', 'tts', 'transcribe', 'translate', 'transcribe_url', 'batch']
