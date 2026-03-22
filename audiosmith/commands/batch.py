"""Batch command — batch processing of multiple files."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from audiosmith.exceptions import AudioSmithError

console = Console()


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--target-lang', '-t', required=True, help='Target language code.')
@click.option('--source-lang', '-s', default='auto', help='Source language.')
@click.option('--output-dir', '-o', default=None, type=click.Path(), help='Output directory.')
@click.option('--continue-on-error', is_flag=True, help='Continue processing if a file fails.')
def batch(files, target_lang, source_lang, output_dir, continue_on_error):
    """Batch-dub multiple audio/video files."""
    from audiosmith.batch_processor import BatchProcessor
    from audiosmith.models import DubbingConfig

    file_paths = [Path(f) for f in files]
    out_dir = Path(output_dir) if output_dir else file_paths[0].parent / 'batch_output'

    try:
        config = DubbingConfig(
            video_path=file_paths[0],
            output_dir=out_dir,
            source_language=source_lang,
            target_language=target_lang,
        )
        with console.status(f"[bold cyan]Processing {len(file_paths)} files...[/bold cyan]", spinner="dots"):
            processor = BatchProcessor()
            results = processor.process(file_paths, config, continue_on_error=continue_on_error)
        summary = BatchProcessor.get_summary(results)

        t = Table(title="Batch Results", show_header=True, header_style="bold")
        t.add_column("Metric", width=20)
        t.add_column("Value", width=30)
        t.add_row("Total", str(summary['total']))
        t.add_row("Succeeded", f"[green]{summary['succeeded']}[/green]")
        t.add_row("Failed", f"[red]{summary['failed']}[/red]" if summary['failed'] else "0")
        t.add_row("Duration", f"{summary['total_duration_seconds']:.1f}s")
        if summary.get('failed_files'):
            t.add_row("Failed Files", ", ".join(summary['failed_files']))
        console.print(t)
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)
