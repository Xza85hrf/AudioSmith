"""Transcribe commands — transcription, translation, URL-based transcription."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from audiosmith.exceptions import AudioSmithError

console = Console()


@click.command()
@click.argument('audio', type=click.Path(exists=True))
@click.option('--output', '-o', default='srt', type=click.Choice(['srt', 'txt', 'vtt', 'json']),
              help='Output format.')
@click.option('--language', '-l', default=None, help='Audio language (auto-detect if omitted).')
@click.option('--model', '-m', default='large-v3', help='Whisper model size.')
@click.option('--diarize', is_flag=True, help='Enable speaker diarization.')
@click.option('--isolate-vocals', is_flag=True, help='Isolate vocals before transcription.')
def transcribe(audio, output, language, model, diarize, isolate_vocals):
    """Transcribe an audio or video file."""
    from audiosmith.transcribe import Transcriber

    audio_path = Path(audio)
    try:
        if isolate_vocals:
            from audiosmith.vocal_isolator import VocalIsolator
            with console.status("[bold cyan]Isolating vocals...[/bold cyan]", spinner="dots"):
                vi = VocalIsolator()
                paths = vi.isolate(audio_path)
                vi.unload()
            audio_path = paths['vocals_path']
            console.print("[dim]Vocal isolation complete[/dim]")

        diar_segments = None
        if diarize:
            from audiosmith.diarizer import Diarizer
            with console.status("[bold cyan]Running speaker diarization...[/bold cyan]", spinner="dots"):
                d = Diarizer()
                diar_segments = d.diarize(audio_path)
                d.unload()
            console.print("[dim]Diarization complete[/dim]")

        with console.status(f"[bold cyan]Transcribing with whisper-{model}...[/bold cyan]", spinner="dots"):
            t = Transcriber(model=model)
            segments = t.transcribe(audio_path, language=language, diarization_segments=diar_segments)
            t.unload()

        out_path = audio_path.with_suffix(f'.{output}')
        if output == 'srt':
            from audiosmith.srt import write_srt
            from audiosmith.srt_formatter import SRTFormatter
            entries = SRTFormatter().format_segments(segments)
            write_srt(entries, out_path)
        elif output == 'txt':
            from audiosmith.download import segments_to_txt
            out_path.write_text(segments_to_txt(segments), encoding='utf-8')
        elif output == 'vtt':
            from audiosmith.download import segments_to_vtt
            out_path.write_text(segments_to_vtt(segments), encoding='utf-8')
        elif output == 'json':
            from audiosmith.download import segments_to_json
            out_path.write_text(segments_to_json(segments), encoding='utf-8')

        console.print(Panel(
            f"[bold]Segments:[/bold] {len(segments)}\n"
            f"[bold]Format:[/bold] {output.upper()}\n"
            f"[bold]Output:[/bold] {out_path}",
            title="[green]Transcription Complete[/green]", border_style="green",
        ))
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)


@click.command()
@click.argument('srt_file', type=click.Path(exists=True))
@click.option('--target-lang', '-t', required=True, help='Target language code.')
@click.option('--source-lang', '-s', default='en', help='Source language code.')
@click.option('--backend', '-b', default='argos', type=click.Choice(['argos', 'gemma']),
              help='Translation backend.')
def translate(srt_file, target_lang, source_lang, backend):
    """Translate an SRT subtitle file."""
    from audiosmith.srt import SRTEntry, parse_srt_file, write_srt
    from audiosmith.translate import translate as do_translate

    srt_path = Path(srt_file)
    try:
        entries = parse_srt_file(srt_path)
        with console.status(f"[bold cyan]Translating {len(entries)} entries ({source_lang} → {target_lang})...[/bold cyan]", spinner="dots"):
            translated = []
            for entry in entries:
                new_text = do_translate(entry.text, source_lang, target_lang, backend=backend)
                translated.append(SRTEntry(
                    index=entry.index, start_time=entry.start_time, end_time=entry.end_time, text=new_text,
                ))

        out_path = srt_path.with_name(f'{srt_path.stem}_translated_{target_lang}.srt')
        write_srt(translated, out_path)
        console.print(Panel(
            f"[bold]Entries:[/bold] {len(translated)}\n"
            f"[bold]Backend:[/bold] {backend}\n"
            f"[bold]Output:[/bold] {out_path}",
            title="[green]Translation Complete[/green]", border_style="green",
        ))
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)


@click.command('transcribe-url')
@click.argument('url')
@click.option('--output', '-o', default='srt', type=click.Choice(['srt', 'txt', 'vtt', 'json']),
              help='Output format.')
@click.option('--language', '-l', default=None, help='Audio language (auto-detect if omitted).')
def transcribe_url(url, output, language):
    """Download and transcribe audio from a URL (YouTube, etc.)."""
    from audiosmith.download import download_media, slugify
    from audiosmith.ffmpeg import extract_audio
    from audiosmith.transcribe import Transcriber

    try:
        with console.status("[bold cyan]Downloading media...[/bold cyan]", spinner="dots"):
            tmp_dir = Path('audiosmith_tmp')
            media_path, title = download_media(url, tmp_dir)
            slug = slugify(title)

        with console.status("[bold cyan]Extracting audio...[/bold cyan]", spinner="dots"):
            audio_path = tmp_dir / f'{slug}_16k.wav'
            extract_audio(media_path, audio_path)

        with console.status("[bold cyan]Transcribing...[/bold cyan]", spinner="dots"):
            t = Transcriber()
            segments = t.transcribe(audio_path, language=language)
            t.unload()

        out_path = Path(f'{slug}.{output}')
        if output == 'srt':
            from audiosmith.srt import write_srt
            from audiosmith.srt_formatter import SRTFormatter
            entries = SRTFormatter().format_segments(segments)
            write_srt(entries, out_path)
        elif output == 'txt':
            from audiosmith.download import segments_to_txt
            out_path.write_text(segments_to_txt(segments), encoding='utf-8')
        elif output == 'vtt':
            from audiosmith.download import segments_to_vtt
            out_path.write_text(segments_to_vtt(segments), encoding='utf-8')
        elif output == 'json':
            from audiosmith.download import segments_to_json
            out_path.write_text(segments_to_json(segments), encoding='utf-8')

        console.print(Panel(
            f"[bold]Title:[/bold] {title}\n"
            f"[bold]Segments:[/bold] {len(segments)}\n"
            f"[bold]Output:[/bold] {out_path}",
            title="[green]URL Transcription Complete[/green]", border_style="green",
        ))
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)
