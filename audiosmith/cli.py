"""Click CLI for AudioSmith — dub, transcribe, translate, transcribe-url."""

import sys
from pathlib import Path

import click

from audiosmith.log import setup_logging, get_logger
from audiosmith.exceptions import AudioSmithError

logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable debug logging.')
def cli(verbose):
    """AudioSmith — CLI audio/video processing toolkit."""
    setup_logging('DEBUG' if verbose else 'INFO')


@cli.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('--target-lang', '-t', required=True, help='Target language code (e.g. pl, es, de).')
@click.option('--source-lang', '-s', default='auto', help='Source language (auto-detect if omitted).')
@click.option('--output-dir', '-o', default=None, help='Output directory.')
@click.option('--resume', is_flag=True, help='Resume from checkpoint.')
def dub(video, target_lang, source_lang, output_dir, resume):
    """Dub a video into another language."""
    from audiosmith.models import DubbingConfig
    from audiosmith.pipeline import DubbingPipeline

    video_path = Path(video)
    if output_dir is None:
        output_dir = str(video_path.parent / f'{video_path.stem}_dubbed')

    try:
        config = DubbingConfig(
            video_path=video_path,
            output_dir=Path(output_dir),
            source_language=source_lang,
            target_language=target_lang,
            resume=resume,
        )
        pipeline = DubbingPipeline(config)
        result = pipeline.run(video_path)

        click.echo(f"Done! {result.segments_dubbed} segments dubbed in {result.total_time:.1f}s")
        if result.output_video_path:
            click.echo(f"Output: {result.output_video_path}")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('audio', type=click.Path(exists=True))
@click.option('--output', '-o', default='srt', type=click.Choice(['srt', 'txt', 'vtt', 'json']),
              help='Output format.')
@click.option('--language', '-l', default=None, help='Audio language (auto-detect if omitted).')
@click.option('--model', '-m', default='large-v3', help='Whisper model size.')
def transcribe(audio, output, language, model):
    """Transcribe an audio or video file."""
    from audiosmith.transcribe import Transcriber

    audio_path = Path(audio)
    try:
        t = Transcriber(model=model)
        segments = t.transcribe(audio_path, language=language)
        t.unload()

        out_path = audio_path.with_suffix(f'.{output}')
        if output == 'srt':
            from audiosmith.srt import SRTEntry, write_srt, seconds_to_timestamp
            entries = [
                SRTEntry(index=i + 1, start_time=seconds_to_timestamp(s['start']),
                         end_time=seconds_to_timestamp(s['end']), text=s['text'])
                for i, s in enumerate(segments)
            ]
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

        click.echo(f"Transcribed {len(segments)} segments -> {out_path}")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('srt_file', type=click.Path(exists=True))
@click.option('--target-lang', '-t', required=True, help='Target language code.')
@click.option('--source-lang', '-s', default='en', help='Source language code.')
@click.option('--backend', '-b', default='argos', type=click.Choice(['argos', 'gemma']),
              help='Translation backend.')
def translate(srt_file, target_lang, source_lang, backend):
    """Translate an SRT subtitle file."""
    from audiosmith.srt import parse_srt_file, SRTEntry, write_srt
    from audiosmith.translate import translate as do_translate

    srt_path = Path(srt_file)
    try:
        entries = parse_srt_file(srt_path)
        translated = []
        for entry in entries:
            new_text = do_translate(entry.text, source_lang, target_lang, backend=backend)
            translated.append(SRTEntry(
                index=entry.index, start_time=entry.start_time, end_time=entry.end_time, text=new_text,
            ))

        out_path = srt_path.with_name(f'{srt_path.stem}_translated_{target_lang}.srt')
        write_srt(translated, out_path)
        click.echo(f"Translated {len(translated)} entries -> {out_path}")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)


@cli.command('transcribe-url')
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
        tmp_dir = Path('audiosmith_tmp')
        media_path, title = download_media(url, tmp_dir)
        slug = slugify(title)

        audio_path = tmp_dir / f'{slug}_16k.wav'
        extract_audio(media_path, audio_path)

        t = Transcriber()
        segments = t.transcribe(audio_path, language=language)
        t.unload()

        out_path = Path(f'{slug}.{output}')
        if output == 'srt':
            from audiosmith.srt import SRTEntry, write_srt, seconds_to_timestamp
            entries = [
                SRTEntry(index=i + 1, start_time=seconds_to_timestamp(s['start']),
                         end_time=seconds_to_timestamp(s['end']), text=s['text'])
                for i, s in enumerate(segments)
            ]
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

        click.echo(f"Transcribed '{title}' -> {out_path} ({len(segments)} segments)")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)
