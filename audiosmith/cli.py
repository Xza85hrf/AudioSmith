"""Click CLI for AudioSmith — dub, transcribe, translate, batch, export, normalize, check, tts, extract-voices."""

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
@click.option('--diarize', is_flag=True, help='Enable speaker diarization (requires pyannote-audio).')
@click.option('--emotion', is_flag=True, help='Enable emotion detection for TTS enhancement.')
@click.option('--isolate-vocals', is_flag=True, help='Isolate vocals before transcription (requires demucs).')
def dub(video, target_lang, source_lang, output_dir, resume, diarize, emotion, isolate_vocals):
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
            isolate_vocals=isolate_vocals,
            diarize=diarize,
            detect_emotion=emotion,
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
@click.option('--diarize', is_flag=True, help='Enable speaker diarization (requires pyannote-audio).')
@click.option('--isolate-vocals', is_flag=True, help='Isolate vocals before transcription (requires demucs).')
def transcribe(audio, output, language, model, diarize, isolate_vocals):
    """Transcribe an audio or video file."""
    from audiosmith.transcribe import Transcriber

    audio_path = Path(audio)
    try:
        if isolate_vocals:
            from audiosmith.vocal_isolator import VocalIsolator
            vi = VocalIsolator()
            paths = vi.isolate(audio_path)
            vi.unload()
            audio_path = paths['vocals_path']

        diar_segments = None
        if diarize:
            from audiosmith.diarizer import Diarizer
            d = Diarizer()
            diar_segments = d.diarize(audio_path)
            d.unload()

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

        click.echo(f"Transcribed '{title}' -> {out_path} ({len(segments)} segments)")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--target-lang', '-t', required=True, help='Target language code.')
@click.option('--source-lang', '-s', default='auto', help='Source language (auto-detect if omitted).')
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
        processor = BatchProcessor()
        results = processor.process(file_paths, config, continue_on_error=continue_on_error)
        summary = BatchProcessor.get_summary(results)

        click.echo(f"Batch complete: {summary['succeeded']}/{summary['total']} succeeded")
        if summary['failed']:
            click.echo(f"Failed ({summary['failed']}): {', '.join(summary['failed_files'])}")
        click.echo(f"Total time: {summary['total_duration_seconds']:.1f}s")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('srt_file', type=click.Path(exists=True))
@click.option('--format', '-f', 'fmt', type=click.Choice(['txt', 'pdf', 'docx']), default='txt',
              help='Output format.')
@click.option('--output', '-o', default=None, type=click.Path(), help='Output file path.')
@click.option('--title', default=None, help='Document title.')
@click.option('--timestamps/--no-timestamps', default=True, help='Include timestamps.')
@click.option('--speakers', is_flag=True, help='Include speaker labels.')
def export(srt_file, fmt, output, title, timestamps, speakers):
    """Export an SRT file to TXT, PDF, or DOCX."""
    from audiosmith.srt import parse_srt_file
    from audiosmith.document_formatter import DocumentFormatter, FormatterOptions
    from audiosmith.models import DubbingSegment

    srt_path = Path(srt_file)
    try:
        entries = parse_srt_file(srt_path)
        segments = [
            DubbingSegment(
                index=e.index, start_time=e.start_time,
                end_time=e.end_time, original_text=e.text,
            )
            for e in entries
        ]

        out_path = Path(output) if output else srt_path.with_suffix(f'.{fmt}')
        options = FormatterOptions(
            title=title or srt_path.stem,
            include_timestamps=timestamps,
            include_speaker_labels=speakers,
        )

        formatter = DocumentFormatter()
        if fmt == 'txt':
            formatter.to_txt(segments, out_path, options)
        elif fmt == 'pdf':
            formatter.to_pdf(segments, out_path, options)
        else:
            formatter.to_docx(segments, out_path, options)

        click.echo(f"Exported {len(segments)} segments -> {out_path}")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('audio', type=click.Path(exists=True))
@click.option('--output', '-o', default=None, type=click.Path(), help='Output file path.')
@click.option('--target-lufs', default=-23.0, type=float, help='Target LUFS level (default: -23.0).')
@click.option('--max-peak', default=-1.0, type=float, help='Maximum peak in dB (default: -1.0).')
def normalize(audio, output, target_lufs, max_peak):
    """Normalize audio loudness to a target LUFS level."""
    from audiosmith.audio_normalizer import AudioNormalizer

    audio_path = Path(audio)
    out_path = Path(output) if output else audio_path.with_name(
        f'{audio_path.stem}_normalized{audio_path.suffix}'
    )

    try:
        normalizer = AudioNormalizer(target_lufs=target_lufs, max_peak_db=max_peak)
        stats = normalizer.analyze(audio_path)
        click.echo(f"Input:  {stats['lufs']:.1f} LUFS, peak {stats['peak_db']:.1f} dB")

        normalizer.normalize(audio_path, out_path)

        stats_out = normalizer.analyze(out_path)
        click.echo(f"Output: {stats_out['lufs']:.1f} LUFS, peak {stats_out['peak_db']:.1f} dB")
        click.echo(f"Saved: {out_path}")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)


@cli.command()
def check():
    """Run system pre-flight checks (FFmpeg, CUDA, Whisper, disk space)."""
    from audiosmith.system_check import SystemChecker

    checker = SystemChecker()
    results = checker.run_all_checks()
    click.echo(checker.get_summary(results))

    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        click.echo(f"\nWarning: {', '.join(failed)} not available.", err=True)


@cli.command()
@click.argument('text')
@click.option('--engine', '-e', type=click.Choice(['piper', 'chatterbox', 'qwen3']),
              default='piper', help='TTS engine.')
@click.option('--voice', default=None, help='Voice name (engine-specific).')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output audio file.')
@click.option('--language', '-l', default='en', help='Language code.')
def tts(text, engine, voice, output, language):
    """Synthesize speech from text."""
    output_path = Path(output)

    try:
        if engine == 'piper':
            from audiosmith.piper_tts import PiperTTS
            p = PiperTTS(voice=voice or 'en_US-lessac-medium')
            audio = p.synthesize(text)
            import soundfile as sf
            sf.write(str(output_path), audio, p.sample_rate)

        elif engine == 'chatterbox':
            from audiosmith.tts import ChatterboxTTS
            cb = ChatterboxTTS()
            cb.load_model()
            audio = cb.synthesize(text, language=language)
            import soundfile as sf
            sf.write(str(output_path), audio, cb.sample_rate)
            cb.cleanup()

        elif engine == 'qwen3':
            from audiosmith.qwen3_tts import Qwen3TTS
            q = Qwen3TTS()
            q.load_model()
            audio, sr = q.synthesize(text, voice=voice or 'Ryan')
            q.save_audio(audio, output_path, sr)
            q.cleanup()

        click.echo(f"TTS ({engine}) -> {output_path}")
    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)


@cli.command('extract-voices')
@click.argument('audio', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default=None, type=click.Path(), help='Output directory for voice samples.')
@click.option('--num-samples', '-n', default=5, type=int, help='Number of samples to extract (even mode).')
@click.option('--sample-duration', default=5.0, type=float, help='Duration of each sample in seconds.')
@click.option('--sample-rate', default=24000, type=int, help='Output sample rate in Hz.')
@click.option('--diarize', is_flag=True, help='Use speaker diarization to identify voices (requires pyannote-audio).')
@click.option('--num-speakers', default=None, type=int, help='Expected number of speakers (with --diarize).')
@click.option('--catalog', '-c', default=None, type=click.Path(), help='Save voice catalog JSON to this path.')
def extract_voices(audio, output_dir, num_samples, sample_duration, sample_rate, diarize, num_speakers, catalog):
    """Extract voice samples from audio for TTS voice cloning."""
    from audiosmith.voice_extractor import VoiceExtractor, create_voice_profiles

    audio_path = Path(audio)
    out_dir = Path(output_dir) if output_dir else audio_path.parent / f'{audio_path.stem}_voices'

    try:
        extractor = VoiceExtractor(
            out_dir, sample_duration=sample_duration, sample_rate=sample_rate,
        )

        if diarize:
            voice_catalog = extractor.extract_with_diarization(audio_path, num_speakers=num_speakers)
        else:
            voice_catalog = extractor.extract_evenly(audio_path, num_samples=num_samples)

        speakers = voice_catalog.get_speakers()
        click.echo(f"Extracted {len(voice_catalog.samples)} samples from {len(speakers)} speaker(s)")

        for sid in speakers:
            best = voice_catalog.get_best_sample(sid)
            if best:
                click.echo(f"  {sid}: {best.sample_path.name} ({best.duration:.1f}s, {best.mean_volume_db:.1f} dB)")

        if catalog:
            catalog_path = Path(catalog)
            voice_catalog.save(catalog_path)
            click.echo(f"Catalog saved: {catalog_path}")

        profiles = create_voice_profiles(voice_catalog)
        if profiles:
            click.echo(f"\nVoice profiles ready for TTS cloning ({len(profiles)} voices)")

    except AudioSmithError as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)
