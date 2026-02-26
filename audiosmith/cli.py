"""Rich CLI for AudioSmith — dub, transcribe, translate, batch, export, normalize, check, tts, extract-voices, info, voices."""

import sys
import shutil
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from audiosmith.log import setup_logging, get_logger
from audiosmith.exceptions import AudioSmithError

logger = get_logger(__name__)
console = Console()
VERSION = '0.5.0'


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable debug logging.')
@click.version_option(VERSION, prog_name='AudioSmith')
def cli(verbose):
    """AudioSmith — CLI audio/video processing toolkit."""
    setup_logging('DEBUG' if verbose else 'INFO')


# ── New commands: info, voices ────────────────────────────────────────


@cli.command()
def info():
    """Show system capabilities, engines, and available models."""
    # System info
    sys_table = Table(title="System", show_header=True, header_style="bold cyan")
    sys_table.add_column("Component", width=20)
    sys_table.add_column("Status", width=50)

    sys_table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            sys_table.add_row("CUDA", f"[green]✓[/green] {gpu} ({vram:.1f} GB)")
        else:
            sys_table.add_row("CUDA", "[red]✗[/red] Not available")
    except ImportError:
        sys_table.add_row("CUDA", "[red]✗[/red] PyTorch not installed")

    ffmpeg = shutil.which('ffmpeg')
    sys_table.add_row("FFmpeg", "[green]✓[/green] Available" if ffmpeg else "[red]✗[/red] Not found")

    disk = shutil.disk_usage('.')
    sys_table.add_row("Disk Free", f"{disk.free / (1024**3):.1f} GB")

    console.print(sys_table)
    console.print()

    # TTS Engines
    eng_table = Table(title="TTS Engines", show_header=True, header_style="bold green")
    eng_table.add_column("Engine", width=20)
    eng_table.add_column("Status", width=12, justify="center")
    eng_table.add_column("Features", width=40)

    engines = [
        ("Piper", "piper_tts", "Fast ONNX, Polish/English voices"),
        ("Chatterbox", "tts", "23 languages, zero-shot voice cloning"),
        ("Qwen3", "qwen3_tts", "Voice design, cloning, 9 premium voices, 10 languages"),
    ]
    for name, mod, features in engines:
        try:
            __import__(f"audiosmith.{mod}")
            eng_table.add_row(name, "[green]✓[/green]", features)
        except Exception:
            eng_table.add_row(name, "[red]✗[/red]", features)

    console.print(eng_table)
    console.print()

    # Processing capabilities
    cap_table = Table(title="Capabilities", show_header=True, header_style="bold magenta")
    cap_table.add_column("Feature", width=25)
    cap_table.add_column("Module", width=25)
    cap_table.add_column("Status", width=12, justify="center")

    capabilities = [
        ("Transcription", "faster_whisper", "Whisper large-v3"),
        ("Translation", "argostranslate", "Argos offline"),
        ("Diarization", "pyannote.audio", "Speaker separation"),
        ("Vocal Isolation", "demucs", "Demucs v4"),
        ("Audio Normalize", "audiosmith.audio_normalizer", "LUFS targeting"),
        ("Emotion Detection", "audiosmith.emotion", "Rule-based + ML"),
        ("Document Export", "fpdf2", "PDF/DOCX/TXT"),
        ("Batch Processing", "audiosmith.batch_processor", "Multi-file"),
        ("Voice Extraction", "audiosmith.voice_extractor", "Catalog + profiles"),
    ]
    for feat, mod, desc in capabilities:
        try:
            __import__(mod)
            cap_table.add_row(feat, desc, "[green]✓[/green]")
        except Exception:
            cap_table.add_row(feat, desc, "[yellow]○[/yellow]")

    console.print(cap_table)


@cli.command()
@click.option('--engine', '-e', type=click.Choice(['piper', 'chatterbox', 'qwen3', 'all']),
              default='all', help='Show voices for specific engine.')
def voices(engine):
    """List available voices for each TTS engine."""
    if engine in ('piper', 'all'):
        t = Table(title="[cyan]Piper TTS Voices[/cyan]", show_header=True, header_style="bold")
        t.add_column("Voice ID", width=28)
        t.add_column("Language", width=10, justify="center")
        for vid, lang in [
            ("en_US-lessac-medium", "en"), ("en_US-ryan-medium", "en"),
            ("en_GB-cori-medium", "en"), ("pl_PL-darkman-medium", "pl"),
            ("de_DE-thorsten-medium", "de"),
        ]:
            t.add_row(vid, lang)
        console.print(t)
        console.print()

    if engine in ('chatterbox', 'all'):
        t = Table(title="[magenta]Chatterbox TTS[/magenta]", show_header=True, header_style="bold")
        t.add_column("Feature", width=20)
        t.add_column("Details", width=50)
        try:
            from audiosmith.tts import LANGUAGE_MAP
            langs = ", ".join(sorted(LANGUAGE_MAP.keys()))
        except ImportError:
            langs = "en, pl, de, fr, es, it, pt, ru, ja, ko, zh, ..."
        t.add_row("Languages", langs)
        t.add_row("Voice Cloning", "Zero-shot from single audio sample")
        t.add_row("Parameters", "exaggeration (0.0-1.0), cfg_weight (0.0-1.0)")
        console.print(t)
        console.print()

    if engine in ('qwen3', 'all'):
        t = Table(title="[green]Qwen3 TTS Premium Voices[/green]", show_header=True, header_style="bold")
        t.add_column("Name", width=12)
        t.add_column("Description", width=45)
        t.add_column("Language", width=10, justify="center")
        t.add_column("Gender", width=8, justify="center")
        try:
            from audiosmith.qwen3_tts import PREMIUM_VOICES
            for name, info in PREMIUM_VOICES.items():
                t.add_row(name, info["description"], info["language"], info["gender"])
        except ImportError:
            t.add_row("—", "qwen-tts not installed", "—", "—")
        console.print(t)

        modes = Table(title="[green]Qwen3 TTS Modes[/green]", show_header=True, header_style="bold")
        modes.add_column("Mode", width=18)
        modes.add_column("Model", width=20)
        modes.add_column("Description", width=35)
        modes.add_row("custom_voice", "CustomVoice 1.7B", "9 premium named speakers")
        modes.add_row("voice_design", "VoiceDesign 1.7B", "Text-described voice creation")
        modes.add_row("voice_clone", "Base 1.7B", "Clone from reference audio")
        console.print(modes)
        console.print()


# ── Core commands ─────────────────────────────────────────────────────


@cli.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('--target-lang', '-t', required=True, help='Target language code (e.g. pl, es, de).')
@click.option('--source-lang', '-s', default='auto', help='Source language (auto-detect if omitted).')
@click.option('--output-dir', '-o', default=None, help='Output directory.')
@click.option('--resume', is_flag=True, help='Resume from checkpoint.')
@click.option('--diarize', is_flag=True, help='Enable speaker diarization.')
@click.option('--emotion', is_flag=True, help='Enable emotion detection for TTS.')
@click.option('--isolate-vocals', is_flag=True, help='Isolate vocals before transcription.')
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
        with console.status("[bold cyan]Running dubbing pipeline...[/bold cyan]", spinner="dots"):
            pipeline = DubbingPipeline(config)
            result = pipeline.run(video_path)

        console.print(Panel(
            f"[bold]Segments dubbed:[/bold] {result.segments_dubbed}\n"
            f"[bold]Total time:[/bold] {result.total_time:.1f}s\n"
            f"[bold]Output:[/bold] {result.output_video_path or output_dir}",
            title="[green]Dubbing Complete[/green]", border_style="green",
        ))
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)


@cli.command()
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


@cli.command()
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

        console.print(Panel(
            f"[bold]Segments:[/bold] {len(segments)}\n"
            f"[bold]Format:[/bold] {fmt.upper()}\n"
            f"[bold]Output:[/bold] {out_path}",
            title="[green]Export Complete[/green]", border_style="green",
        ))
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
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

        with console.status("[bold cyan]Normalizing...[/bold cyan]", spinner="dots"):
            normalizer.normalize(audio_path, out_path)
        stats_out = normalizer.analyze(out_path)

        t = Table(title="Normalization Results", show_header=True, header_style="bold")
        t.add_column("", width=10)
        t.add_column("LUFS", width=12, justify="right")
        t.add_column("Peak dB", width=12, justify="right")
        t.add_row("Before", f"{stats['lufs']:.1f}", f"{stats['peak_db']:.1f}")
        t.add_row("After", f"[green]{stats_out['lufs']:.1f}[/green]", f"[green]{stats_out['peak_db']:.1f}[/green]")
        t.add_row("Target", f"{target_lufs:.1f}", f"{max_peak:.1f}")
        console.print(t)
        console.print(f"[dim]Saved: {out_path}[/dim]")
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)


@cli.command()
def check():
    """Run system pre-flight checks (FFmpeg, CUDA, Whisper, disk space)."""
    from audiosmith.system_check import SystemChecker

    checker = SystemChecker()
    results = checker.run_all_checks()

    t = Table(title="System Pre-Flight Checks", show_header=True, header_style="bold magenta")
    t.add_column("Check", width=20)
    t.add_column("Status", width=10, justify="center")

    check_names = {'ffmpeg': 'FFmpeg', 'torch': 'PyTorch', 'cuda': 'CUDA',
                   'faster_whisper': 'Faster Whisper', 'disk_space': 'Disk Space'}
    for key, name in check_names.items():
        if key in results:
            passed = results[key]
            t.add_row(name, "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]")

    console.print(Panel(t, border_style="blue"))

    if not all(results.values()):
        failed = [check_names.get(k, k) for k, v in results.items() if not v]
        console.print(f"[bold yellow]Warning:[/bold yellow] {', '.join(failed)} not available.")


# ── TTS command (enhanced with full Qwen3 options) ───────────────────


@cli.command()
@click.argument('text')
@click.option('--engine', '-e', type=click.Choice(['piper', 'chatterbox', 'qwen3']),
              default='piper', help='TTS engine.')
@click.option('--voice', default=None, help='Voice name (engine-specific).')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output audio file.')
@click.option('--language', '-l', default='en', help='Language code.')
@click.option('--model-type', type=click.Choice(['base', 'voice_design', 'custom_voice']),
              default=None, help='Qwen3 model type (auto-detected if omitted).')
@click.option('--instruct', default=None,
              help='Voice description for Qwen3 voice design (e.g. "A deep male narrator").')
@click.option('--ref-audio', default=None, type=click.Path(exists=True),
              help='Reference audio for voice cloning (Qwen3 base or Chatterbox).')
@click.option('--ref-text', default=None,
              help='Transcript of reference audio (improves Qwen3 clone quality).')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode — prompts for missing options.')
def tts(text, engine, voice, output, language, model_type, instruct, ref_audio, ref_text, interactive):
    """Synthesize speech from text with Piper, Chatterbox, or Qwen3."""
    import numpy as np

    output_path = Path(output)

    if interactive:
        engine = Prompt.ask("Choose engine", choices=["piper", "chatterbox", "qwen3"], default=engine)
        if engine == 'qwen3':
            if not voice and not instruct and not ref_audio:
                mode = Prompt.ask("Qwen3 mode", choices=["premium", "design", "clone"], default="premium")
                if mode == "premium":
                    model_type = "custom_voice"
                    voice = Prompt.ask("Voice name", default="Ryan")
                elif mode == "design":
                    model_type = "voice_design"
                    instruct = Prompt.ask("Describe the voice", default="A calm male narrator")
                elif mode == "clone":
                    model_type = "base"
                    ref_audio = Prompt.ask("Path to reference audio")
                    ref_text = Prompt.ask("Reference audio transcript (optional, Enter to skip)", default="") or None

    try:
        if engine == 'piper':
            from audiosmith.piper_tts import PiperTTS
            with console.status("[bold cyan]Loading Piper...[/bold cyan]", spinner="dots"):
                p = PiperTTS(voice=voice or 'en_US-lessac-medium')
            audio = p.synthesize(text)
            import soundfile as sf
            sf.write(str(output_path), audio, p.sample_rate)
            sample_rate = p.sample_rate

        elif engine == 'chatterbox':
            from audiosmith.tts import ChatterboxTTS
            with console.status("[bold cyan]Loading Chatterbox...[/bold cyan]", spinner="dots"):
                cb = ChatterboxTTS()
                cb.load_model()
            audio = cb.synthesize(text, language=language, audio_prompt_path=ref_audio)
            import soundfile as sf
            sf.write(str(output_path), audio, cb.sample_rate)
            sample_rate = cb.sample_rate
            cb.cleanup()

        elif engine == 'qwen3':
            from audiosmith.qwen3_tts import Qwen3TTS, _normalize_language

            # Auto-detect model type from options
            if model_type is None:
                if instruct:
                    model_type = 'voice_design'
                elif ref_audio:
                    model_type = 'base'
                else:
                    model_type = 'custom_voice'

            normalized_lang = _normalize_language(language)

            with console.status(f"[bold cyan]Loading Qwen3 ({model_type})...[/bold cyan]", spinner="dots"):
                q = Qwen3TTS()
                q.load_model(model_type)

            if model_type == 'voice_design':
                audio, sr = q._design_model.generate_voice_design(
                    text, instruct or "A natural narrator voice",
                    language=normalized_lang,
                )
                if isinstance(audio, list):
                    audio = np.concatenate(audio)
            elif model_type == 'base':
                audio, sr = q._base_model.generate_voice_clone(
                    text, language=normalized_lang,
                    ref_audio=ref_audio, ref_text=ref_text,
                    x_vector_only_mode=(ref_text is None),
                )
                if isinstance(audio, list):
                    audio = np.concatenate(audio)
            else:  # custom_voice
                audio, sr = q.synthesize(text, voice=voice or 'Ryan', language=language)

            q.save_audio(audio, output_path, sr)
            sample_rate = sr
            q.cleanup()

        # Result panel
        duration = len(audio) / sample_rate
        file_size = output_path.stat().st_size
        console.print(Panel(
            f"[bold]Engine:[/bold] {engine}\n"
            f"[bold]Voice:[/bold] {voice or 'default'}\n"
            f"[bold]Duration:[/bold] {duration:.2f}s\n"
            f"[bold]Sample Rate:[/bold] {sample_rate} Hz\n"
            f"[bold]File Size:[/bold] {file_size / 1024:.1f} KB\n"
            f"[bold]Output:[/bold] {output_path}",
            title="[green]Synthesis Complete[/green]", border_style="green",
        ))
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)


# ── Voice extraction ──────────────────────────────────────────────────


@cli.command('extract-voices')
@click.argument('audio', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default=None, type=click.Path(), help='Output directory.')
@click.option('--num-samples', '-n', default=5, type=int, help='Number of samples (even mode).')
@click.option('--sample-duration', default=5.0, type=float, help='Sample duration in seconds.')
@click.option('--sample-rate', default=24000, type=int, help='Output sample rate in Hz.')
@click.option('--diarize', is_flag=True, help='Use speaker diarization.')
@click.option('--num-speakers', default=None, type=int, help='Expected speakers (with --diarize).')
@click.option('--catalog', '-c', default=None, type=click.Path(), help='Save voice catalog JSON.')
def extract_voices(audio, output_dir, num_samples, sample_duration, sample_rate, diarize, num_speakers, catalog):
    """Extract voice samples from audio for TTS voice cloning."""
    from audiosmith.voice_extractor import VoiceExtractor, create_voice_profiles

    audio_path = Path(audio)
    out_dir = Path(output_dir) if output_dir else audio_path.parent / f'{audio_path.stem}_voices'

    try:
        extractor = VoiceExtractor(
            out_dir, sample_duration=sample_duration, sample_rate=sample_rate,
        )

        with console.status("[bold cyan]Extracting voice samples...[/bold cyan]", spinner="dots"):
            if diarize:
                voice_catalog = extractor.extract_with_diarization(audio_path, num_speakers=num_speakers)
            else:
                voice_catalog = extractor.extract_evenly(audio_path, num_samples=num_samples)

        speakers = voice_catalog.get_speakers()

        t = Table(title=f"Extracted {len(voice_catalog.samples)} samples", show_header=True, header_style="bold magenta")
        t.add_column("Speaker", width=15)
        t.add_column("File", width=25)
        t.add_column("Duration", width=10, justify="right")
        t.add_column("Volume", width=10, justify="right")

        for sid in speakers:
            best = voice_catalog.get_best_sample(sid)
            if best:
                t.add_row(sid, best.sample_path.name, f"{best.duration:.1f}s", f"{best.mean_volume_db:.1f} dB")

        console.print(t)

        if catalog:
            catalog_path = Path(catalog)
            voice_catalog.save(catalog_path)
            console.print(f"[dim]Catalog saved: {catalog_path}[/dim]")

        profiles = create_voice_profiles(voice_catalog)
        if profiles:
            console.print(f"[green]Voice profiles ready for TTS cloning ({len(profiles)} voices)[/green]")

    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)
