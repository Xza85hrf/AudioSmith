"""Rich CLI for AudioSmith — dub, transcribe, translate, batch, export, normalize, check, tts, extract-voices, info, voices."""

import shutil
import sys
from pathlib import Path

# Ensure Rich's Unicode glyphs can be emitted on Windows even when stdout is
# redirected or running under a cp1252 code page. Must happen before Rich
# captures the stream encoding.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audiosmith.exceptions import AudioSmithError
from audiosmith.log import get_logger, setup_logging

try:
    from aiml_training.exceptions import AimlTrainingError
except ImportError:
    AimlTrainingError = AudioSmithError  # fallback when training package absent

logger = get_logger(__name__)
console = Console()
VERSION = '0.5.0'


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable debug logging.')
@click.version_option(VERSION, prog_name='AudioSmith')
def cli(verbose):
    """AudioSmith — CLI audio/video processing toolkit."""
    setup_logging('DEBUG' if verbose else 'INFO')


# ── Small/simple commands (kept in main cli.py) ────────────────────────────


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
        ("ElevenLabs", "elevenlabs_tts", "[CLOUD] 70+ languages, voice cloning, preset voices"),
        ("IndexTTS-2", "indextts_tts", "Emotion disentanglement, EN/ZH, voice cloning"),
        ("CosyVoice2", "cosyvoice_tts", "MOS 5.53, 9 languages, zero-shot cloning, instruct"),
        ("Orpheus", "orpheus_tts", "13 languages, 8 voices, emotion tags, expressive"),
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
@click.option('--engine', '-e', type=click.Choice(['piper', 'chatterbox', 'qwen3', 'fish', 'elevenlabs', 'indextts', 'cosyvoice', 'orpheus', 'f5', 'all']),
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

    if engine in ('elevenlabs', 'all'):
        t = Table(title="[yellow]ElevenLabs TTS Voices[/yellow]", show_header=True, header_style="bold")
        t.add_column("Name", width=14)
        t.add_column("Voice ID", width=28)
        try:
            from audiosmith.elevenlabs_tts import ELEVENLABS_MODELS, VOICE_MAP
            for name, vid in VOICE_MAP.items():
                t.add_row(name, vid)
        except ImportError:
            t.add_row("—", "elevenlabs not installed")
        console.print(t)

        models_t = Table(title="[yellow]ElevenLabs Models[/yellow]", show_header=True, header_style="bold")
        models_t.add_column("Model ID", width=28)
        models_t.add_column("Description", width=42)
        try:
            for mid, desc in ELEVENLABS_MODELS.items():
                models_t.add_row(mid, desc)
        except NameError:
            pass
        console.print(models_t)
        console.print()

    if engine in ('indextts', 'all'):
        t = Table(title="[red]IndexTTS-2[/red]", show_header=True, header_style="bold")
        t.add_column("Language", width=12)
        t.add_column("Emoji", width=6, justify="center")
        for lang, emoji in [("English", "🇬🇧"), ("Chinese", "🇨🇳")]:
            t.add_row(lang, emoji)
        console.print(t)
        console.print()

    if engine in ('cosyvoice', 'all'):
        t = Table(title="[blue]CosyVoice2[/blue]", show_header=True, header_style="bold")
        t.add_column("Feature", width=20)
        t.add_column("Details", width=50)
        t.add_row("Languages", "9: EN, ZH, JA, KO, ES, FR, DE, IT, PT")
        t.add_row("Zero-shot", "Clone from reference (ref audio + text)")
        t.add_row("Instruction", "eg. 'Speak happily' or 'Whisper'")
        console.print(t)
        console.print()

    if engine in ('orpheus', 'all'):
        t = Table(title="[purple]Orpheus[/purple]", show_header=True, header_style="bold")
        t.add_column("Preset Voice", width=14)
        t.add_column("Description", width=50)
        for voice in ["Tara", "Leah", "Jess", "Leo", "Dan", "Mia", "Zac", "Zoe"]:
            t.add_row(voice, f"{voice} voice preset")
        console.print(t)
        console.print()

    if engine in ('fish', 'all'):
        t = Table(title="[cyan]Fish Speech[/cyan]", show_header=True, header_style="bold")
        t.add_column("Feature", width=20)
        t.add_column("Details", width=50)
        t.add_row("Languages", "Multi-lingual")
        t.add_row("Voice Cloning", "Zero-shot from reference audio")
        console.print(t)
        console.print()

    if engine in ('f5', 'all'):
        t = Table(title="[green]F5-TTS[/green]", show_header=True, header_style="bold")
        t.add_column("Feature", width=20)
        t.add_column("Details", width=50)
        t.add_row("Generation", "Non-autoregressive, fast")
        t.add_row("Voice Cloning", "Clone from reference audio")
        console.print(t)


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
    from audiosmith.document_formatter import (DocumentFormatter,
                                               FormatterOptions)
    from audiosmith.models import DubbingSegment
    from audiosmith.srt import parse_srt_file

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
        formatter = DocumentFormatter(options)
        formatter.format(segments, out_path)

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
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        extractor = VoiceExtractor(sample_rate=sample_rate, output_dir=out_dir)

        if diarize:
            with console.status("[bold cyan]Extracting with diarization...[/bold cyan]", spinner="dots"):
                voice_catalog = extractor.extract_with_diarization(
                    audio_path, num_speakers=num_speakers, sample_duration=sample_duration
                )
        else:
            with console.status(f"[bold cyan]Extracting {num_samples} even samples...[/bold cyan]", spinner="dots"):
                voice_catalog = extractor.extract_evenly(
                    audio_path, num_samples=num_samples, sample_duration=sample_duration
                )

        # Create voice profiles
        profiles = create_voice_profiles(voice_catalog.get_speakers(), voice_catalog)

        # Save catalog if requested
        if catalog:
            import json
            cat_path = Path(catalog)
            cat_path.write_text(json.dumps(voice_catalog.to_dict()), encoding='utf-8')

        console.print(Panel(
            f"[bold]Extracted:[/bold] {len(voice_catalog.samples)} samples\n"
            f"[bold]Speakers:[/bold] {len(voice_catalog.get_speakers())}\n"
            f"[bold]Output:[/bold] {out_dir}",
            title="[green]Voice Extraction Complete[/green]", border_style="green",
        ))
    except AudioSmithError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        sys.exit(1)


# ── Command registration (must be after cli group is defined) ─────────────
import audiosmith.commands  # noqa: E402,F401 — triggers add_command() calls
