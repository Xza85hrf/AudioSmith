"""Dub command — video dubbing into another language."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from audiosmith.exceptions import AudioSmithError

console = Console()


# Define the command without decorating to cli — will be registered in __init__
@click.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('--target-lang', '-t', required=True, help='Target language code (e.g. pl, es, de).')
@click.option('--source-lang', '-s', default='auto', help='Source language (auto-detect if omitted).')
@click.option('--output-dir', '-o', default=None, help='Output directory.')
@click.option('--resume', is_flag=True, help='Resume from checkpoint.')
@click.option('--diarize', is_flag=True, help='Enable speaker diarization.')
@click.option('--emotion', is_flag=True, help='Enable emotion detection for TTS.')
@click.option('--isolate-vocals', is_flag=True, help='Isolate vocals before transcription.')
@click.option('--engine', type=click.Choice(['auto', 'chatterbox', 'qwen3', 'piper', 'fish', 'elevenlabs', 'indextts', 'cosyvoice', 'orpheus', 'f5']), default='auto',
              help='TTS engine (auto picks best for target language).')
@click.option('--audio-prompt', default=None, type=click.Path(exists=True),
              help='Voice sample for cloning (WAV file).')
@click.option('--max-speedup', default=None, type=float,
              help='Max TTS speedup factor (default: 2.0). Lower = less distortion.')
@click.option('--elevenlabs-model', default=None,
              type=click.Choice(['eleven_v3', 'eleven_multilingual_v2', 'eleven_flash_v2_5', 'eleven_turbo_v2_5']),
              help='ElevenLabs model variant (default: eleven_v3).')
@click.option('--elevenlabs-voice', default=None,
              help='ElevenLabs voice name or UUID.')
@click.option('--indextts-model', default=None, type=click.Choice(['base', 'design']),
              help='IndexTTS-2 model variant (default: base).')
@click.option('--indextts-emo-alpha', default=None, type=float,
              help='IndexTTS-2 emotion alpha [0=identity, 1=emotion] (default: 0.5).')
@click.option('--indextts-emotion-audio', default=None, type=click.Path(exists=True),
              help='IndexTTS-2 emotion reference audio (separate from voice).')
@click.option('--cosyvoice-model-dir', default=None, type=click.Path(exists=True),
              help='CosyVoice2 model directory (or set COSYVOICE_MODEL_DIR).')
@click.option('--cosyvoice-instruct', default=None,
              help='CosyVoice2 instruction text (e.g. "Speak happily").')
@click.option('--orpheus-voice', default=None,
              type=click.Choice(['tara', 'leah', 'jess', 'leo', 'dan', 'mia', 'zac', 'zoe']),
              help='Orpheus preset voice (default: tara).')
@click.option('--orpheus-temperature', default=None, type=float,
              help='Orpheus generation temperature (default: 0.7).')
@click.option('--post-process/--no-post-process', 'post_process_tts', default=True,
              help='Post-process local TTS for naturalness (silence, dynamics, warmth).')
@click.option('--post-process-intensity', default=0.7, type=float,
              help='Post-processing aggressiveness [0=off, 2=aggressive] (default: 0.7).')
def dub(video, target_lang, source_lang, output_dir, resume, diarize, emotion, isolate_vocals, engine, audio_prompt, max_speedup, elevenlabs_model, elevenlabs_voice, indextts_model, indextts_emo_alpha, indextts_emotion_audio, cosyvoice_model_dir, cosyvoice_instruct, orpheus_voice, orpheus_temperature, post_process_tts, post_process_intensity):
    """Dub a video into another language."""
    from audiosmith.models import DubbingConfig
    from audiosmith.pipeline import DubbingPipeline

    video_path = Path(video)
    if output_dir is None:
        output_dir = str(video_path.parent / f'{video_path.stem}_dubbed')

    try:
        kwargs = dict(
            video_path=video_path,
            output_dir=Path(output_dir),
            source_language=source_lang,
            target_language=target_lang,
            isolate_vocals=isolate_vocals,
            diarize=diarize,
            detect_emotion=emotion,
            resume=resume,
            tts_engine=engine,
            audio_prompt_path=Path(audio_prompt) if audio_prompt else None,
        )
        if max_speedup is not None:
            kwargs['max_speedup'] = max_speedup
        if elevenlabs_model:
            kwargs['elevenlabs_model'] = elevenlabs_model
        if elevenlabs_voice:
            # Check if it looks like a UUID or a human name
            from audiosmith.elevenlabs_tts import VOICE_MAP
            if elevenlabs_voice in VOICE_MAP:
                kwargs['elevenlabs_voice_name'] = elevenlabs_voice
            else:
                kwargs['elevenlabs_voice_id'] = elevenlabs_voice
        if indextts_model:
            kwargs['indextts_model'] = indextts_model
        if indextts_emo_alpha is not None:
            kwargs['indextts_emo_alpha'] = indextts_emo_alpha
        if indextts_emotion_audio:
            kwargs['indextts_emotion_prompt'] = Path(indextts_emotion_audio)
        if cosyvoice_model_dir:
            kwargs['cosyvoice_model_dir'] = cosyvoice_model_dir
        if cosyvoice_instruct:
            kwargs['cosyvoice_instruct'] = cosyvoice_instruct
        if orpheus_voice:
            kwargs['orpheus_voice'] = orpheus_voice
        if orpheus_temperature is not None:
            kwargs['orpheus_temperature'] = orpheus_temperature
        kwargs['post_process_tts'] = post_process_tts
        kwargs['post_process_intensity'] = post_process_intensity

        config = DubbingConfig(**kwargs)
        console.print(f"[dim]Engine: {engine} | Target: {target_lang} | Max speedup: {config.max_speedup}x[/dim]")
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
