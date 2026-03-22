"""TTS command — text-to-speech synthesis with multiple engines."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from audiosmith.exceptions import AudioSmithError

console = Console()


@click.command()
@click.argument('text')
@click.option('--engine', '-e', type=click.Choice(['piper', 'chatterbox', 'qwen3', 'fish', 'elevenlabs', 'indextts', 'cosyvoice', 'orpheus', 'f5']),
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
@click.option('--post-process/--no-post-process', 'post_process', default=True,
              help='Post-process local TTS for naturalness.')
def tts(text, engine, voice, output, language, model_type, instruct, ref_audio, ref_text, interactive, post_process):
    """Synthesize speech from text with Piper, Chatterbox, Qwen3, or ElevenLabs."""
    import numpy as np

    output_path = Path(output)

    if interactive:
        engine = Prompt.ask("Choose engine", choices=["piper", "chatterbox", "qwen3", "fish", "elevenlabs"], default=engine)
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

        elif engine == 'fish':
            from audiosmith.fish_speech_tts import FishSpeechTTS
            with console.status("[bold cyan]Calling Fish Speech API...[/bold cyan]", spinner="dots"):
                fish = FishSpeechTTS()
                if ref_audio:
                    fish.create_voice_clone('clone', ref_audio=ref_audio, ref_text=ref_text)
                audio, sr = fish.synthesize(text, voice='clone' if ref_audio else voice, language=language)
            fish.save_audio(audio, output_path, sr)
            sample_rate = sr
            fish.cleanup()

        elif engine == 'indextts':
            from audiosmith.indextts_tts import IndexTTS2TTS
            with console.status("[bold cyan]Loading IndexTTS-2...[/bold cyan]", spinner="dots"):
                idx = IndexTTS2TTS()
                if ref_audio:
                    idx.create_voice_clone('clone', ref_audio=ref_audio)
                audio, sr = idx.synthesize(
                    text, voice='clone' if ref_audio else voice, language=language,
                )
            import soundfile as sf
            sf.write(str(output_path), audio, sr)
            sample_rate = sr
            idx.cleanup()

        elif engine == 'cosyvoice':
            from audiosmith.cosyvoice_tts import CosyVoice2TTS
            with console.status("[bold cyan]Loading CosyVoice2...[/bold cyan]", spinner="dots"):
                cv = CosyVoice2TTS()
                if ref_audio:
                    cv.create_voice_clone('clone', ref_audio=ref_audio, ref_text=ref_text)
                audio, sr = cv.synthesize(
                    text, voice='clone' if ref_audio else None,
                    language=language, instruct=instruct,
                )
            import soundfile as sf
            sf.write(str(output_path), audio, sr)
            sample_rate = sr
            cv.cleanup()

        elif engine == 'orpheus':
            from audiosmith.orpheus_tts import OrpheusTTS
            with console.status("[bold cyan]Loading Orpheus...[/bold cyan]", spinner="dots"):
                orph = OrpheusTTS(voice=voice or 'tara')
                if ref_audio:
                    orph.create_voice_clone('clone', ref_audio=ref_audio)
                audio, sr = orph.synthesize(
                    text, voice='clone' if ref_audio else (voice or 'tara'),
                    language=language,
                )
            import soundfile as sf
            sf.write(str(output_path), audio, sr)
            sample_rate = sr
            orph.cleanup()

        elif engine == 'f5':
            from audiosmith.f5_tts import F5TTS
            with console.status("[bold cyan]Loading F5-TTS...[/bold cyan]", spinner="dots"):
                f5 = F5TTS()
                if ref_audio:
                    f5.clone_voice('clone', audio_path_or_array=ref_audio, ref_text=ref_text)
                audio, sr = f5.synthesize(text, voice='clone' if ref_audio else None, language=language)
            f5.save_audio(audio, output_path, sr)
            sample_rate = sr
            f5.cleanup()

        elif engine == 'elevenlabs':
            from audiosmith.elevenlabs_tts import ElevenLabsTTS
            with console.status("[bold cyan]Calling ElevenLabs API...[/bold cyan]", spinner="dots"):
                el = ElevenLabsTTS(voice_name=voice or 'Rachel')
                audio, sr = el.synthesize(text)
            el.save_audio(audio, output_path, sr)
            sample_rate = sr
            el.cleanup()

        # Post-process TTS for naturalness (skip ElevenLabs, custom config for Fish)
        if post_process and engine != 'elevenlabs':
            try:
                from audiosmith.tts_postprocessor import (PostProcessConfig,
                                                          TTSPostProcessor)
                if engine == 'fish':
                    pp_config = PostProcessConfig(
                        enable_silence=False, enable_dynamics=True,
                        enable_breath=True, enable_warmth=False,
                        enable_normalize=True, target_rms=0.14,
                        spectral_tilt=-1.0,
                        global_intensity=0.8,
                    )
                else:
                    pp_config = PostProcessConfig()
                pp = TTSPostProcessor(pp_config)
                audio = pp.process(audio, sample_rate, text=text)
                # Re-save with post-processed audio
                import soundfile as sf
                sf.write(str(output_path), audio, sample_rate)
            except Exception as e:
                console.print(f"[dim]Post-processing skipped: {e}[/dim]")

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
