# AudioSmith

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A CLI-first toolkit for high-quality audio and video dubbing, transcription, translation, and speech synthesis.

## Features

- **Video dubbing**: End-to-end 10-step pipeline (transcribe → translate → TTS → mix → encode) with checkpoint/resume
- **Standalone transcription**: Faster-Whisper with output in SRT, VTT, TXT, or JSON
- **URL transcription**: Direct transcription of YouTube and other platforms via yt-dlp
- **Subtitle translation**: Offline via Argos or GPU-accelerated via TranslateGemma
- **9 TTS engines**: Chatterbox (23-lang voice cloning), Qwen3 (premium/cloned/designed voices), Piper (lightweight ONNX), F5 (flow matching), ElevenLabs (cloud, 70+ languages), Fish Speech (cloud), IndexTTS (emotion-aware), CosyVoice2 (5.53 MOS), Orpheus (13-lang expressive)
- **Voice extraction**: Extract and catalog voice samples from audio with optional speaker diarization
- **Speaker diarization**: Identify and label individual speakers (PyAnnote Audio)
- **Emotion detection**: Rule-based + optional ML emotion analysis for expressive TTS
- **Vocal isolation**: Separate vocals from background audio (Demucs)
- **Audio normalization**: LUFS analysis and loudness normalization
- **Document export**: Convert SRT to TXT, PDF, or DOCX
- **Batch processing**: Process multiple files with `--continue-on-error`
- **Broadcast-quality subtitles**: 42-char/line, word-level splitting, duration enforcement
- **Rich terminal UI**: Tables, panels, spinners, and interactive prompts via Rich
- **CLI-first**: No web UI; designed for automation and scripting

## Requirements

- Python 3.11+
- FFmpeg installed and available on `PATH`
- CUDA-capable GPU recommended (CPU fallback supported)

## Installation

```bash
git clone https://github.com/Xza85hrf/AudioSmith.git
cd AudioSmith
pip install -e ".[dev]"
```

Optional extras:
```bash
pip install -e ".[quality]"   # Speaker diarization (PyAnnote) + vocal isolation (Demucs)
pip install -e ".[gemma]"     # TranslateGemma (requires CUDA)
pip install -e ".[qwen3]"     # Qwen3 TTS engine
pip install -e ".[piper]"     # Piper TTS engine
pip install -e ".[docs]"      # PDF/DOCX export (fpdf2, python-docx)
pip install -e ".[all]"       # Everything above
```

## Commands

| Command | Description |
|---------|-------------|
| `dub` | Full dubbing pipeline: transcribe → translate → TTS → mix → encode |
| `transcribe` | Transcribe audio/video to SRT/VTT/TXT/JSON |
| `transcribe-url` | Download and transcribe from YouTube/supported URLs |
| `translate` | Translate subtitle files (Argos or TranslateGemma) |
| `tts` | Text-to-speech synthesis (9 engines: Chatterbox, Qwen3, Piper, F5, ElevenLabs, Fish Speech, IndexTTS, CosyVoice2, Orpheus) |
| `batch` | Process multiple files in one run |
| `export` | Convert SRT to TXT, PDF, or DOCX |
| `normalize` | Analyze and normalize audio loudness (LUFS) |
| `extract-voices` | Extract voice samples from audio for cloning |
| `check` | System pre-flight checks (FFmpeg, CUDA, disk space) |
| `info` | Show system info, available engines, and capabilities |
| `voices` | Browse available voices across all TTS engines |
| `train-data-gen` | Generate training data for F5-TTS fine-tuning (paired text+audio) |
| `train-f5` | Fine-tune F5-TTS for custom voices |

## Quick Start

```bash
# Dub a video to Polish
audiosmith dub video.mp4 --target-lang pl

# Full quality pipeline
audiosmith dub video.mp4 --target-lang pl --isolate-vocals --diarize --emotion

# Transcribe audio to SRT
audiosmith transcribe audio.wav --output srt

# Transcribe a YouTube video
audiosmith transcribe-url "https://youtube.com/watch?v=..." --output srt

# Translate subtitles to Spanish
audiosmith translate subs.srt --target-lang es

# Text-to-speech with Qwen3 premium voice
audiosmith tts "Hello world" -o output.wav --engine qwen3 --voice Ryan

# Text-to-speech with voice cloning
audiosmith tts "Hello world" -o output.wav --engine qwen3 --ref-audio sample.wav

# Interactive TTS mode
audiosmith tts "Hello" -o out.wav --engine qwen3 -i

# Extract voice samples from audio
audiosmith extract-voices recording.wav -n 5

# Normalize audio loudness
audiosmith normalize audio.mp3

# Export subtitles to PDF
audiosmith export subs.srt -f pdf

# Check system readiness
audiosmith check

# Browse available voices
audiosmith voices

# System info
audiosmith info
```

## TTS Engines

| Engine | Voices | Languages | Features |
|--------|--------|-----------|----------|
| **Chatterbox** | Zero-shot cloning | 23 languages | Voice cloning from audio prompt, emotion modulation |
| **Qwen3** | 9 premium + cloning + design | 10 languages | Premium named voices, voice cloning (ICL/x-vector), voice design from text description |
| **Piper** | Pre-trained ONNX models | English, Polish | Lightweight, fast, CPU-friendly, local |
| **F5-TTS** | Zero-shot cloning | Multi-language | Flow matching, fast inference, voice cloning |
| **ElevenLabs** | 70+ preset + cloning | 70+ languages | Cloud-based, high quality, voice cloning (premium) |
| **Fish Speech** | Zero-shot cloning | Multi-language | Cloud-based, fast, high naturalness |
| **IndexTTS-2** | Cloning + emotion | EN, ZH | Emotion disentanglement, controllable emotion synthesis |
| **CosyVoice2** | Cloning + instruct | 9 languages | MOS 5.53, zero-shot cloning, instruction control |
| **Orpheus** | 8 preset voices | 13 languages | Emotion tags, expressive synthesis, fast |

## Pipeline Architecture

The dubbing pipeline runs up to 10 steps (optional steps enabled via flags):

```
Extract Audio → [Isolate Vocals] → Transcribe → [Post-Process] → [Diarize] → [Detect Emotion] → Translate → Generate TTS → Mix Audio → Encode Video
```

Steps in brackets are optional. Each step writes intermediate artifacts and a JSON checkpoint. Resume interrupted jobs with:
```bash
audiosmith dub video.mp4 --target-lang fr --resume
```

See [docs/quality-features.md](docs/quality-features.md) for detailed documentation on quality features.

## Project Structure

```
audiosmith/
├── commands/                       # CLI command modules
│   ├── dub.py                      # Dub command
│   ├── tts_cmd.py                  # TTS command (9 engines)
│   ├── transcribe.py               # Transcribe command
│   └── batch.py                    # Batch processing command
├── pipeline/                       # Dubbing pipeline package
│   ├── core.py                     # DubbingPipeline orchestrator with checkpoint/resume
│   ├── tts_synthesis.py            # TTS engine init, synthesis, parameter building
│   └── helpers.py                  # Segment serialization, SRT writing, text processing
├── postprocessing/                 # TTS audio post-processing
│   ├── processor.py                # TTSPostProcessor orchestrator (13-step chain)
│   ├── spectral.py                 # Spectral correction, presence synthesis, warmth
│   ├── dynamics.py                 # Dynamic range expansion and reshaping
│   ├── silence.py                  # Silence trimming and punctuation pauses
│   └── config.py                   # PostProcessConfig dataclass
├── cli.py                          # Rich CLI entry point (14 commands)
├── language_data.py                # Multi-language config (pl, en, es, fr, de)
├── prosody.py                      # Language-aware prosody (stress, intonation, timing)
├── emotion_config.py               # Centralized emotion → TTS parameter mappings
├── pipeline_config.py              # Engine presets and language overrides
├── tts_protocol.py                 # TTSEngine protocol + factory
├── tts.py                          # Chatterbox TTS (23-lang voice cloning)
├── qwen3_tts.py                    # Qwen3 TTS (premium, cloning, design)
├── piper_tts.py                    # Piper lightweight ONNX TTS
├── f5_tts.py                       # F5-TTS (flow matching, voice cloning)
├── elevenlabs_tts.py               # ElevenLabs cloud TTS
├── fish_speech_tts.py              # Fish Speech cloud TTS
├── indextts_tts.py                 # IndexTTS-2 (emotion-aware)
├── cosyvoice_tts.py                # CosyVoice2 (5.53 MOS)
├── orpheus_tts.py                  # Orpheus (13-lang expressive)
├── multi_voice_tts.py              # Speaker-aware multi-voice TTS
├── transcribe.py                   # Faster-Whisper transcription
├── translate.py                    # Argos + TranslateGemma translation
├── transcription_post_processor.py # 4-stage text post-processing (language-aware)
├── punctuation_restorer.py         # Punctuation restoration (multi-language)
├── tech_corrections.py             # Technical term corrections (language-aware)
├── voice_extractor.py              # Voice sample extraction and cataloging
├── diarizer.py                     # Speaker diarization (PyAnnote)
├── emotion.py                      # Emotion detection engine
├── vocal_isolator.py               # Vocal isolation (Demucs)
├── spectral_profiles.py            # Emotion-specific spectral targets
├── srt_formatter.py                # Broadcast-quality SRT formatting
├── audio_normalizer.py             # LUFS analysis and normalization
├── document_formatter.py           # SRT → TXT/PDF/DOCX export
├── batch_processor.py              # Multi-file batch processing
├── mixer.py                        # Audio scheduling and rendering
├── ffmpeg.py                       # FFmpeg audio/video operations
├── models.py                       # Data models (DubbingSegment, DubbingConfig, etc.)
├── exceptions.py                   # Exception hierarchy
└── error_codes.py                  # Error code catalog
```

## Supported Languages

**Chatterbox TTS (23 languages):**
`en`, `pl`, `de`, `fr`, `es`, `it`, `pt`, `ru`, `ja`, `ko`, `zh`, `ar`, `hi`, `nl`, `sv`, `da`, `fi`, `el`, `he`, `ms`, `nb`, `sw`, `tr`

**Qwen3 TTS (10 languages):**
English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

**Piper TTS:** English, Polish (pre-trained ONNX models)

**Translation (Argos):** See [Argos Translate language pairs](https://www.argosopentech.com/argospopular/)
**Translation (TranslateGemma):** Supports all languages available in the model.

**Text Processing:**
Question detection, technical term corrections, and prosody (stress patterns, syllable timing) are language-aware. Configure via `language_data.py` — currently supports `pl`, `en`, `es`, `fr`, `de` with extensible `LanguageConfig` dataclass.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

1091 unit tests, ~15s runtime, no GPU required.

## License

MIT
