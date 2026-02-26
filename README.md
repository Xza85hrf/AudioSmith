# AudioSmith

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A CLI-first toolkit for high-quality audio and video dubbing, transcription, translation, and speech synthesis.

## Features

- **Video dubbing**: End-to-end 10-step pipeline (transcribe → translate → TTS → mix → encode) with checkpoint/resume
- **Standalone transcription**: Faster-Whisper with output in SRT, VTT, TXT, or JSON
- **URL transcription**: Direct transcription of YouTube and other platforms via yt-dlp
- **Subtitle translation**: Offline via Argos or GPU-accelerated via TranslateGemma
- **4 TTS engines**: Chatterbox (23-lang voice cloning), Qwen3 (premium/cloned/designed voices), Piper (lightweight ONNX), MultiVoice (speaker-aware)
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
| `tts` | Text-to-speech synthesis (Chatterbox, Qwen3, or Piper) |
| `batch` | Process multiple files in one run |
| `export` | Convert SRT to TXT, PDF, or DOCX |
| `normalize` | Analyze and normalize audio loudness (LUFS) |
| `extract-voices` | Extract voice samples from audio for cloning |
| `check` | System pre-flight checks (FFmpeg, CUDA, disk space) |
| `info` | Show system info, available engines, and capabilities |
| `voices` | Browse available voices across all TTS engines |

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
| **Piper** | Pre-trained ONNX models | English, Polish | Lightweight, fast, CPU-friendly |
| **MultiVoice** | Speaker-aware | Per-speaker | Auto-assigns distinct voices per speaker ID |

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
├── cli.py                      # Rich CLI (12 commands)
├── pipeline.py                 # 10-step dubbing orchestrator with checkpoint/resume
├── transcribe.py               # Faster-Whisper transcription
├── translate.py                # Argos + TranslateGemma translation
├── tts.py                      # Chatterbox multilingual TTS (voice cloning)
├── qwen3_tts.py                # Qwen3 TTS (premium, cloning, design)
├── piper_tts.py                # Piper lightweight ONNX TTS
├── multi_voice_tts.py          # Speaker-aware multi-voice TTS
├── voice_extractor.py          # Voice sample extraction and cataloging
├── diarizer.py                 # Speaker diarization (PyAnnote)
├── emotion.py                  # Emotion detection engine
├── vocal_isolator.py           # Vocal isolation (Demucs)
├── srt_formatter.py            # Broadcast-quality SRT formatting
├── audio_normalizer.py         # LUFS analysis and normalization
├── document_formatter.py       # SRT → TXT/PDF/DOCX export
├── batch_processor.py          # Multi-file batch processing
├── mixer.py                    # Audio scheduling and rendering
├── ffmpeg.py                   # FFmpeg audio/video operations
├── download.py                 # yt-dlp download and format helpers
├── srt.py                      # SRT parsing and writing
├── vad.py                      # Voice activity detection (Silero)
├── transcription_post_processor.py  # Post-processing pipeline
├── punctuation_restorer.py     # Punctuation restoration
├── content_validator.py        # Content validation
├── tech_corrections.py         # Technical term corrections
├── transcript_corrector.py     # LLM-based transcript correction
├── language_detect.py          # Language detection
├── input_handler.py            # Input file handling
├── system_check.py             # System pre-flight checks
├── memory_manager.py           # GPU/RAM memory management
├── metrics.py                  # Processing metrics
├── progress.py                 # Progress tracking
├── retry.py                    # Retry with backoff
├── models.py                   # Data models (DubbingSegment, DubbingConfig)
├── exceptions.py               # Exception hierarchy
├── error_codes.py              # Error code catalog
└── log.py                      # Logging setup
```

## Supported Languages

**Chatterbox TTS (23 languages):**
`en`, `pl`, `de`, `fr`, `es`, `it`, `pt`, `ru`, `ja`, `ko`, `zh`, `ar`, `hi`, `nl`, `sv`, `da`, `fi`, `el`, `he`, `ms`, `nb`, `sw`, `tr`

**Qwen3 TTS (10 languages):**
English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

**Piper TTS:** English, Polish (pre-trained ONNX models)

**Translation (Argos):** See [Argos Translate language pairs](https://www.argosopentech.com/argospopular/)
**Translation (TranslateGemma):** Supports all languages available in the model.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

402 unit tests, ~2s runtime, no GPU required.

## License

MIT
