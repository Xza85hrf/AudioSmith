# AudioSmith

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A CLI-first toolkit for high-quality audio and video dubbing, transcription, and subtitle processing.

## Features

- **Video dubbing**: End-to-end pipeline (transcribe → translate → TTS → mix → encode)
- **Standalone transcription**: Output in SRT, VTT, TXT, or JSON
- **Subtitle translation**: Offline via Argos or GPU-accelerated via TranslateGemma
- **URL transcription**: Direct transcription of YouTube and other supported platforms via yt-dlp
- **6-step pipeline with JSON checkpoint resume**: Resume interrupted jobs without reprocessing
- **23-language TTS**: Multilingual speech synthesis via Chatterbox
- **CLI-first**: No web UI; designed for automation and scripting

## Requirements

- Python 3.11+
- FFmpeg installed and available on `PATH`
- CUDA-capable GPU recommended (CPU fallback fully supported)

## Installation

```bash
git clone https://github.com/Xza85hrf/AudioSmith.git
cd AudioSmith
pip install -e ".[dev]"
```

Optional TranslateGemma support (requires CUDA):
```bash
pip install -e ".[gemma]"
```

## Quick Start

1. Dub a video to Polish:
   ```bash
   audiosmith dub video.mp4 --target-lang pl
   ```

2. Transcribe an audio file to SRT:
   ```bash
   audiosmith transcribe audio.wav --output srt
   ```

3. Translate subtitles to Spanish:
   ```bash
   audiosmith translate subs.srt --target-lang es
   ```

4. Transcribe a YouTube video to SRT:
   ```bash
   audiosmith transcribe-url "https://youtube.com/watch?v=..." --output srt
   ```

## Pipeline Architecture

The dubbing pipeline consists of six sequential steps:

```
Extract Audio → Transcribe → Translate → Generate TTS → Mix Audio → Encode Video
```

Each step writes intermediate artifacts and a JSON checkpoint. Resume partial runs with:
```bash
audiosmith dub video.mp4 --target-lang fr --resume
```

## Project Structure

```
audiosmith/
├── cli.py          # Click CLI (4 commands)
├── pipeline.py     # 6-step dubbing orchestrator
├── transcribe.py   # Faster-Whisper transcription
├── translate.py    # Argos + TranslateGemma translation
├── tts.py          # Chatterbox multilingual TTS
├── mixer.py        # Audio scheduling & rendering
├── ffmpeg.py       # FFmpeg audio/video operations
├── download.py     # yt-dlp download & format helpers
├── srt.py          # SRT parsing & writing
├── models.py       # Data models & pipeline state
├── exceptions.py   # Exception hierarchy
├── error_codes.py  # Error code catalog
└── log.py          # Logging setup
```

## Supported Languages

TTS languages (via Chatterbox):
`en`, `pl`, `de`, `fr`, `es`, `it`, `pt`, `ru`, `ja`, `ko`, `zh`, `ar`, `hi`, `nl`, `sv`, `da`, `fi`, `el`, `he`, `ms`, `nb`, `sw`, `tr`

Subtitle translation (Argos): See [Argos Translate language pairs](https://www.argosopentech.com/argospopular/)
Subtitle translation (TranslateGemma): Supports all languages available in the model.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

73 unit tests, ~10s runtime, no GPU required.

## License

MIT
