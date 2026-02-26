# AudioSmith Quality Features

Advanced audio processing features ported from VoiceSmith, adapted to AudioSmith's lean CLI architecture.

## Speaker Diarization

Identifies individual speakers in multi-speaker audio using PyAnnote Audio 3.0.

**Usage:**
```bash
audiosmith dub video.mp4 --target-lang pl --diarize
audiosmith transcribe audio.wav --output srt --diarize
```

**How it works:**
1. PyAnnote segments audio into speaker turns
2. Each transcription segment gets a `speaker_id` label (SPEAKER_00, SPEAKER_01, ...)
3. When combined with Multi-Voice TTS, each speaker gets a distinct cloned voice

**Requirements:** `pip install 'audiosmith[quality]'` (includes `pyannote-audio>=3.0`)

---

## Emotion Detection

Detects emotional tone in speech segments using a rule-based analyzer with optional ML enhancement.

**Usage:**
```bash
audiosmith dub video.mp4 --target-lang pl --emotion
```

**How it works:**
1. Rule-based analysis of text patterns, punctuation, and keywords
2. Detects 10 emotions: happy, sad, angry, fearful, surprised, whisper, sarcastic, tender, excited, determined
3. Emotion + intensity are mapped to Chatterbox TTS parameters (exaggeration, cfg_weight)
4. Intensity-scaled interpolation ensures natural-sounding output

**Supported emotions and TTS mapping:**

| Emotion | Exaggeration | CFG Weight | Effect |
|---------|-------------|------------|--------|
| happy | 0.7 | 0.5 | Brighter, more energetic |
| sad | 0.3 | 0.4 | Subdued, slower |
| angry | 0.9 | 0.7 | Intense, forceful |
| whisper | 0.2 | 0.3 | Quiet, intimate |
| excited | 0.8 | 0.6 | High energy, fast |

---

## SRT Formatter

Professional subtitle formatting following broadcast standards.

**Standards enforced:**
- 42 characters per line maximum
- Maximum 2 lines per subtitle
- Duration: 1–7 seconds per subtitle
- 40ms minimum gap between subtitles
- Word-level timestamp splitting for long segments
- Smart line breaking at natural word boundaries

**Automatically applied** in all SRT output (transcribe, transcribe-url, and dub commands).

---

## Vocal Isolation

Separates vocals from background audio using Demucs (htdemucs model) before transcription.

**Usage:**
```bash
audiosmith dub video.mp4 --target-lang pl --isolate-vocals
audiosmith transcribe audio.wav --output srt --isolate-vocals
```

**How it works:**
1. Demucs htdemucs model separates audio into vocals + background
2. Vocals are resampled to 16kHz mono for Whisper transcription
3. Background track is preserved for mixing back into the final output
4. Mono audio is automatically converted to stereo for Demucs compatibility

**Requirements:** `pip install 'audiosmith[quality]'` (includes `demucs>=4.0`)

---

## Multi-Voice TTS

Speaker-aware voice cloning that assigns distinct voices to different speakers.

**How it works:**
1. Automatically activated when segments have `speaker_id` or `emotion` metadata
2. Voice prompts are auto-assigned from WAV files in a voice directory
3. Each speaker gets a consistent cloned voice throughout the output
4. Emotion parameters modulate synthesis for natural expressiveness

**Voice assignment:**
- Place WAV files named `SPEAKER_00.wav`, `SPEAKER_01.wav`, etc. in the output directory
- Or provide a default voice prompt via `--audio-prompt`
- Voices are cached per speaker for consistent output

---

## TTS Engines

AudioSmith supports four TTS engines, each suited for different use cases.

### Chatterbox (Default)

Multilingual zero-shot voice cloning via ResembleAI Chatterbox. Supports 23 languages with emotion modulation.

```bash
audiosmith tts "Hello world" -o output.wav --engine chatterbox
audiosmith tts "Hello world" -o output.wav --engine chatterbox --audio-prompt voice.wav
```

### Qwen3

Premium named voices, voice cloning (ICL/x-vector), and text-described voice design. 10 languages, 9 premium voices.

```bash
# Premium voice
audiosmith tts "Hello world" -o output.wav --engine qwen3 --voice Ryan

# Voice cloning from reference audio
audiosmith tts "Hello world" -o output.wav --engine qwen3 --ref-audio sample.wav

# Voice design from text description
audiosmith tts "Hello world" -o output.wav --engine qwen3 --instruct "Male, 30 years old, warm and calm"

# Interactive mode
audiosmith tts "Hello" -o out.wav --engine qwen3 -i
```

**Qwen3 model types:** `base` (voice cloning), `voice_design` (text-described voices), `custom_voice` (premium named speakers). Auto-selected based on options.

**Premium voices:** Ryan, Aiden, Dylan, Eric, Serena, Luna, Mia, Aria, Ethan

### Piper

Lightweight ONNX-based TTS for fast CPU inference. Pre-trained English and Polish voices.

```bash
audiosmith tts "Hello world" -o output.wav --engine piper
audiosmith tts "Hello world" -o output.wav --engine piper --voice en_US-lessac-medium
```

### Voice Extraction

Extract voice samples from audio files for use with voice cloning:

```bash
# Extract 5 evenly-spaced samples
audiosmith extract-voices recording.wav -n 5

# Extract with speaker diarization
audiosmith extract-voices recording.wav --diarize

# Save voice catalog
audiosmith extract-voices recording.wav -n 5 --catalog voices/
```

---

## Combining Features

Features can be stacked for maximum quality:

```bash
# Full quality pipeline: isolate vocals, identify speakers, detect emotion
audiosmith dub video.mp4 --target-lang pl \
    --isolate-vocals \
    --diarize \
    --emotion
```

**Pipeline with all quality features enabled:**
```
Extract Audio → Isolate Vocals → Transcribe → Diarize → Detect Emotion → Translate → Generate TTS → Mix Audio → Encode Video
```

Without quality features, the pipeline uses the core 6 steps (Extract → Transcribe → Translate → TTS → Mix → Encode).
