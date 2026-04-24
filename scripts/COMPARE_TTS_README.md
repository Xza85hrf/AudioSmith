# TTS Engine Comparison Script

## Overview

`compare_tts_engines.py` generates the first 10 minutes of a Polish SRT subtitle file through 5 different TTS engines for quality comparison. This tool helps evaluate voice quality, naturalness, and language handling across multiple synthesis backends.

## Usage

```bash
python scripts/compare_tts_engines.py <path_to_srt_file>
```

### Example

```bash
python scripts/compare_tts_engines.py test-files/videos/Original_subtitiles/Marty.Supreme.2025.pl.srt
```

## Supported TTS Engines

1. **Piper** — CPU-friendly ONNX-based synthesis
   - Voice: `pl_PL-darkman-medium` (Polish)
   - Fast, low-resource, reliable

2. **F5-TTS** — Flow-matching synthesis with Polish support
   - Device: GPU (CUDA)
   - High-quality, natural-sounding output
   - Polish checkpoint available

3. **Chatterbox** — Multilingual synthesis fallback
   - Device: GPU (CUDA)
   - Supports emotion and expressiveness parameters

4. **Qwen3-TTS** — Premium voice cloning and design
   - Device: GPU (CUDA)
   - 10 languages including Polish
   - Streaming synthesis capability

5. **Fish Speech S2-Pro** — Local server-based synthesis
   - Requires: Fish Speech server running on `http://127.0.0.1:8080`
   - Best for long-form, context-aware output
   - Parameters: `temperature=0.5`, `top_p=0.7`

## Output

Generated WAV files are saved to `test-files/tts_comparison/`:

```
test-files/tts_comparison/
├── piper.wav       (22,050 Hz)
├── f5.wav          (24,000 Hz)
├── chatterbox.wav  (24,000 Hz)
├── qwen3.wav       (24,000 Hz)
└── fish.wav        (24,000 Hz)
```

Each WAV contains the full first 10 minutes of synthesized audio with 300ms silence gaps between segments.

## SRT Processing

The script performs the following preprocessing on each SRT entry:

1. **Text Cleaning** — Removes non-speakable content:
   - Bracketed stage directions: `[Muzyka]`, `[chrząkanie]`
   - Parenthetical notes: `(laughing)`, `(whispering)`
   - Music notation: `♪ lyrics ♪`
   - Leading dialogue dashes: `– Text` → `Text`

2. **Deduplication** — Collapses runs of 3+ identical words to 2 max

3. **Filtering** — Skips segments with:
   - Empty text after cleaning
   - Less than 2 words
   - Start time after 600 seconds (10 minutes)

## Parsing Results (Polish Example)

For `Marty.Supreme.2025.pl.srt`:

```
Total SRT entries:     2,789
Valid segments (10m):  183
Total words:           1,254
Avg segment length:    6.9 words
Time coverage:         600 seconds (10 minutes)
```

### Sample Segments

1. `[29.5s] Pani Mariann. Czy mam dziewięć i pół?`
2. `[31.4s] Tak, panienka, szczęściarka. Oh, jestem pod wrażeniem.`
3. `[33.0s] Ostatnia para. Pokażcie mi te piękne stopy.`

## Requirements

### Python Packages
- `numpy` — Audio array manipulation
- `soundfile` — WAV file I/O
- `piper-tts` — Piper engine (pip install piper-tts)
- `f5-tts` — F5 engine (optional)
- `qwen-tts` — Qwen3 engine (optional)
- `fish-speech` — Fish engine client (optional, for server mode)

### System Requirements

- **Piper**: CPU only (no GPU needed)
- **F5, Chatterbox, Qwen3**: CUDA-capable GPU (16GB+ VRAM recommended for multiple engines)
- **Fish Speech**: Local server running on `127.0.0.1:8080` (separate installation)

## Logging

The script logs detailed progress to stderr:

```
INFO Parsed 183 segments from first 600 seconds
INFO Sample: [29.5-31.4] Pani Mariann. Czy mam dziewięć i pół?
INFO === Generating with PIPER ===
INFO Loading TTS engine 'piper' into VRAM
INFO   piper: 20/183 segments done
INFO   piper: 183/183 segments in 45.3s
INFO   piper: saved 184.5s of audio to test-files/tts_comparison/piper.wav

============================================================
TTS ENGINE COMPARISON — RESULTS
============================================================
  piper       : 184.5s  piper.wav
  f5          : 183.2s  f5.wav
  chatterbox  : 182.8s  chatterbox.wav
  qwen3       : 181.5s  qwen3.wav
  fish        : FAILED

Output directory: test-files/tts_comparison
```

## GPU Memory Management

- Engines are loaded sequentially (only one GPU engine active at a time)
- After each engine completes, VRAM is freed via `torch.cuda.empty_cache()`
- Garbage collection is forced between engine switches
- Total VRAM footprint never exceeds a single engine's requirements

## Error Handling

- Failed segments log a warning but do not stop processing
- Failed engines are logged but other engines continue
- Incomplete runs (0 segments generated) return None and log an error
- Missing `load_model()` method is handled gracefully

## Testing

Full test suite with 18 tests covering:

```bash
python -m pytest tests/test_compare_tts_engines.py -v

✓ SRT parsing and timestamp conversion
✓ Text cleaning and deduplication
✓ Segment filtering logic
✓ Real Polish SRT file parsing (183 segments)
✓ Audio concatenation and normalization
✓ Engine factory and protocol compliance
✓ Output directory creation
✓ Error handling for missing files
```

All tests pass on Python 3.10+.

## Quality Metrics

After generation, compare the WAV files:

1. **Latency** — Time to generate all segments
2. **Duration** — Total audio output (should be ~183-185 seconds including gaps)
3. **Voice Quality** — Listen for naturalness, pronunciation, intonation
4. **Language Handling** — Polish diacritics, consonant clusters, stress patterns
5. **Segment Transitions** — Smoothness between concatenated segments

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'piper'`
**Solution**: Install Piper TTS
```bash
pip install piper-tts
```

### Issue: `Fish Speech server not responding`
**Solution**: Start the Fish Speech server in another terminal
```bash
fish-speech-server --listen 127.0.0.1:8080
```

### Issue: `CUDA out of memory` during F5/Qwen3
**Solution**:
- Reduce batch size (if supported)
- Close other GPU applications
- Reduce max segments (edit `MAX_TIME_S` in script)

### Issue: Script generates no audio for an engine
**Solution**: Check engine logs above the error message. Likely causes:
- Missing dependencies
- Unsupported language code
- Model download/initialization failure

## Development

The script uses the AudioSmith TTS abstraction layer:

- `audiosmith.tts_protocol.get_engine()` — Factory creates engines by name
- `audiosmith.tts_manager.TTSModelManager` — Hot-swap engine management (not used in this script, but available)
- `audiosmith.srt.parse_srt()` — SRT parsing with timestamp support
- `audiosmith.pipeline.helpers._clean_tts_text()` — Text preprocessing

Extending with new engines:

1. Implement the `TTSEngine` protocol in `audiosmith/`
2. Register loader in `audiosmith/tts_protocol.py`
3. Add engine to `ENGINES` dict in this script with initialization kwargs

## License

Script is part of AudioSmith project. TTS engines have individual licenses:
- **Piper**: MIT
- **F5-TTS**: MIT
- **Chatterbox**: Apache 2.0
- **Qwen3-TTS**: Custom license
- **Fish Speech**: Custom license

For Polish language support, refer to individual engine documentation.
