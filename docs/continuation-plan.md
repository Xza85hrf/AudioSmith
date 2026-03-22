# AudioSmith Continuation Plan

> Generated: 2026-03-22 | Version: 0.5.0 | Tests: 1091 | Branch: main

## Current State

The v0.5.0 cleanup is complete. The codebase is well-organized with:
- 3 packages (`commands/`, `pipeline/`, `postprocessing/`) replacing monolithic files
- Multi-language support (5 languages via `language_data.py`)
- Full CI enforcement: tests (75%), ruff, mypy (strict), bandit
- 1091 tests, 0 mypy errors, 0 bandit high/medium

## Next Phase: Feature Development

### Priority 1: Improve Pipeline Test Coverage (pipeline/ at 16%)

The pipeline package has the lowest coverage because its methods require mocking
heavy external dependencies (FFmpeg, Whisper, TTS engines, Demucs). Fix with:

- [ ] Mock-based tests for each pipeline step in isolation
- [ ] Test `_resolve_engine()` auto-selection logic
- [ ] Test `_build_synthesis_kwargs()` for all engine types
- [ ] Test `_merge_segments()` and `_split_long_segment()` edge cases
- [ ] Test checkpoint/resume for each step transition
- [ ] Target: raise pipeline coverage from 16% to 60%+

### Priority 2: Add Language Packs

The `language_data.py` infrastructure is ready. Extend with:

- [ ] Add tech correction patterns for German (Whisper German phonetic mishearings)
- [ ] Add tech correction patterns for Spanish
- [ ] Add ultimate stress prosody for French (currently only penultimate implemented)
- [ ] Add question intonation variants (French uses rising intonation differently)
- [ ] Test with real audio samples in each language

### Priority 3: CLI Integration Tests for Remaining Commands

Currently tested: `transcribe`, `tts` (piper, chatterbox), `dub`, `check`, `normalize`, `export`, `batch`, `extract-voices`. Missing:

- [ ] `translate` command E2E test
- [ ] `transcribe-url` command E2E test (mock yt-dlp)
- [ ] `info` and `voices` commands
- [ ] `train-data-gen` and `train-f5` commands
- [ ] Error path testing (network failures, corrupt files, OOM)

### Priority 4: Performance & Scalability

- [ ] Profile TTS post-processing chain — spectral correction is the heaviest step
- [ ] Add async/parallel TTS generation for non-dependent segments
- [ ] Memory-map large audio files instead of loading fully into RAM
- [ ] Pipeline step timing metrics to `DubbingResult`

### Priority 5: New Features

- [ ] Streaming TTS output (synthesize and play in real-time)
- [ ] Web UI (optional, for users who prefer visual control)
- [ ] Pre-built Docker image with all optional deps
- [ ] Voice cloning quality benchmark (MOS scoring)
- [ ] A/B comparison tool for TTS engine selection

## Architecture Notes

### Key Decisions (for future developers)

1. **Mixin pattern for pipeline** — `TTSSynthesisMixin` keeps TTS logic separate while
   sharing `self.config`/`self.state`. Tradeoff: less explicit than composition, but
   avoids breaking the existing `DubbingPipeline` API.

2. **Language config is data, not inheritance** — `LanguageConfig` is a frozen dataclass,
   not an abstract base class. Adding a language is adding 10 lines of data, not a class.
   `get_language()` returns a fallback for unknown codes.

3. **Replace, don't deprecate** — Old modules are deleted, not wrapped. Backward compat
   wrappers were removed this session. If something was replaced, the old version is gone.

4. **Type ignores are targeted** — All `# type: ignore[error-code]` comments use specific
   error codes, not bare `# type: ignore`. This ensures new real errors are still caught.

5. **CI is strict** — mypy runs without `|| true`. New code must type-check. Bandit runs
   at `-ll` (medium+ severity). Ruff enforces import hygiene.

### File Size Budget

| Module | Current | Target | Notes |
|--------|---------|--------|-------|
| pipeline/core.py | 674 | <500 | Extract transcription/translation helpers next |
| pipeline/tts_synthesis.py | 429 | OK | Self-contained, one concern |
| cli.py | 273 | OK | Entry point, delegates to commands/ |
| qwen3_tts.py | 314 | OK | Complex engine, justified size |
| postprocessing/processor.py | 318 | OK | 13-step chain, justified |

### Test Categories

| Category | Count | Coverage Focus |
|----------|-------|---------------|
| Unit tests | ~900 | Individual functions/classes |
| Integration tests | ~120 | Multi-module interactions |
| CLI E2E tests | ~28 | Command execution with mocked backends |
| Checkpoint tests | ~49 | Pipeline resume/recovery |
| Language tests | ~88 | Multi-language parameterization |
| Skipped | 7 | Require torch/torchaudio (GPU) |
