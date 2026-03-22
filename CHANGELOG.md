# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] — 2026-03-22

### Added

- TTSEngine protocol and factory pattern for unified interface across 9 TTS engines
- Multi-language support infrastructure: `language_data.py` with language-specific configs (Polish, English, Spanish, French, German)
- Bandit security scanning in CI workflow
- 1081 tests across pipeline, quality modules, and CLI (up from ~300 in 0.4.x)
- Type hints on all public functions across 25 modules
- Pipeline checkpoint/resume functionality with `PipelineState` and `DubbingStep` (49 tests)
- Spectral profiles and emotion-aware post-processing modules
- `torchaudio` added to `[quality]` optional dependencies

### Changed

- Split monolithic `cli.py` (55KB) into `audiosmith/commands/` package with modular command handlers (dub, tts, transcribe, batch)
- Split large `tts_postprocessor.py` (38KB) into `audiosmith/postprocessing/` filter package
- Consolidated 4 emotion configuration files into single `emotion_config.py` module
- Consolidated pipeline preset configurations into dedicated `pipeline_config.py` module
- Renamed `polish_prosody.py` to `prosody.py` with language-parameterized behavior
- `PunctuationRestorer` now accepts `language` parameter for language-aware processing
- `TechTermCorrections` now accepts `language` parameter (no-op for non-Polish languages)
- `TranscriptionPostProcessor` now accepts `language` parameter for language-specific processing
- Prosody functions now gracefully skip inapplicable languages (e.g., penultimate stress rules only for Polish)
- Coverage threshold increased from 60% to 75%

### Removed

- Deprecated re-export wrapper modules: `polish_corpus.py`, `f5_finetune.py`, `qwen3_finetune.py`, `training_data_gen.py`
- Hardcoded `POLISH_VOWELS` constant (replaced by dynamic `language_data.get_language()` API)
- Hardcoded `QUESTION_STARTERS` constant (replaced by per-language configuration)

### Fixed

- `_EMOTION_STYLE_MAP` NameError in pipeline.py that would crash when using ElevenLabs engine with emotion parameter
- `include_speakers` bug in export command
- MD5 hash computation now uses `usedforsecurity=False` for FIPS/bandit compliance
- 15 unused imports removed across codebase
- Export fixes for 12 modules in `__init__.py`

## [0.4.0] — 2026-03-15

### Added

- F5-TTS engine with Polish fine-tuning pipeline and 64 comprehensive tests
- Polish TTS training data generation pipeline
- IndexTTS-2, CosyVoice2, and Orpheus TTS engine implementations
- ElevenLabs and Fish Speech cloud TTS engines
- TTS post-processor module for naturalness enhancement
- Pipeline quality upgrades and validation

### Changed

- Qwen3 TTS API overhaul with voice design and premium voice support across 10 languages
- CLI redesign with Rich UI for improved user experience
- Added `--engine` and `--audio-prompt` options to dub command

[Unreleased]: https://github.com/yourusername/AudioSmith/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/yourusername/AudioSmith/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/yourusername/AudioSmith/releases/tag/v0.4.0
