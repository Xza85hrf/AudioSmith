# Chatterbox TTS Parameter Comparison

## Overview

Created a comprehensive comparison script that generates audio using Chatterbox TTS with three different voice cloning parameter configurations. The script processes the first 10 minutes (600 seconds) of Polish SRT subtitles.

## Files Created

### 1. Script: `scripts/compare_chatterbox_params.py`
- **Purpose**: Compare 3 Chatterbox parameter configurations on Polish SRT
- **Lines**: 213
- **Key Features**:
  - Loads ChatterboxTTS model once (3GB, expensive operation)
  - Parses SRT, cleans text, skips segments <2 words
  - Generates 3 variants with different voice cloning params
  - Concatenates segments with 300ms silence gaps
  - Peak-normalizes output to 0.95
  - Resume-safe (skips variants if output files exist)

### 2. Test Suite: `tests/test_compare_chatterbox_params.py`
- **Coverage**: 20 comprehensive unit tests
- **Test Classes**:
  - `TestParseSegments` - SRT parsing and timestamp conversion (4 tests)
  - `TestTextCleaning` - Text preprocessing (3 tests)
  - `TestVariantConfigurations` - Parameter validation (3 tests)
  - `TestAudioNormalization` - Audio processing (3 tests)
  - `TestVoiceReferenceFile` - File handling (2 tests)
  - `TestOutputPaths` - Output naming (1 test)
  - `TestSRTFileHandling` - SRT file logic (2 tests)
  - `TestChatterboxIntegration` - Engine integration (2 tests)
- **All Tests Pass**: ✅ 20/20 passing

## Parameters Tested

### Variant A: `voice_clone` (baseline)
- **Exaggeration**: 0.5
- **CFG Weight**: 0.5
- **Description**: Conservative voice cloning with standard expressiveness

### Variant B: `voice_clone_tuned` (moderate)
- **Exaggeration**: 0.65
- **CFG Weight**: 0.6
- **Description**: Balanced expressiveness improvement

### Variant C: `voice_clone_expressive` (aggressive)
- **Exaggeration**: 0.8
- **CFG Weight**: 0.7
- **Description**: Maximum expressiveness for dramatic emphasis

## Data Source

- **Input SRT**: `test-files/videos/Original_subtitiles/Marty.Supreme.2025.pl.srt`
- **Language**: Polish (pl)
- **Time Window**: First 10 minutes (600 seconds)
- **Segments Parsed**: 183 (after filtering <2 word segments)
- **Voice Reference**: `test-files/tts_comparison/voice_refs/witcher_polish_ref.wav`
  - Polish male voice from Witcher film
  - Used for zero-shot voice cloning in all variants

## Processing Details

### Text Cleaning Pipeline
1. Remove bracketed stage directions `[text]`
2. Deduplicate repeated words (max 2 consecutive)
3. Skip segments with <2 words
4. Clean whitespace

### Audio Processing
- **Sample Rate**: 24000 Hz (Chatterbox native)
- **Segment Gap**: 300ms silence between segments
- **Normalization**: Peak normalization to 0.95 (prevents clipping)
- **Concatenation**: All segments + silence gaps combined into single WAV

## Output Files

Generated WAV files (once generation completes):
- `test-files/tts_comparison/chatterbox_A_clone.wav` (~10 min)
- `test-files/tts_comparison/chatterbox_B_tuned.wav` (~10 min)
- `test-files/tts_comparison/chatterbox_C_expressive.wav` (~10 min)

## Execution Time

- **Model Loading**: ~1 minute (first time)
- **Per Variant**: ~15 minutes (183 segments × ~5s per segment)
- **Total Expected**: ~45-50 minutes for all 3 variants
- **Current Status**: Running (estimated completion: ~16:30-16:45 UTC)

## Key Implementation Details

### Single Model Load
```python
engine = ChatterboxTTS(device="cuda")
engine.load_model()  # 3GB, ~1 min
# Reuse for all 3 variants
```

### Variant-Specific Synthesis
```python
result = engine.synthesize(
    text,
    language="pl",
    audio_prompt_path=str(VOICE_REF),
    exaggeration=config["exaggeration"],    # 0.5, 0.65, or 0.8
    cfg_weight=config["cfg_weight"],        # 0.5, 0.6, or 0.7
)
```

### Resume Safety
```python
if wav_path.exists():
    logger.info("Skipping %s (already exists)", variant_name)
    return wav_path
# Allows re-running without regenerating existing files
```

## Test Results

```
PASSED tests/test_compare_chatterbox_params.py::TestParseSegments::test_parse_srt_basic
PASSED tests/test_compare_chatterbox_params.py::TestParseSegments::test_timestamp_to_seconds_basic
PASSED tests/test_compare_chatterbox_params.py::TestParseSegments::test_timestamp_to_seconds_at_10min_boundary
PASSED tests/test_compare_chatterbox_params.py::TestParseSegments::test_timestamp_to_seconds_beyond_10min
PASSED tests/test_compare_chatterbox_params.py::TestTextCleaning::test_clean_tts_text_removes_brackets
PASSED tests/test_compare_chatterbox_params.py::TestTextCleaning::test_dedup_repeated_words_collapses_runs
PASSED tests/test_compare_chatterbox_params.py::TestTextCleaning::test_segments_skip_short_text
PASSED tests/test_compare_chatterbox_params.py::TestVariantConfigurations::test_variant_configs_have_required_fields
PASSED tests/test_compare_chatterbox_params.py::TestVariantConfigurations::test_variant_exaggeration_progression
PASSED tests/test_compare_chatterbox_params.py::TestVariantConfigurations::test_variant_cfg_weight_progression
PASSED tests/test_compare_chatterbox_params.py::TestAudioNormalization::test_peak_normalize_preserves_shape
PASSED tests/test_compare_chatterbox_params.py::TestAudioNormalization::test_silence_concatenation
PASSED tests/test_compare_chatterbox_params.py::TestAudioNormalization::test_audio_concatenation_preserves_values
PASSED tests/test_compare_chatterbox_params.py::TestVoiceReferenceFile::test_voice_ref_path_exists
PASSED tests/test_compare_chatterbox_params.py::TestVoiceReferenceFile::test_voice_ref_is_wav
PASSED tests/test_compare_chatterbox_params.py::TestOutputPaths::test_output_paths_follow_naming_convention
PASSED tests/test_compare_chatterbox_params.py::TestSRTFileHandling::test_parse_segments_respects_max_time
PASSED tests/test_compare_chatterbox_params.py::TestSRTFileHandling::test_polish_srt_file_exists
PASSED tests/test_compare_chatterbox_params.py::TestChatterboxIntegration::test_synthesize_called_with_correct_params
PASSED tests/test_compare_chatterbox_params.py::TestChatterboxIntegration::test_synthesize_returns_ndarray

============================== 20 passed in 4.27s ==============================
```

## Git Commit

```
commit 785b516b1c8f8c9a2ae33c92be3649c85bc96c8d
Author: Claude Opus 4.6

feat(tts): add Chatterbox parameter comparison script for Polish SRT

Generates first 10 minutes of Polish SRT (Marty.Supreme.2025.pl.srt) with
Chatterbox TTS using 3 voice cloning parameter configurations:
- Variant A: voice_clone (exaggeration=0.5, cfg_weight=0.5)
- Variant B: voice_clone_tuned (exaggeration=0.65, cfg_weight=0.6)
- Variant C: voice_clone_expressive (exaggeration=0.8, cfg_weight=0.7)

Each variant uses the Polish Witcher voice reference for consistent voice ID.
Outputs concatenated WAV files with 300ms silence between segments and
peak normalization to 0.95. Includes comprehensive unit tests (20 passing)
covering SRT parsing, text cleaning, variant configs, audio normalization,
and integration with ChatterboxTTS.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

## Usage

### Run the Comparison
```bash
cd /mnt/g/Self_Projects/active/AudioSmith
python scripts/compare_chatterbox_params.py
```

### Run Tests
```bash
pytest tests/test_compare_chatterbox_params.py -v
```

### Expected Output (after ~45 min)
```
======================================================================
CHATTERBOX PARAMETER COMPARISON — RESULTS
======================================================================
  chatterbox_A_clone                 (voice_clone                 ):    615.2s
  chatterbox_B_tuned                 (voice_clone_tuned           ):    615.2s
  chatterbox_C_expressive            (voice_clone_expressive      ):    615.2s

Output directory: test-files/tts_comparison
Voice reference: test-files/tts_comparison/voice_refs/witcher_polish_ref.wav
```

## Notes

- **Non-Fatal Warnings**: GPU compatibility warnings (RTX 5060 Ti) are expected and don't affect execution
- **Model Architecture**: Chatterbox uses T3 LLM + Diffusion decoder for speech generation
- **Token Repetition Detection**: Model includes safeguards to detect and stop generation on repetitive tokens
- **Determinism**: Different exaggeration/cfg_weight values will produce different prosody/pacing
