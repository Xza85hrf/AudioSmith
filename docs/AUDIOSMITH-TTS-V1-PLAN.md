# AudioSmith-TTS-v1: Multi-Teacher Polish Fine-Tuning Pipeline

## Context

Chatterbox won the 5-engine Polish TTS comparison (variant B_tuned with Witcher voice clone). To push quality further, we'll fine-tune Chatterbox's T3 language model (LLaMA-520M) on 24h of professional Polish narration (Witcher audiobook) + best outputs from all 5 TTS engines. Only the T3 text→speech-token stage is trainable — S3Gen decoder and vocoder stay frozen.

**Goal:** Create a LoRA adapter that makes Chatterbox natively excellent at Polish, not just "acceptable with voice cloning."

---

## Pipeline Overview

```
Witcher Audiobook (281 MP3s, 6.8GB, ~24h Polish)
    ↓ Stage 1: Transcribe + chunk + speaker embeddings
~1500 segments (5-15s each with text + speaker conditioning)
    ↓ Stage 2: Generate same text with all 5 TTS engines, score, pick best
Best teacher output per segment + scoring metadata
    ↓ Stage 3: Extract speech tokens from real audio + best teachers
(text, conditioning, target_speech_tokens) training tuples
    ↓ Stage 4: LoRA fine-tune T3 on mixed real + synthetic data
LoRA adapter weights (~2MB)
    ↓ Stage 5: Evaluate vs base Chatterbox on Marty SRT
Comparison WAVs + metrics report
```

---

## Stage 1: Data Preparation (~2h)

**What:** Transcribe all Witcher audiobook files, chunk into training segments.

**Input:** `test-files/audio/Wiedzmin-audiobook/` (281 MP3s, 6.8GB)

**Steps:**
1. Convert each MP3 to 16kHz mono WAV
2. Transcribe with faster-whisper large-v3 (word-level timestamps enabled)
3. Chunk at sentence boundaries into 5-15s segments
4. Extract 256-dim speaker embedding from each chunk using Chatterbox's frozen VoiceEncoder
5. Save dataset manifest

**Output:** `data/witcher_dataset/segments.jsonl`
```json
{"segment_id": "witcher_001_00", "audio_path": "...", "start_ms": 0, "end_ms": 8500, "text": "Geralt odwrócił się...", "speaker_embedding": [256 floats], "language": "pl"}
```

**Code to create:** `audiosmith/training/data_prep.py`
- Reuse: `audiosmith/transcribe.py` (Transcriber class)
- Reuse: Chatterbox's `VoiceEncoder` for speaker embeddings

**Run:**
```bash
audiosmith train prepare \
  --input-dir test-files/audio/Wiedzmin-audiobook \
  --output-dir data/witcher_dataset \
  --language pl --whisper-model large-v3
```

---

## Stage 2: Teacher Generation (~5h)

**What:** For each text segment, generate TTS with all 5 engines. Score and pick the best.

**Input:** `data/witcher_dataset/segments.jsonl`

**Steps:**
1. Load TTSModelManager (hot-swaps engines, one in VRAM at a time)
2. For each of ~1500 segments, synthesize with: Chatterbox, Fish Speech, Qwen3, F5-TTS, Piper
3. Score each output:
   - Duration accuracy: `1 - abs(generated_ms - expected_ms) / expected_ms`
   - Silence ratio: % of audio below noise floor
   - Word count ratio: generated vs original
4. Select best teacher per segment
5. Save audio files + scoring metadata

**Output:** `data/witcher_dataset/teachers.jsonl` + `data/witcher_dataset/teacher_audio/`
```json
{"segment_id": "witcher_001_00", "best_teacher": "fish", "best_score": 0.94, "teachers": {"chatterbox": {"path": "...", "score": 0.91}, "fish": {"path": "...", "score": 0.94}, ...}}
```

**Code to create:** `audiosmith/training/teacher_gen.py`
- Reuse: `audiosmith/tts_manager.py` (TTSModelManager)
- Reuse: All 5 TTS engine wrappers

**Run:**
```bash
audiosmith train generate-teachers \
  --dataset data/witcher_dataset/segments.jsonl \
  --output-dir data/witcher_dataset \
  --teachers chatterbox fish qwen3 f5 piper
```

---

## Stage 3: Speech Token Extraction (~1h)

**What:** Convert audio to speech tokens using Chatterbox's frozen S3Tokenizer (25Hz).

**Input:** Real Witcher audio + best teacher outputs from Stage 2

**Steps:**
1. Load S3Tokenizer from Chatterbox checkpoint (`s3gen.pt`)
2. Tokenize real Witcher audio chunks → target speech tokens
3. Tokenize best teacher audio → synthetic target speech tokens
4. Package as training tuples: (text_tokens, speaker_embedding, language_id, target_speech_tokens)

**Output:** `data/witcher_dataset/tokens.jsonl`
```json
{"segment_id": "witcher_001_00", "audio_type": "real", "text": "Geralt...", "tokens": [123, 456, 789, ...], "tokens_len": 212, "speaker_embedding": [...]}
```

**Code to create:** `audiosmith/training/token_extractor.py`
- Reuse: Chatterbox's `S3Tokenizer` from `chatterbox.models.s3gen.tokenizer`

**Run:**
```bash
audiosmith train extract-tokens \
  --dataset data/witcher_dataset/segments.jsonl \
  --teacher-outputs data/witcher_dataset/teachers.jsonl \
  --output data/witcher_dataset/tokens.jsonl
```

---

## Stage 4: LoRA Fine-Tuning (~6-8h overnight)

**What:** Apply LoRA to Chatterbox T3's LLaMA transformer layers. Train on mixed real + synthetic data.

**Input:** `data/witcher_dataset/tokens.jsonl`

**Architecture:**
- **Base model:** Chatterbox T3 (LLaMA-520M, 30 layers, 1024 hidden, 16 heads)
- **LoRA config:**
  - Rank: 16
  - Alpha: 32
  - Target modules: `q_proj, v_proj, gate_proj`
  - Dropout: 0.05
  - Trainable params: ~2-5M (1-2% of T3)
- **Loss:** Cross-entropy on speech token prediction
- **Data mix:** 70% real Witcher audio tokens, 30% best teacher tokens
- **Curriculum:**
  - Epoch 1: Long segments (>10s) + high-scoring teachers only
  - Epoch 2: All segments
  - Epoch 3: Focus on short/edge-case segments

**Hyperparameters:**
- Epochs: 3
- Batch size: 4
- Gradient accumulation: 4 (effective batch 16)
- Learning rate: 2e-4
- Warmup: 500 steps
- Max grad norm: 1.0

**VRAM budget (RTX 3090 Ti, 24GB):**
- T3 model: ~6GB
- LoRA adapter: ~0.4GB
- Batch (size=4, seq_len=512): ~10GB
- Optimizer + gradients: ~4GB
- **Total: ~20.4GB** (fits with margin)

**Output:** `models/chatterbox_polish_lora/`
```
adapter_config.json         # LoRA config
adapter_model.safetensors   # Adapter weights (~2MB)
training_log.json           # Loss curves
```

**Code to create:** `audiosmith/training/lora_trainer.py`
- Uses: `peft` (LoraConfig, get_peft_model) — already installed v0.18.1
- Uses: HuggingFace `Trainer` or custom training loop

**Run:**
```bash
audiosmith train finetune \
  --tokens data/witcher_dataset/tokens.jsonl \
  --output models/chatterbox_polish_lora \
  --lora-rank 16 --epochs 3 --batch-size 4 --lr 2e-4 \
  --device cuda:0
```

---

## Stage 5: Evaluation (~1h)

**What:** Generate test audio with base, variant B, and fine-tuned model. Compare.

**Input:** `test-files/videos/Original_subtitiles/Marty.Supreme.2025.pl.srt` (first 10 min)

**Steps:**
1. Generate with base Chatterbox (no voice clone)
2. Generate with variant B (Witcher voice clone, exag=0.65, cfg=0.6)
3. Generate with LoRA-v1 (fine-tuned + Witcher voice clone)
4. Compute metrics: speaker similarity, duration accuracy, MCD
5. Output comparison WAVs for listening test

**Output:** `data/evaluation/`
```
chatterbox_base.wav
chatterbox_variant_b.wav
chatterbox_lora_v1.wav
evaluation_metrics.json
```

**Code to create:** `audiosmith/training/evaluator.py`

**Run:**
```bash
audiosmith train evaluate \
  --adapter models/chatterbox_polish_lora \
  --test-srt test-files/videos/Original_subtitiles/Marty.Supreme.2025.pl.srt \
  --output data/evaluation
```

---

## Full Overnight Run

```bash
# All stages sequentially
audiosmith train run-all \
  --input-dir test-files/audio/Wiedzmin-audiobook \
  --output-dir data/witcher_dataset \
  --model-output models/chatterbox_polish_lora \
  --device cuda:0
```

---

## Using the Fine-Tuned Model

### CLI
```bash
audiosmith dub video.mp4 -t pl --engine chatterbox \
  --audio-prompt test-files/tts_comparison/voice_refs/witcher_polish_ref.wav \
  --lora-adapter models/chatterbox_polish_lora
```

### Python
```python
from audiosmith.tts import ChatterboxTTS

engine = ChatterboxTTS(device='cuda')
engine.load_model()
engine.load_lora_adapter('models/chatterbox_polish_lora')
audio = engine.synthesize("Cześć, jak się masz?", language='pl',
                          audio_prompt_path='voice_refs/witcher_polish_ref.wav',
                          exaggeration=0.65, cfg_weight=0.6)
```

---

## Files to Create

```
audiosmith/training/
  __init__.py
  data_prep.py            # Stage 1
  teacher_gen.py          # Stage 2
  token_extractor.py      # Stage 3
  lora_trainer.py         # Stage 4
  evaluator.py            # Stage 5
  config.py               # TrainingConfig dataclass
audiosmith/commands/
  train_cmd.py            # CLI: audiosmith train <stage>
```

## Files to Modify

- `audiosmith/cli.py` — register `train` command group
- `audiosmith/tts.py` — add `load_lora_adapter(path)` method
- `pyproject.toml` — add `training` optional deps (peft, datasets)

## Dependencies to Install (when ready)

```bash
uv pip install datasets pesq
# peft already installed (0.18.1)
```

---

## Resource Summary

| Stage | VRAM | Time | Disk |
|-------|------|------|------|
| 1 Data Prep | 4GB | 2h | 2GB |
| 2 Teachers | 10GB | 5h | 40GB |
| 3 Tokens | 6GB | 1h | 200MB |
| 4 LoRA | 20GB | 6-8h | 2MB |
| 5 Eval | 8GB | 1h | 500MB |
| **Total** | **20GB max** | **15-17h** | **~43GB** |

---

## Success Criteria

1. ~1500 valid training segments from Witcher
2. Training loss decreases monotonically across 3 epochs
3. LoRA model produces more natural Polish than base Chatterbox
4. No regressions in existing AudioSmith tests
5. Adapter weights < 5MB (lightweight, shareable)
