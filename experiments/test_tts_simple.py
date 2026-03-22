#!/usr/bin/env python3
"""Simple TTS comparison - test each engine on first 20 SRT segments."""

import os
import sys
import time
from pathlib import Path

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('tts_test')

# Load tokens
def load_tokens():
    hf_token = Path.home() / '.cache' / 'huggingface' / 'token'
    if hf_token.exists():
        os.environ['HF_TOKEN'] = hf_token.read_text().strip()
    fish_key = os.environ.get('FISH_API_KEY')
    return {'fish': fish_key}

def test_engine(engine: str, segments: list, output_dir: Path) -> dict:
    """Test a TTS engine on segments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_dir = output_dir / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    success_count = 0
    total_duration = 0
    
    logger.info(f"[{engine}] Testing {len(segments)} segments...")
    
    try:
        if engine == 'piper':
            from audiosmith.piper_tts import PiperTTS
            tts = PiperTTS(voice='pl_PL')
            sample_rate = 22050
            
        elif engine == 'f5':
            from audiosmith.f5_tts import F5TTS
            tts = F5TTS(model_name='f5-polish')
            sample_rate = 24000
            
        elif engine == 'qwen3':
            from audiosmith.qwen3_tts import Qwen3TTS
            tts = Qwen3TTS()
            sample_rate = 24000
            
        elif engine == 'fish':
            from audiosmith.fish_speech_tts import FishSpeechTTS
            api_key = os.environ.get('FISH_API_KEY')
            if not api_key:
                return {'engine': engine, 'success': False, 'error': 'FISH_API_KEY not set'}
            tts = FishSpeechTTS(api_key=api_key)
            sample_rate = 44100
        else:
            return {'engine': engine, 'success': False, 'error': f'Unknown engine: {engine}'}
        
        import soundfile as sf
        
        for i, seg in enumerate(segments):
            text = seg['text']
            if not text.strip():
                continue
                
            out_path = engine_dir / f'seg_{i:04d}.wav'
            
            if engine == 'piper':
                audio = tts.synthesize(text)
            elif engine == 'f5':
                audio = tts.synthesize(text, language='pl')
            elif engine == 'qwen3':
                audio = tts.synthesize(text, voice='clone')
            elif engine == 'fish':
                audio, sr = tts.synthesize(text, language='pl')
                sample_rate = sr
            
            sf.write(str(out_path), audio, sample_rate)
            success_count += 1
            total_duration += seg['end'] - seg['start']
            
            if (i + 1) % 5 == 0:
                logger.info(f"[{engine}] Progress: {i+1}/{len(segments)}")
        
        elapsed = time.time() - start_time
        logger.info(f"[{engine}] Complete: {success_count} segments in {elapsed:.1f}s")
        
        return {
            'engine': engine,
            'success': True,
            'segments': success_count,
            'time': elapsed,
            'duration': total_duration,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{engine}] Failed: {e}")
        return {'engine': engine, 'success': False, 'error': str(e), 'time': elapsed}


def load_srt(srt_path: Path, limit: int = 20) -> list:
    """Load first N segments from SRT."""
    from audiosmith.srt import parse_srt_file, timestamp_to_seconds
    
    entries = parse_srt_file(srt_path)
    segments = []
    
    for entry in entries[:limit]:
        text = entry.text.strip()
        # Skip non-speech
        if text.startswith('[') and text.endswith(']'):
            continue
        if not text:
            continue
        
        start = timestamp_to_seconds(entry.start_time)
        end = timestamp_to_seconds(entry.end_time)
        
        if end - start < 0.3:
            continue
            
        segments.append({
            'start': start,
            'end': end,
            'text': text,
        })
    
    return segments


def main():
    load_tokens()
    
    srt_path = Path("test-files/videos/Marty_Supreme_archive/original-transcriptions/Marty.Supreme.2025_pl.srt")
    output_dir = Path("test-files/videos/Marty_Supreme_samples")
    
    if not srt_path.exists():
        logger.error(f"SRT not found: {srt_path}")
        sys.exit(1)
    
    # Load first 20 segments (about 1-2 minutes of audio)
    segments = load_srt(srt_path, limit=20)
    logger.info(f"Loaded {len(segments)} segments from SRT")
    logger.info(f"Duration: {sum(s['end']-s['start'] for s in segments):.1f} seconds")
    
    # Test engines
    engines = ['piper', 'f5', 'qwen3', 'fish']
    results = []
    
    for engine in engines:
        result = test_engine(engine, segments, output_dir)
        results.append(result)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    for r in results:
        if r['success']:
            logger.info(f"OK {r['engine']}: {r['segments']} segments, {r['time']:.1f}s")
        else:
            logger.info(f"FAIL {r['engine']}: {r.get('error', 'Unknown')}")
    
    logger.info(f"\nSamples saved to: {output_dir}")
    logger.info("Listen to each sample to compare quality.")


if __name__ == '__main__':
    main()
