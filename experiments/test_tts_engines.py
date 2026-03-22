#!/usr/bin/env python3
"""Test multiple TTS engines on first 10 minutes - produce audio samples for comparison."""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from audiosmith.pipeline import DubbingPipeline
from audiosmith.models import DubbingConfig
from audiosmith.ffmpeg import probe_duration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('tts_test')

# Load HF token
def load_hf_token():
    if os.environ.get('HF_TOKEN'):
        return
    token_path = Path.home() / '.cache' / 'huggingface' / 'token'
    if token_path.exists():
        os.environ['HF_TOKEN'] = token_path.read_text().strip()

load_hf_token()

# TTS engines to test (local + Fish Speech)
TTS_ENGINES = [
    ('piper', 'Piper (local, fast)'),
    ('f5', 'F5-TTS (local, Polish fine-tuned)'),
    ('qwen3', 'Qwen3-TTS (local, voice cloning)'),
    ('fish', 'Fish Speech (cloud API)'),
]

def run_tts_test(engine: str, name: str, config_base: DubbingConfig, output_dir: Path) -> dict:
    """Run TTS test for a single engine."""
    engine_dir = output_dir / f'tts_{engine}'
    engine_dir.mkdir(parents=True, exist_ok=True)
    
    config = DubbingConfig(
        video_path=config_base.video_path,
        output_dir=engine_dir,
        source_language=config_base.source_language,
        target_language=config_base.target_language,
        external_srt_path=config_base.external_srt_path,
        use_srt_timing=True,
        max_duration=600,  # 10 minutes
        tts_engine=engine,
        isolate_vocals=False,  # Skip for speed
        diarize=False,
        detect_emotion=False,
        burn_subtitles=False,  # Audio only
        post_process_tts=True,
    )
    
    # Set engine-specific options
    if engine == 'f5':
        config.f5_model = 'f5-polish'
    elif engine == 'qwen3':
        config.qwen3_voice = 'clone'
    
    logger.info(f"[{name}] Starting...")
    start_time = time.time()
    
    try:
        pipeline = DubbingPipeline(config)
        result = pipeline.run(config.video_path)
        elapsed = time.time() - start_time
        
        if result.success:
            # Copy dubbed audio to main output dir for easy comparison
            dubbed_audio = engine_dir / 'dubbed_audio.wav'
            if dubbed_audio.exists():
                sample_audio = output_dir / f'sample_{engine}.wav'
                import shutil
                shutil.copy(dubbed_audio, sample_audio)
                logger.info(f"[{name}] Complete in {elapsed:.1f}s - {sample_audio.name}")
                return {'engine': engine, 'name': name, 'success': True, 'time': elapsed, 'audio': str(sample_audio)}
            else:
                logger.error(f"[{name}] No audio produced")
                return {'engine': engine, 'name': name, 'success': False, 'time': elapsed, 'error': 'No audio'}
        else:
            logger.error(f"[{name}] Failed: {result.errors}")
            return {'engine': engine, 'name': name, 'success': False, 'time': elapsed, 'error': str(result.errors)}
            
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{name}] Exception: {e}")
        return {'engine': engine, 'name': name, 'success': False, 'time': elapsed, 'error': str(e)}


def main():
    video_path = Path("test-files/videos/Marty Supreme-enchaced.mp4")
    srt_path = Path("test-files/videos/Marty_Supreme_archive/original-transcriptions/Marty.Supreme.2025_pl.srt")
    output_dir = Path("test-files/videos/Marty_Supreme_samples")
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    if not srt_path.exists():
        logger.error(f"SRT not found: {srt_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TTS ENGINE COMPARISON - First 10 Minutes")
    logger.info("=" * 60)
    logger.info(f"Video: {video_path}")
    logger.info(f"SRT: {srt_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)
    
    # Get video duration
    duration = probe_duration(video_path)
    logger.info(f"Video duration: {duration/60:.1f} minutes")
    logger.info("Testing first 10 minutes...")
    
    config_base = DubbingConfig(
        video_path=video_path,
        output_dir=output_dir,
        source_language='en',
        target_language='pl',
        external_srt_path=srt_path,
        use_srt_timing=True,
        max_duration=600,  # 10 minutes
    )
    
    results = []
    
    # Run engines in parallel (up to 2 at a time to avoid memory issues)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_tts_test, engine, name, config_base, output_dir): (engine, name)
            for engine, name in TTS_ENGINES
        }
        
        for future in as_completed(futures):
            engine, name = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"[{name}] Exception: {e}")
                results.append({'engine': engine, 'name': name, 'success': False, 'error': str(e)})
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    for r in sorted(results, key=lambda x: x.get('time', 999)):
        if r['success']:
            logger.info(f"OK {r['name']}: {r['time']:.1f}s - {Path(r['audio']).name}")
        else:
            logger.info(f"FAIL {r['name']}: Failed - {r.get('error', 'Unknown')}")
    
    logger.info("")
    logger.info(f"Audio samples saved to: {output_dir}")
    logger.info("Listen to each sample and choose the best engine for full processing.")
    
    return results


if __name__ == '__main__':
    main()
