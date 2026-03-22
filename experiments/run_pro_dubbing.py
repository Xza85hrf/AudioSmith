#!/usr/bin/env python3
"""Run professional dubbing pipeline for Marty Supreme with monitoring."""

import logging
import os
import sys
from pathlib import Path
from audiosmith.pipeline import DubbingPipeline
from audiosmith.models import DubbingConfig

# Load HuggingFace token for pyannote diarization
def load_hf_token():
    """Load HF token from cache or environment."""
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN']

    token_path = Path.home() / '.cache' / 'huggingface' / 'token'
    if token_path.exists():
        token = token_path.read_text().strip()
        os.environ['HF_TOKEN'] = token
        return token
    return None

load_hf_token()

# Set up logging to see all warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('dubbing_monitor')

# Track speedup warnings
speedup_warnings = []

class SpeedupWarningHandler(logging.Handler):
    def emit(self, record):
        if 'speedup' in record.getMessage().lower():
            speedup_warnings.append(record.getMessage())

handler = SpeedupWarningHandler()
logging.getLogger('audiosmith.mixer').addHandler(handler)

def main():
    logger.info("=" * 60)
    logger.info("PROFESSIONAL DUBBING PIPELINE - Marty Supreme")
    logger.info("=" * 60)

    config = DubbingConfig(
        video_path=Path("test-files/videos/Marty Supreme-enchaced.mp4"),
        output_dir=Path("test-files/videos/Marty_Supreme_pro"),
        source_language='en',
        target_language='pl',
        external_srt_path=Path("test-files/videos/original-transcriptions/Marty.Supreme.2025_pl.srt"),
        isolate_vocals=True,
        diarize=False,  # Disabled until pyannote terms accepted,
        detect_emotion=True,
        tts_engine='fish',
        
        
        max_speedup=1.3,
        allow_extended_timing=True,
    )

    logger.info("Configuration:")
    logger.info(f"  - External SRT: {config.external_srt_path}")
    logger.info(f"  - Max speedup: {config.max_speedup}x")
    logger.info(f"  - Extended timing: {config.allow_extended_timing}")
    logger.info(f"  - TTS engine: {config.tts_engine}")
    logger.info(f"  - Voice: {config.elevenlabs_voice_name}")
    logger.info(f"  - Detect emotion: {config.detect_emotion}")

    logger.info("=" * 60)
    logger.info("Starting pipeline...")
    logger.info("=" * 60)

    pipeline = DubbingPipeline(config)
    result = pipeline.run(config.video_path)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    if result.success:
        logger.info(f"✓ Output video: {result.output_video_path}")
        logger.info(f"✓ Dubbed audio: {result.dubbed_audio_path}")
        logger.info(f"✓ Total segments: {result.total_segments}")
        logger.info(f"✓ Segments dubbed: {result.segments_dubbed}")
        logger.info(f"✓ Total time: {result.total_time:.1f}s")

        if speedup_warnings:
            logger.warning(f"⚠ Speedup warnings: {len(speedup_warnings)}")
            for w in speedup_warnings[:5]:  # Show first 5
                logger.warning(f"  {w}")
        else:
            logger.info("✓ No excessive speedup warnings!")

        logger.info("")
        logger.info("Step times:")
        for step, time_s in result.step_times.items():
            logger.info(f"  {step}: {time_s:.1f}s")
    else:
        logger.error("✗ Pipeline failed!")
        for err in result.errors:
            logger.error(f"  {err}")

    return result

if __name__ == '__main__':
    result = main()
    sys.exit(0 if result.success else 1)