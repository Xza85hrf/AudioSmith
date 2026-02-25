"""Batch processing of multiple audio/video files."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from audiosmith.exceptions import BatchProcessingError
from audiosmith.models import DubbingConfig, DubbingResult

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of processing a single file in a batch."""
    file_path: Path
    success: bool
    result: Optional[DubbingResult] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class BatchProcessor:
    """Process multiple audio/video files with shared config."""

    def process(
        self,
        file_paths: List[Path],
        config: DubbingConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        continue_on_error: bool = True,
    ) -> List[BatchResult]:
        """Process batch of files sequentially.

        Args:
            file_paths: Input files to process.
            config: Shared DubbingConfig (output_dir adjusted per file).
            progress_callback: Called with (current_idx, total, filename).
            continue_on_error: If False, stop on first failure.
        """
        from audiosmith.pipeline import DubbingPipeline

        results: List[BatchResult] = []
        total = len(file_paths)

        for idx, fpath in enumerate(file_paths):
            logger.info("Processing [%d/%d]: %s", idx + 1, total, fpath.name)
            if progress_callback:
                progress_callback(idx, total, fpath.name)

            t0 = time.time()
            batch_result = BatchResult(file_path=fpath, success=False)

            try:
                per_file_config = self._adjust_config(config, fpath)
                pipeline = DubbingPipeline(per_file_config)
                dubbing_result = pipeline.run(fpath)
                batch_result.success = True
                batch_result.result = dubbing_result
            except Exception as e:
                batch_result.error = f"{type(e).__name__}: {e}"
                logger.error("Failed %s: %s", fpath.name, batch_result.error)
                if not continue_on_error:
                    batch_result.duration_seconds = time.time() - t0
                    results.append(batch_result)
                    raise BatchProcessingError(
                        f"Batch stopped at {fpath.name}: {e}"
                    ) from e
            finally:
                batch_result.duration_seconds = time.time() - t0
                if batch_result not in results:
                    results.append(batch_result)

        if progress_callback:
            progress_callback(total, total, "Complete")
        return results

    @staticmethod
    def _adjust_config(config: DubbingConfig, file_path: Path) -> DubbingConfig:
        """Create per-file config with dedicated output directory."""
        return DubbingConfig(
            video_path=file_path,
            output_dir=Path(config.output_dir) / file_path.stem,
            source_language=config.source_language,
            target_language=config.target_language,
            whisper_model=config.whisper_model,
            whisper_compute_type=config.whisper_compute_type,
            whisper_device=config.whisper_device,
            audio_prompt_path=config.audio_prompt_path,
            chatterbox_exaggeration=config.chatterbox_exaggeration,
            chatterbox_cfg_weight=config.chatterbox_cfg_weight,
            burn_subtitles=config.burn_subtitles,
            isolate_vocals=config.isolate_vocals,
            diarize=config.diarize,
            detect_emotion=config.detect_emotion,
            post_process=config.post_process,
        )

    @staticmethod
    def get_summary(results: List[BatchResult]) -> Dict[str, Any]:
        """Generate summary statistics from batch results."""
        total = len(results)
        succeeded = sum(1 for r in results if r.success)
        total_dur = sum(r.duration_seconds for r in results)
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "total_duration_seconds": round(total_dur, 2),
            "avg_duration_seconds": round(total_dur / total, 2) if total else 0.0,
            "failed_files": [str(r.file_path) for r in results if not r.success],
        }
