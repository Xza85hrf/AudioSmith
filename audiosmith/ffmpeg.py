"""FFmpeg-based audio extraction, duration probing, and video encoding."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from audiosmith.exceptions import DubbingError
from audiosmith.error_codes import ErrorCode

logger = logging.getLogger(__name__)


def probe_duration(video_path: Path) -> float:
    """Probe video/audio file duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, TypeError) as e:
        raise DubbingError(
            f"Failed to probe duration for {video_path}: {e}",
            error_code=str(ErrorCode.DUBBING_EXTRACTION_ERROR.value),
            original_error=e,
        )


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    max_duration: Optional[float] = None,
) -> Path:
    """Extract audio from video file as PCM WAV."""
    cmd = ['ffmpeg', '-i', str(video_path)]
    if max_duration is not None:
        cmd.extend(['-t', str(max_duration)])
    cmd.extend([
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate), '-ac', str(channels),
        '-y', str(output_path),
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode().strip() if e.stderr else "Unknown error"
        raise DubbingError(
            f"Failed to extract audio from {video_path}: {stderr_msg}",
            error_code=str(ErrorCode.DUBBING_EXTRACTION_ERROR.value),
            original_error=e,
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Extracted audio: %s (%.1f MB)", output_path, size_mb)
    return output_path


def encode_video(
    video_path: Path,
    dubbed_audio_path: Path,
    output_path: Path,
    subtitle_path: Optional[Path] = None,
    max_duration: Optional[float] = None,
    source_language: str = 'en',
    target_language: str = 'pl',
) -> Path:
    """Encode final video with original + dubbed audio tracks and optional burned subtitles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ['ffmpeg', '-i', str(video_path), '-i', str(dubbed_audio_path)]
    if max_duration is not None:
        cmd.extend(['-t', str(max_duration)])

    # Subtitle burning
    subs_copy = None
    if subtitle_path is not None and subtitle_path.exists():
        subs_copy = output_path.parent / 'dubbed_subs.srt'
        if subtitle_path.resolve() != subs_copy.resolve():
            shutil.copy2(subtitle_path, subs_copy)
        else:
            subs_copy = subtitle_path
        escaped = subs_copy.name.replace("'", r"\'").replace(":", r"\:")
        vf = (
            f"subtitles={escaped}"
            ":force_style='FontSize=22,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,Outline=2'"
        )
        cmd.extend(['-vf', vf, '-c:v', 'libx264', '-preset', 'medium', '-crf', '20'])
    else:
        cmd.extend(['-c:v', 'copy'])

    cmd.extend([
        '-map', '0:v:0', '-map', '0:a:0', '-map', '1:a:0',
        '-c:a', 'aac', '-b:a', '192k',
        f'-metadata:s:a:0', f'language={source_language}',
        f'-metadata:s:a:0', f'title=Original {source_language.upper()}',
        f'-metadata:s:a:1', f'language={target_language}',
        f'-metadata:s:a:1', f'title=Dubbed {target_language.upper()}',
        '-y', str(output_path),
    ])

    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=str(output_path.parent))
    except subprocess.CalledProcessError as e:
        if subs_copy is not None:
            logger.warning("Subtitle burning failed, retrying without subtitles")
            return _encode_without_subtitles(
                video_path, dubbed_audio_path, output_path,
                max_duration, source_language, target_language,
            )
        stderr_msg = e.stderr.decode() if e.stderr else "Unknown"
        raise DubbingError(
            f"Video encoding failed: {stderr_msg[:500]}",
            error_code=str(ErrorCode.DUBBING_ENCODE_ERROR.value),
            original_error=e,
        )

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 ** 2)
        logger.info("Encoded video: %s (%.1f MB)", output_path, size_mb)
    return output_path


def _encode_without_subtitles(
    video_path: Path,
    dubbed_audio_path: Path,
    output_path: Path,
    max_duration: Optional[float],
    source_language: str,
    target_language: str,
) -> Path:
    """Fallback: encode without subtitle burning."""
    cmd = ['ffmpeg', '-i', str(video_path), '-i', str(dubbed_audio_path)]
    if max_duration is not None:
        cmd.extend(['-t', str(max_duration)])
    cmd.extend([
        '-c:v', 'copy',
        '-map', '0:v:0', '-map', '0:a:0', '-map', '1:a:0',
        '-c:a', 'aac', '-b:a', '192k',
        f'-metadata:s:a:0', f'language={source_language}',
        f'-metadata:s:a:1', f'language={target_language}',
        '-y', str(output_path),
    ])
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode() if e.stderr else "Unknown"
        raise DubbingError(
            f"Video encoding failed (no subtitles): {stderr_msg[:500]}",
            error_code=str(ErrorCode.DUBBING_ENCODE_ERROR.value),
            original_error=e,
        )
    return output_path
