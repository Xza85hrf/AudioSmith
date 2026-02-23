"""URL download and format helpers using yt-dlp."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import yt_dlp

from audiosmith.srt import SRTEntry, seconds_to_timestamp, write_srt
from audiosmith.ffmpeg import extract_audio

logger = logging.getLogger(__name__)


def is_url(source: str) -> bool:
    """Return True if *source* looks like a URL."""
    return '://' in source


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def download_media(url: str, output_dir: Path) -> tuple[Path, str]:
    """Download media from url using yt-dlp. Returns (path, title)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'untitled')
        filename = ydl.prepare_filename(info)
        return Path(filename), title


def segments_to_txt(segments: List[Dict[str, Any]]) -> str:
    """Plain-text output — one segment per line."""
    return '\n'.join(seg['text'] for seg in segments)


def segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    """WebVTT output."""
    lines = ['WEBVTT', '']
    for i, seg in enumerate(segments, 1):
        start = seconds_to_timestamp(seg['start']).replace(',', '.')
        end = seconds_to_timestamp(seg['end']).replace(',', '.')
        lines.append(str(i))
        lines.append(f'{start} --> {end}')
        lines.append(seg['text'])
        lines.append('')
    return '\n'.join(lines)


def segments_to_json(segments: List[Dict[str, Any]]) -> str:
    """JSON array with start/end/text per segment."""
    data = [{'start': s['start'], 'end': s['end'], 'text': s['text']} for s in segments]
    return json.dumps(data, indent=2, ensure_ascii=False)
