"""AudioSmith SRT subtitle parsing and writing utilities."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SRTEntry:
    """Represents a single SRT subtitle entry."""
    index: int
    start_time: str
    end_time: str
    text: str

    def to_srt(self) -> str:
        """Convert the SRT entry to its string representation."""
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.text}\n"


def parse_srt(content: str) -> List[SRTEntry]:
    """Parse SRT content string into a list of SRTEntry objects."""
    entries = []
    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0].strip())
                time_match = re.match(
                    r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})',
                    lines[1].strip()
                )
                if time_match:
                    text = '\n'.join(lines[2:])
                    entries.append(SRTEntry(
                        index=index,
                        start_time=time_match.group(1),
                        end_time=time_match.group(2),
                        text=text
                    ))
            except (ValueError, IndexError):
                continue
    return entries


def parse_srt_file(path: Path) -> List[SRTEntry]:
    """Parse an SRT file and return a list of SRTEntry objects."""
    content = path.read_text(encoding='utf-8')
    return parse_srt(content)


def write_srt(entries: List[SRTEntry], path: Path) -> None:
    """Write SRT entries to a file, re-indexing sequentially."""
    with open(path, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(entries, 1):
            entry.index = i
            f.write(entry.to_srt() + '\n')


def timestamp_to_seconds(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.replace(',', '.')
    parts = ts.split(':')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def seconds_to_timestamp(s: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace('.', ',')
