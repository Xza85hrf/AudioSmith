"""Shared test fixtures for AudioSmith."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temporary output directory."""
    out = tmp_path / 'output'
    out.mkdir()
    return out


@pytest.fixture
def sample_srt_file(tmp_path):
    """Create a sample SRT file for testing."""
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,500\n"
        "Hello, world!\n"
        "\n"
        "2\n"
        "00:00:04,000 --> 00:00:06,000\n"
        "This is a test.\n"
        "\n"
    )
    path = tmp_path / 'test.srt'
    path.write_text(content, encoding='utf-8')
    return path


@pytest.fixture
def sample_segments():
    """Return sample transcription segments."""
    return [
        {'text': 'Hello world', 'start': 0.0, 'end': 2.5},
        {'text': 'This is a test', 'start': 3.0, 'end': 5.0},
        {'text': 'Goodbye', 'start': 5.5, 'end': 7.0},
    ]
