"""Tests for audiosmith.document_formatter module."""

import pytest
from pathlib import Path
from audiosmith.document_formatter import DocumentFormatter, FormatterOptions
from audiosmith.models import DubbingSegment
from audiosmith.exceptions import DocumentFormattingError


def _seg(text, idx=0, start=0.0, end=5.0, speaker=None, translated=""):
    return DubbingSegment(
        index=idx, start_time=start, end_time=end,
        original_text=text, translated_text=translated, speaker_id=speaker,
    )


class TestFormatterOptions:
    def test_defaults(self):
        opts = FormatterOptions()
        assert opts.title is None
        assert opts.include_timestamps is True
        assert opts.include_speaker_labels is False
        assert opts.bilingual is False
        assert opts.font_size == 11


class TestDocumentFormatter:
    @pytest.fixture
    def formatter(self):
        return DocumentFormatter()

    @pytest.fixture
    def segments(self):
        return [_seg("Hello World", 0, 0.0, 5.0), _seg("Audio Smith", 1, 5.0, 10.5)]

    def test_format_timestamp(self, formatter):
        assert formatter._format_timestamp(65.5, 128.3) == "[1:05.5 - 2:08.3]"

    def test_to_txt_basic(self, formatter, segments, tmp_path):
        out = tmp_path / "out.txt"
        formatter.to_txt(segments, out)
        content = out.read_text()
        assert "Hello World" in content
        assert "Audio Smith" in content

    def test_to_txt_with_title(self, formatter, segments, tmp_path):
        out = tmp_path / "out.txt"
        formatter.to_txt(segments, out, FormatterOptions(title="My Title"))
        assert "My Title" in out.read_text()

    def test_to_txt_with_timestamps(self, formatter, segments, tmp_path):
        out = tmp_path / "out.txt"
        formatter.to_txt(segments, out)
        content = out.read_text()
        assert "[0:00.0 - 0:05.0]" in content

    def test_to_txt_with_speaker(self, formatter, tmp_path):
        segs = [_seg("Hello", speaker="Speaker_1")]
        out = tmp_path / "out.txt"
        formatter.to_txt(segs, out, FormatterOptions(include_speaker_labels=True))
        assert "[Speaker_1]" in out.read_text()

    def test_to_txt_bilingual(self, formatter, tmp_path):
        segs = [_seg("Hello", translated="Hola")]
        out = tmp_path / "out.txt"
        formatter.to_txt(segs, out, FormatterOptions(bilingual=True))
        content = out.read_text()
        assert "Hello" in content
        assert "Hola" in content

    def test_to_txt_returns_path(self, formatter, segments, tmp_path):
        out = tmp_path / "out.txt"
        assert formatter.to_txt(segments, out) == out

    def test_to_pdf_missing_fpdf(self, formatter, segments, tmp_path):
        import sys
        saved = sys.modules.get("fpdf")
        sys.modules["fpdf"] = None
        try:
            with pytest.raises(DocumentFormattingError):
                formatter.to_pdf(segments, tmp_path / "out.pdf")
        finally:
            if saved is not None:
                sys.modules["fpdf"] = saved
            else:
                sys.modules.pop("fpdf", None)

    def test_to_docx_missing_docx(self, formatter, segments, tmp_path):
        import sys
        saved = sys.modules.get("docx")
        sys.modules["docx"] = None
        try:
            with pytest.raises(DocumentFormattingError):
                formatter.to_docx(segments, tmp_path / "out.docx")
        finally:
            if saved is not None:
                sys.modules["docx"] = saved
            else:
                sys.modules.pop("docx", None)

    def test_to_txt_no_timestamps(self, formatter, segments, tmp_path):
        out = tmp_path / "out.txt"
        formatter.to_txt(segments, out, FormatterOptions(include_timestamps=False))
        content = out.read_text()
        assert "[0:" not in content
        assert "Hello World" in content
