"""Tests for audiosmith.transcript_corrector module."""

import json
import pytest
from pathlib import Path
from audiosmith.transcript_corrector import TranscriptCorrector
from audiosmith.models import DubbingSegment


def _seg(text, index=0, start=0.0, end=1.0):
    return DubbingSegment(index=index, start_time=start, end_time=end, original_text=text)


class TestTranscriptCorrector:
    @pytest.fixture
    def corrector(self):
        return TranscriptCorrector()

    def test_correct_common_misspelling(self, corrector):
        result = corrector.correct([_seg("I recieve mail")])
        assert result[0].original_text == "I receive mail"

    def test_correct_informal_speech(self, corrector):
        result = corrector.correct([_seg("I am gonna do it")])
        assert "going to" in result[0].original_text

    def test_correct_preserves_case(self, corrector):
        r1 = corrector.correct([_seg("Gonna")])
        assert r1[0].original_text == "Going to"
        r2 = corrector.correct([_seg("GONNA")])
        assert r2[0].original_text == "GOING TO"

    def test_no_correction_needed(self, corrector):
        result = corrector.correct([_seg("Hello world")])
        assert result[0].original_text == "Hello world"

    def test_empty_text(self, corrector):
        result = corrector.correct([_seg("")])
        assert result[0].original_text == ""

    def test_context_aware_correction(self, corrector):
        corrector.add_correction("test", "exam", context_words=["school", "student"])
        segs = [
            _seg("I have a test tomorrow", 0, 0.0, 1.0),
            _seg("at school", 1, 1.0, 2.0),
            _seg("for physics", 2, 2.0, 3.0),
        ]
        result = corrector.correct(segs)
        assert "exam" in result[0].original_text

    def test_context_not_matching(self, corrector):
        corrector.add_correction("test", "exam", context_words=["school", "student"])
        segs = [
            _seg("I have a test tomorrow", 0, 0.0, 1.0),
            _seg("at home", 1, 1.0, 2.0),
            _seg("for fun", 2, 2.0, 3.0),
        ]
        result = corrector.correct(segs)
        assert "test" in result[0].original_text

    def test_add_correction(self, corrector):
        corrector.add_correction("foo", "bar")
        result = corrector.correct([_seg("foo fighters")])
        assert result[0].original_text == "bar fighters"

    def test_get_correction_count(self, corrector):
        assert corrector.get_correction_count() == 0
        corrector.correct([_seg("recieve gonna")])
        assert corrector.get_correction_count() == 2

    def test_domain_dict_loading(self, tmp_path):
        d = tmp_path / "dict.json"
        d.write_text(json.dumps({"colour": "color"}))
        c = TranscriptCorrector(domain_dict_path=d)
        result = c.correct([_seg("the colour red")])
        assert "color" in result[0].original_text

    def test_domain_dict_missing_file(self):
        result = TranscriptCorrector.load_domain_dict(Path("/nonexistent.json"))
        assert result == {}

    def test_phonetically_similar(self, corrector):
        assert corrector._is_phonetically_similar("recieve", "receive") is True

    def test_phonetically_dissimilar(self, corrector):
        assert corrector._is_phonetically_similar("cat", "elephant") is False
