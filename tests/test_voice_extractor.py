"""Tests for audiosmith.voice_extractor module."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from audiosmith.voice_extractor import (
    VoiceSample, VoiceCatalog, VoiceExtractor, create_voice_profiles,
)


class TestVoiceSample:
    def test_defaults(self):
        s = VoiceSample("spk1", Path("a.wav"), Path("src.mp3"), 10.0, 15.0)
        assert s.speaker_id == "spk1"
        assert s.mean_volume_db == 0.0
        assert s.description == ""
        assert s.language == "auto"

    def test_duration(self):
        s = VoiceSample("spk1", Path("a.wav"), Path("src.mp3"), 10.0, 15.0)
        assert s.duration == 5.0


class TestVoiceCatalog:
    def test_empty(self):
        c = VoiceCatalog()
        assert c.samples == []
        assert c.source_files == []

    def test_add_sample(self):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("src.mp3"), 0, 5))
        assert len(c.samples) == 1
        assert "src.mp3" in c.source_files

    def test_add_sample_tracks_source_once(self):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("src.mp3"), 0, 5))
        c.add_sample(VoiceSample("spk2", Path("b.wav"), Path("src.mp3"), 5, 10))
        assert len(c.source_files) == 1

    def test_get_speakers(self):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("s.mp3"), 0, 5))
        c.add_sample(VoiceSample("spk2", Path("b.wav"), Path("s.mp3"), 5, 10))
        speakers = c.get_speakers()
        assert set(speakers) == {"spk1", "spk2"}

    def test_get_speakers_unique(self):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("s.mp3"), 0, 5))
        c.add_sample(VoiceSample("spk1", Path("b.wav"), Path("s.mp3"), 5, 10))
        assert len(c.get_speakers()) == 1

    def test_get_samples_for_speaker(self):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("s.mp3"), 0, 5))
        c.add_sample(VoiceSample("spk1", Path("b.wav"), Path("s.mp3"), 5, 10))
        c.add_sample(VoiceSample("spk2", Path("c.wav"), Path("s.mp3"), 10, 15))
        assert len(c.get_samples_for_speaker("spk1")) == 2
        assert len(c.get_samples_for_speaker("spk2")) == 1

    def test_get_best_sample(self):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("s.mp3"), 0, 5, mean_volume_db=-15.0))
        c.add_sample(VoiceSample("spk1", Path("b.wav"), Path("s.mp3"), 5, 10, mean_volume_db=-10.0))
        best = c.get_best_sample("spk1")
        assert best.mean_volume_db == -10.0

    def test_get_best_sample_excludes_clipping(self):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("s.mp3"), 0, 5, mean_volume_db=-2.0))
        c.add_sample(VoiceSample("spk1", Path("b.wav"), Path("s.mp3"), 5, 10, mean_volume_db=-15.0))
        best = c.get_best_sample("spk1")
        assert best.mean_volume_db == -15.0

    def test_get_best_sample_unknown(self):
        c = VoiceCatalog()
        assert c.get_best_sample("unknown") is None

    def test_save_and_load(self, tmp_path):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("s.mp3"), 0, 5, mean_volume_db=-10.0))
        c.add_sample(VoiceSample("spk2", Path("b.wav"), Path("s.mp3"), 5, 10, mean_volume_db=-15.0))
        p = tmp_path / "catalog.json"
        c.save(p)

        loaded = VoiceCatalog.load(p)
        assert len(loaded.samples) == 2
        assert loaded.samples[0].speaker_id == "spk1"
        assert loaded.samples[1].speaker_id == "spk2"
        assert loaded.samples[0].sample_path == Path("a.wav")
        assert isinstance(loaded.samples[0].created_at, float)


class TestVoiceExtractor:
    def test_init(self, tmp_path):
        ve = VoiceExtractor(tmp_path)
        assert ve.output_dir == tmp_path
        assert ve.sample_duration == 5.0
        assert ve.sample_rate == 24000
        assert ve.min_volume_db == -30.0

    def test_extract_segment(self, tmp_path):
        ve = VoiceExtractor(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = ve._extract_segment(Path("in.mp3"), 10.0, 15.0, "sample_01")
        assert result == tmp_path / "sample_01.wav"
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert str(Path("in.mp3")) in " ".join(args)

    def test_measure_volume(self, tmp_path):
        ve = VoiceExtractor(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stderr="[volumedetect] mean_volume: -18.5 dB\n[volumedetect] max_volume: -2.0 dB"
            )
            assert ve._measure_volume(Path("test.wav")) == -18.5

    def test_measure_volume_fallback(self, tmp_path):
        ve = VoiceExtractor(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stderr="no volume info")
            assert ve._measure_volume(Path("test.wav")) == -60.0

    def test_get_duration(self, tmp_path):
        ve = VoiceExtractor(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="125.5\n")
            assert ve._get_duration(Path("test.wav")) == 125.5

    def test_extract_evenly(self, tmp_path):
        ve = VoiceExtractor(tmp_path)
        with patch.object(ve, "_get_duration", return_value=50.0), \
             patch.object(ve, "_extract_segment", return_value=tmp_path / "s.wav"), \
             patch.object(ve, "_measure_volume", return_value=-15.0):
            cat = ve.extract_evenly(Path("a.mp3"), num_samples=3)
        assert len(cat.samples) == 3
        assert all(s.speaker_id.startswith("voice_") for s in cat.samples)

    def test_extract_at_intervals(self, tmp_path):
        ve = VoiceExtractor(tmp_path)
        with patch.object(ve, "_extract_segment", return_value=tmp_path / "s.wav"), \
             patch.object(ve, "_measure_volume", return_value=-15.0):
            cat = ve.extract_at_intervals(
                Path("a.mp3"), [(0, 5), (10, 15)], speaker_ids=["narrator", "character"],
            )
        assert len(cat.samples) == 2
        assert cat.samples[0].speaker_id == "narrator"
        assert cat.samples[1].speaker_id == "character"

    def test_extract_at_intervals_auto_names(self, tmp_path):
        ve = VoiceExtractor(tmp_path)
        with patch.object(ve, "_extract_segment", return_value=tmp_path / "s.wav"), \
             patch.object(ve, "_measure_volume", return_value=-15.0):
            cat = ve.extract_at_intervals(Path("a.mp3"), [(0, 5), (10, 15)])
        assert cat.samples[0].speaker_id == "speaker_00"
        assert cat.samples[1].speaker_id == "speaker_01"


class TestCreateVoiceProfiles:
    def test_creates_profiles(self):
        c = VoiceCatalog()
        c.add_sample(VoiceSample("spk1", Path("a.wav"), Path("s.mp3"), 0, 5, mean_volume_db=-10.0))
        c.add_sample(VoiceSample("spk2", Path("b.wav"), Path("s.mp3"), 5, 10, mean_volume_db=-15.0))
        profiles = create_voice_profiles(c)
        assert len(profiles) == 2
        assert "spk1" in profiles
        assert "sample_path" in profiles["spk1"]

    def test_empty_catalog(self):
        assert create_voice_profiles(VoiceCatalog()) == {}
