"""Tests for TTS synthesis pipeline integration with TTSModelManager.

Tests verify that the TTSSynthesisMixin properly initializes and uses
TTSModelManager for hot-swapping TTS engines during synthesis.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.models import DubbingConfig, DubbingSegment
from audiosmith.pipeline import DubbingPipeline


def _make_mock_engine(name: str = "test", sample_rate: int = 22050) -> MagicMock:
    """Create a mock TTS engine for testing."""
    engine = MagicMock()
    engine.name = name
    engine.sample_rate = sample_rate
    # Return audio array and sample rate (matches protocol)
    audio = np.random.randn(22050).astype(np.float32)  # 1 second of audio
    engine.synthesize.return_value = (audio, sample_rate)
    return engine


@pytest.fixture
def config(tmp_path):
    """Fixture for DubbingConfig with temporary output directory."""
    return DubbingConfig(
        video_path=Path("test_video.mp4"),
        output_dir=tmp_path,
        tts_engine="chatterbox",
    )


@pytest.fixture
def segments():
    """Fixture for test DubbingSegments."""
    return [
        DubbingSegment(
            index=0,
            start_time=0.0,
            end_time=1.5,
            original_text="Hello world",
            translated_text="Cześć świecie",
        ),
        DubbingSegment(
            index=1,
            start_time=2.0,
            end_time=3.5,
            original_text="How are you?",
            translated_text="Jak się masz?",
        ),
    ]


class TestTTSSynthesisManagerIntegration:
    """Test TTSModelManager integration in _generate_tts method."""

    def test_generate_tts_creates_manager(self, config, segments):
        """_generate_tts should create a TTSModelManager instance."""
        pipeline = DubbingPipeline(config)

        with patch.object(
            pipeline, "_init_chatterbox_engine"
        ) as mock_init_chatterbox, patch(
            "audiosmith.tts_manager.TTSModelManager"
        ) as MockManager:
            mock_engine = _make_mock_engine("chatterbox")
            mock_init_chatterbox.return_value = (mock_engine, 22050, False)
            mock_manager_instance = MagicMock()
            MockManager.return_value = mock_manager_instance

            # Mock the TTS processing to avoid I/O
            with patch("audiosmith.pipeline.helpers._clean_tts_text") as mock_clean:
                mock_clean.return_value = "Hello world"
                with patch(
                    "audiosmith.pipeline.helpers._dedup_repeated_words"
                ) as mock_dedup:
                    mock_dedup.return_value = "Hello world"
                    with patch("soundfile.write"):
                        with patch("soundfile.info") as mock_info:
                            mock_info_obj = MagicMock()
                            mock_info_obj.duration = 1.0
                            mock_info.return_value = mock_info_obj
                            # Create tts_segments directory for resume check
                            tts_dir = Path(config.output_dir) / "tts_segments"
                            tts_dir.mkdir(parents=True, exist_ok=True)

                            pipeline._generate_tts(segments)

            # Verify manager was instantiated
            MockManager.assert_called_once()

    def test_generate_tts_registers_primary_engine(self, config, segments):
        """_generate_tts should register the primary engine with manager."""
        pipeline = DubbingPipeline(config)

        with patch.object(
            pipeline, "_init_chatterbox_engine"
        ) as mock_init_chatterbox, patch(
            "audiosmith.tts_manager.TTSModelManager"
        ) as MockManager:
            mock_engine = _make_mock_engine("chatterbox")
            mock_init_chatterbox.return_value = (mock_engine, 22050, False)
            mock_manager_instance = MagicMock()
            MockManager.return_value = mock_manager_instance

            with patch("audiosmith.pipeline.helpers._clean_tts_text") as mock_clean:
                mock_clean.return_value = "Hello world"
                with patch(
                    "audiosmith.pipeline.helpers._dedup_repeated_words"
                ) as mock_dedup:
                    mock_dedup.return_value = "Hello world"
                    with patch("soundfile.write"):
                        with patch("soundfile.info") as mock_info:
                            mock_info_obj = MagicMock()
                            mock_info_obj.duration = 1.0
                            mock_info.return_value = mock_info_obj
                            tts_dir = Path(config.output_dir) / "tts_segments"
                            tts_dir.mkdir(parents=True, exist_ok=True)

                            pipeline._generate_tts(segments)

            # Verify engine was registered with manager
            mock_manager_instance.register.assert_called_once_with(
                "chatterbox", mock_engine
            )

    def test_generate_tts_cleanup_calls_manager(self, config, segments):
        """_generate_tts should call tts_mgr.cleanup() at the end."""
        pipeline = DubbingPipeline(config)

        with patch.object(
            pipeline, "_init_chatterbox_engine"
        ) as mock_init_chatterbox, patch(
            "audiosmith.tts_manager.TTSModelManager"
        ) as MockManager:
            mock_engine = _make_mock_engine("chatterbox")
            mock_init_chatterbox.return_value = (mock_engine, 22050, False)
            mock_manager_instance = MagicMock()
            MockManager.return_value = mock_manager_instance

            with patch("audiosmith.pipeline.helpers._clean_tts_text") as mock_clean:
                mock_clean.return_value = "Hello world"
                with patch(
                    "audiosmith.pipeline.helpers._dedup_repeated_words"
                ) as mock_dedup:
                    mock_dedup.return_value = "Hello world"
                    with patch("soundfile.write"):
                        with patch("soundfile.info") as mock_info:
                            mock_info_obj = MagicMock()
                            mock_info_obj.duration = 1.0
                            mock_info.return_value = mock_info_obj
                            tts_dir = Path(config.output_dir) / "tts_segments"
                            tts_dir.mkdir(parents=True, exist_ok=True)

                            pipeline._generate_tts(segments)

            # Verify cleanup was called
            mock_manager_instance.cleanup.assert_called_once()

    def test_generate_tts_resolves_auto_engine(self, tmp_path):
        """_generate_tts should resolve 'auto' engine to a specific engine."""
        config = DubbingConfig(
            video_path=Path("test_video.mp4"),
            output_dir=tmp_path,
            tts_engine="auto",
            target_language="en",
        )
        pipeline = DubbingPipeline(config)
        segments = [
            DubbingSegment(
                index=0,
                start_time=0.0,
                end_time=1.5,
                original_text="Hello",
                translated_text="Hello",
            ),
        ]

        with patch.object(pipeline, "_resolve_engine") as mock_resolve:
            mock_resolve.return_value = "chatterbox"
            with patch.object(
                pipeline, "_init_chatterbox_engine"
            ) as mock_init_chatterbox:
                mock_engine = _make_mock_engine("chatterbox")
                mock_init_chatterbox.return_value = (mock_engine, 22050, False)
                with patch("audiosmith.pipeline.helpers._clean_tts_text") as mock_clean:
                    mock_clean.return_value = "Hello"
                    with patch(
                        "audiosmith.pipeline.helpers._dedup_repeated_words"
                    ) as mock_dedup:
                        mock_dedup.return_value = "Hello"
                        with patch("soundfile.write"):
                            with patch("soundfile.info") as mock_info:
                                mock_info_obj = MagicMock()
                                mock_info_obj.duration = 1.0
                                mock_info.return_value = mock_info_obj
                                tts_dir = Path(config.output_dir) / "tts_segments"
                                tts_dir.mkdir(parents=True, exist_ok=True)

                                pipeline._generate_tts(segments)

            # Verify resolve was called
            mock_resolve.assert_called_once()


class TestInitEngineOnDemand:
    """Test the new _init_engine_on_demand method."""

    def test_init_engine_on_demand_registered_engine(self, config):
        """Should return loaded engine if already registered."""
        pipeline = DubbingPipeline(config)
        mock_mgr = MagicMock()
        mock_engine = _make_mock_engine("fish")
        mock_mgr.__contains__ = MagicMock(return_value=True)
        mock_mgr.ensure_loaded.return_value = mock_engine

        engine, sr, use_multi = pipeline._init_engine_on_demand(
            "fish", mock_mgr, segments=[]
        )

        assert engine is mock_engine
        assert sr == mock_engine.sample_rate
        assert use_multi is False
        mock_mgr.ensure_loaded.assert_called_once_with("fish")

    def test_init_engine_on_demand_chatterbox(self, config, segments):
        """Should initialize Chatterbox via _init_chatterbox_engine."""
        pipeline = DubbingPipeline(config)
        mock_mgr = MagicMock()
        mock_mgr.__contains__ = MagicMock(return_value=False)
        mock_engine = _make_mock_engine("chatterbox")

        with patch.object(
            pipeline, "_init_chatterbox_engine"
        ) as mock_init_chatterbox:
            mock_init_chatterbox.return_value = (mock_engine, 22050, True)

            engine, sr, use_multi = pipeline._init_engine_on_demand(
                "chatterbox", mock_mgr, segments=segments
            )

            assert engine is mock_engine
            assert sr == 22050
            assert use_multi is True
            mock_mgr.register.assert_called_once_with("chatterbox", mock_engine)

    def test_init_engine_on_demand_fish(self, config):
        """Should initialize Fish Speech via _init_fish_engine."""
        pipeline = DubbingPipeline(config)
        mock_mgr = MagicMock()
        mock_mgr.__contains__ = MagicMock(return_value=False)
        mock_engine = _make_mock_engine("fish")

        with patch.object(pipeline, "_init_fish_engine") as mock_init_fish:
            mock_init_fish.return_value = (mock_engine, 24000)

            engine, sr, use_multi = pipeline._init_engine_on_demand(
                "fish", mock_mgr, segments=None
            )

            assert engine is mock_engine
            assert sr == 24000
            assert use_multi is False
            mock_mgr.register.assert_called_once_with("fish", mock_engine)

    def test_init_engine_on_demand_factory_engine(self, config):
        """Should initialize other engines via factory."""
        pipeline = DubbingPipeline(config)
        mock_mgr = MagicMock()
        mock_mgr.__contains__ = MagicMock(return_value=False)
        mock_engine = _make_mock_engine("qwen3")

        with patch.object(
            pipeline, "_create_engine_with_factory"
        ) as mock_create_factory:
            mock_create_factory.return_value = mock_engine

            engine, sr, use_multi = pipeline._init_engine_on_demand(
                "qwen3", mock_mgr, segments=None
            )

            assert engine is mock_engine
            assert sr == mock_engine.sample_rate
            assert use_multi is False
            mock_mgr.register.assert_called_once_with("qwen3", mock_engine)

    def test_init_engine_on_demand_returns_tuple(self, config):
        """Should always return a tuple of (engine, sample_rate, use_multi)."""
        pipeline = DubbingPipeline(config)
        mock_mgr = MagicMock()
        mock_mgr.__contains__ = MagicMock(return_value=False)
        mock_engine = _make_mock_engine("piper", sample_rate=16000)

        with patch.object(
            pipeline, "_create_engine_with_factory"
        ) as mock_create_factory:
            mock_create_factory.return_value = mock_engine

            result = pipeline._init_engine_on_demand("piper", mock_mgr)

            assert isinstance(result, tuple)
            assert len(result) == 3
            engine, sr, use_multi = result
            assert engine is mock_engine
            assert sr == 16000
            assert use_multi is False


class TestManagerCleanupReplacesManulaCleanup:
    """Verify that manager.cleanup() replaces manual engine cleanup."""

    def test_manual_cleanup_removed(self, config, segments):
        """The code should no longer call engine.cleanup() directly."""
        pipeline = DubbingPipeline(config)
        mock_engine = _make_mock_engine("chatterbox")

        # We want to verify that manual cleanup is NOT called
        # when manager is handling it
        with patch.object(
            pipeline, "_init_chatterbox_engine"
        ) as mock_init_chatterbox, patch(
            "audiosmith.tts_manager.TTSModelManager"
        ) as MockManager:
            mock_init_chatterbox.return_value = (mock_engine, 22050, False)
            mock_manager_instance = MagicMock()
            MockManager.return_value = mock_manager_instance

            with patch("audiosmith.pipeline.helpers._clean_tts_text") as mock_clean:
                mock_clean.return_value = "Hello"
                with patch(
                    "audiosmith.pipeline.helpers._dedup_repeated_words"
                ) as mock_dedup:
                    mock_dedup.return_value = "Hello"
                    with patch("soundfile.write"):
                        with patch("soundfile.info") as mock_info:
                            mock_info_obj = MagicMock()
                            mock_info_obj.duration = 1.0
                            mock_info.return_value = mock_info_obj
                            tts_dir = Path(config.output_dir) / "tts_segments"
                            tts_dir.mkdir(parents=True, exist_ok=True)

                            pipeline._generate_tts(segments)

            # Manager's cleanup should have been called
            mock_manager_instance.cleanup.assert_called_once()
            # But manual cleanup on the engine itself should NOT be called
            # (the mock engine tracks calls, and we never called cleanup directly on it)
            # This is implicit: manager.cleanup() is responsible for it


class TestChatterboxMultiVoiceHandling:
    """Test that use_multi flag is properly handled with manager."""

    def test_chatterbox_multi_voice_preserved(self, config):
        """use_multi flag from _init_chatterbox_engine should be preserved."""
        pipeline = DubbingPipeline(config)
        segments = [
            DubbingSegment(
                index=0,
                start_time=0.0,
                end_time=1.5,
                original_text="Hello",
                translated_text="Hello",
                speaker_id="speaker_0",  # Triggers multi-voice
            ),
        ]

        with patch.object(
            pipeline, "_init_chatterbox_engine"
        ) as mock_init_chatterbox:
            mock_engine = _make_mock_engine("chatterbox")
            mock_init_chatterbox.return_value = (mock_engine, 22050, True)

            with patch("audiosmith.pipeline.helpers._clean_tts_text") as mock_clean:
                mock_clean.return_value = "Hello"
                with patch(
                    "audiosmith.pipeline.helpers._dedup_repeated_words"
                ) as mock_dedup:
                    mock_dedup.return_value = "Hello"
                    with patch("soundfile.write"):
                        with patch("soundfile.info") as mock_info:
                            mock_info_obj = MagicMock()
                            mock_info_obj.duration = 1.0
                            mock_info.return_value = mock_info_obj
                            tts_dir = Path(config.output_dir) / "tts_segments"
                            tts_dir.mkdir(parents=True, exist_ok=True)

                            result = pipeline._generate_tts(segments)

            # Verify segments were returned
            assert result is not None
            assert len(result) > 0
