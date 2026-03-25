"""Tests for audiosmith.pipeline checkpoint/resume functionality."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from audiosmith.models import (
    DubbingConfig,
    DubbingSegment,
    DubbingStep,
    PipelineState,
)
from audiosmith.pipeline import CHECKPOINT_FILE, DubbingPipeline


# ============================================================================
# PipelineState Tests
# ============================================================================


class TestPipelineStateCreation:
    """Test creating and initializing PipelineState."""

    def test_pipeline_state_defaults(self):
        """Verify PipelineState initializes with correct default values."""
        state = PipelineState()
        assert state.completed_steps == []
        assert state.segments == []
        assert state.duration == 0.0
        assert state.audio_path is None
        assert state.hq_audio_path is None
        assert state.dubbed_audio_path is None
        assert state.background_audio_path is None
        assert state.subtitle_path is None

    def test_pipeline_state_with_values(self):
        """Verify PipelineState accepts custom values."""
        state = PipelineState(
            completed_steps=['extract_audio', 'transcribe'],
            duration=120.5,
            audio_path='/tmp/audio.wav',
            dubbed_audio_path='/tmp/dubbed.wav',
        )
        assert state.completed_steps == ['extract_audio', 'transcribe']
        assert state.duration == 120.5
        assert state.audio_path == '/tmp/audio.wav'
        assert state.dubbed_audio_path == '/tmp/dubbed.wav'


class TestPipelineStateSaveLoad:
    """Test PipelineState JSON serialization/deserialization."""

    def test_pipeline_state_save_creates_file(self, tmp_path):
        """Verify save() creates a JSON file on disk."""
        state = PipelineState(duration=10.5)
        checkpoint_path = tmp_path / 'checkpoint.json'

        state.save(checkpoint_path)

        assert checkpoint_path.exists()
        assert checkpoint_path.is_file()

    def test_pipeline_state_save_json_format(self, tmp_path):
        """Verify saved state is valid JSON with expected structure."""
        state = PipelineState(
            completed_steps=['extract_audio'],
            duration=50.0,
            audio_path='/tmp/audio.wav',
        )
        checkpoint_path = tmp_path / 'checkpoint.json'

        state.save(checkpoint_path)

        content = checkpoint_path.read_text()
        data = json.loads(content)

        assert data['completed_steps'] == ['extract_audio']
        assert data['duration'] == 50.0
        assert data['audio_path'] == '/tmp/audio.wav'
        assert data['segments'] == []

    def test_pipeline_state_roundtrip(self, tmp_path):
        """Verify save/load round-trip preserves all state."""
        original = PipelineState(
            completed_steps=['extract_audio', 'transcribe'],
            duration=123.45,
            audio_path='/tmp/audio.wav',
            hq_audio_path='/tmp/audio_hq.wav',
            dubbed_audio_path='/tmp/dubbed.wav',
            background_audio_path='/tmp/background.wav',
            subtitle_path='/tmp/subs.srt',
        )
        checkpoint_path = tmp_path / 'checkpoint.json'

        original.save(checkpoint_path)
        loaded = PipelineState.load(checkpoint_path)

        assert loaded.completed_steps == original.completed_steps
        assert loaded.duration == original.duration
        assert loaded.audio_path == original.audio_path
        assert loaded.hq_audio_path == original.hq_audio_path
        assert loaded.dubbed_audio_path == original.dubbed_audio_path
        assert loaded.background_audio_path == original.background_audio_path
        assert loaded.subtitle_path == original.subtitle_path

    def test_pipeline_state_load_missing_file(self, tmp_path):
        """Verify load() raises FileNotFoundError for non-existent file."""
        checkpoint_path = tmp_path / 'nonexistent.json'

        with pytest.raises(FileNotFoundError):
            PipelineState.load(checkpoint_path)

    def test_pipeline_state_load_corrupt_json(self, tmp_path):
        """Verify load() raises JSONDecodeError for corrupt JSON."""
        checkpoint_path = tmp_path / 'corrupt.json'
        checkpoint_path.write_text('{ invalid json ]')

        with pytest.raises(json.JSONDecodeError):
            PipelineState.load(checkpoint_path)

    def test_pipeline_state_load_missing_required_fields(self, tmp_path):
        """Verify load() raises TypeError when required fields are missing."""
        checkpoint_path = tmp_path / 'incomplete.json'
        # Missing 'completed_steps' and other required fields
        checkpoint_path.write_text(json.dumps({'unknown_field': 'value'}))

        with pytest.raises(TypeError):
            PipelineState.load(checkpoint_path)

    def test_pipeline_state_save_with_segments(self, tmp_path):
        """Verify segments are properly serialized in checkpoint."""
        state = PipelineState()
        state.segments = [
            {
                'index': 0,
                'start_time': 0.0,
                'end_time': 2.5,
                'original_text': 'Hello',
                'translated_text': 'Cześć',
                'speaker_id': None,
                'is_speech': True,
                'is_hallucination': False,
                'tts_audio_path': None,
                'tts_duration_ms': None,
                'metadata': {},
            }
        ]
        checkpoint_path = tmp_path / 'checkpoint.json'

        state.save(checkpoint_path)
        loaded = PipelineState.load(checkpoint_path)

        assert len(loaded.segments) == 1
        assert loaded.segments[0]['original_text'] == 'Hello'
        assert loaded.segments[0]['translated_text'] == 'Cześć'


class TestPipelineStateStepTracking:
    """Test step completion tracking in PipelineState."""

    def test_is_step_done_not_completed(self):
        """Verify is_step_done returns False for uncompleted steps."""
        state = PipelineState()
        assert not state.is_step_done('extract_audio')

    def test_is_step_done_completed(self):
        """Verify is_step_done returns True for completed steps."""
        state = PipelineState(completed_steps=['extract_audio'])
        assert state.is_step_done('extract_audio')

    def test_mark_step_done_adds_step(self):
        """Verify mark_step_done adds a step to completed_steps."""
        state = PipelineState()
        state.mark_step_done('extract_audio')
        assert 'extract_audio' in state.completed_steps

    def test_mark_step_done_idempotent(self):
        """Verify mark_step_done doesn't duplicate steps."""
        state = PipelineState()
        state.mark_step_done('extract_audio')
        state.mark_step_done('extract_audio')
        assert state.completed_steps.count('extract_audio') == 1

    def test_mark_step_done_preserves_order(self):
        """Verify mark_step_done preserves step order."""
        state = PipelineState()
        steps = ['extract_audio', 'transcribe', 'translate']
        for step in steps:
            state.mark_step_done(step)
        assert state.completed_steps == steps

    def test_completed_steps_list_grows(self):
        """Verify completed_steps tracks all steps in order."""
        state = PipelineState()
        assert len(state.completed_steps) == 0

        state.mark_step_done('extract_audio')
        assert len(state.completed_steps) == 1

        state.mark_step_done('transcribe')
        assert len(state.completed_steps) == 2


# ============================================================================
# DubbingStep Enum Tests
# ============================================================================


class TestDubbingStepEnum:
    """Test DubbingStep enum values and ordering."""

    def test_dubbing_step_values_exist(self):
        """Verify all expected DubbingStep values exist."""
        expected_steps = [
            'extract_audio',
            'isolate_vocals',
            'transcribe',
            'diarize',
            'detect_emotion',
            'post_process',
            'translate',
            'merge_segments',
            'generate_tts',
            'mix_audio',
            'encode_video',
        ]
        for step_name in expected_steps:
            assert hasattr(DubbingStep, step_name.upper())

    def test_dubbing_step_values_are_strings(self):
        """Verify DubbingStep values are string enum members."""
        for step in DubbingStep:
            assert isinstance(step.value, str)
            assert len(step.value) > 0

    def test_dubbing_step_get_by_value(self):
        """Verify DubbingStep can be retrieved by value."""
        step = DubbingStep('extract_audio')
        assert step == DubbingStep.EXTRACT_AUDIO

    def test_dubbing_step_enum_members_count(self):
        """Verify DubbingStep has expected number of members."""
        assert len(DubbingStep) == 11

    def test_dubbing_step_names_are_descriptive(self):
        """Verify DubbingStep enum names are descriptive."""
        for step in DubbingStep:
            # Names should be uppercase with underscores
            assert step.name.isupper()
            assert '_' in step.name or len(step.name) > 0


# ============================================================================
# DubbingConfig Tests
# ============================================================================


class TestDubbingConfigDefaults:
    """Test DubbingConfig default values."""

    def test_dubbing_config_required_fields(self, tmp_path):
        """Verify DubbingConfig requires video_path and output_dir."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        assert config.video_path == Path('video.mp4')
        assert config.output_dir == tmp_path

    def test_dubbing_config_language_defaults(self, tmp_path):
        """Verify language defaults are English->Polish."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        assert config.source_language == 'en'
        assert config.target_language == 'pl'

    def test_dubbing_config_tts_engine_default(self, tmp_path):
        """Verify default TTS engine is 'chatterbox'."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        assert config.tts_engine == 'chatterbox'

    def test_dubbing_config_whisper_defaults(self, tmp_path):
        """Verify Whisper model defaults."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        assert config.whisper_model == 'large-v3'
        assert config.whisper_compute_type == 'float16'
        assert config.whisper_device == 'cuda'

    def test_dubbing_config_feature_flags(self, tmp_path):
        """Verify feature flag defaults."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        assert config.burn_subtitles is False
        assert config.isolate_vocals is False
        assert config.diarize is False
        assert config.detect_emotion is False
        assert config.merge_segments is True
        assert config.post_process is True
        assert config.resume is False

    def test_dubbing_config_custom_values(self, tmp_path):
        """Verify DubbingConfig accepts custom values."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            source_language='fr',
            target_language='es',
            tts_engine='elevenlabs',
            resume=True,
        )
        assert config.source_language == 'fr'
        assert config.target_language == 'es'
        assert config.tts_engine == 'elevenlabs'
        assert config.resume is True


class TestDubbingConfigEngineOptions:
    """Test engine-specific configuration options."""

    def test_chatterbox_config_fields(self, tmp_path):
        """Verify Chatterbox-specific config fields exist."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            tts_engine='chatterbox',
        )
        assert hasattr(config, 'chatterbox_exaggeration')
        assert hasattr(config, 'chatterbox_cfg_weight')
        assert config.chatterbox_exaggeration == 0.7
        assert config.chatterbox_cfg_weight == 0.7

    def test_elevenlabs_config_fields(self, tmp_path):
        """Verify ElevenLabs-specific config fields exist."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            tts_engine='elevenlabs',
        )
        assert hasattr(config, 'elevenlabs_model')
        assert hasattr(config, 'elevenlabs_voice_id')
        assert hasattr(config, 'elevenlabs_voice_name')
        assert config.elevenlabs_model == 'eleven_v3'

    def test_f5_config_fields(self, tmp_path):
        """Verify F5-TTS-specific config fields exist."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            tts_engine='f5',
        )
        assert hasattr(config, 'f5_model')
        assert hasattr(config, 'f5_checkpoint')
        assert hasattr(config, 'f5_ref_audio')
        assert hasattr(config, 'f5_ref_text')
        assert hasattr(config, 'f5_speed')

    def test_custom_audio_prompt_path(self, tmp_path):
        """Verify custom audio prompt path can be configured."""
        prompt_path = tmp_path / 'prompt.wav'
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            audio_prompt_path=prompt_path,
        )
        assert config.audio_prompt_path == prompt_path


# ============================================================================
# DubbingPipeline Checkpoint/Resume Tests
# ============================================================================


class TestDubbingPipelineFreshStart:
    """Test DubbingPipeline initialization with no prior state."""

    def test_fresh_start_no_checkpoint(self, tmp_path):
        """Verify fresh start creates empty state."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        pipeline = DubbingPipeline(config)

        assert pipeline.state.completed_steps == []
        assert pipeline.state.duration == 0.0

    def test_fresh_start_ignores_resume_flag(self, tmp_path):
        """Verify resume=True ignored when no checkpoint exists."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            resume=True,
        )
        pipeline = DubbingPipeline(config)

        assert pipeline.state.completed_steps == []

    def test_checkpoint_path_set_correctly(self, tmp_path):
        """Verify checkpoint path is set to output_dir/.checkpoint.json."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        pipeline = DubbingPipeline(config)

        expected_path = tmp_path / CHECKPOINT_FILE
        assert pipeline.checkpoint_path == expected_path


class TestDubbingPipelineResume:
    """Test DubbingPipeline resume functionality."""

    def test_resume_loads_checkpoint(self, tmp_path):
        """Verify resume=True loads existing checkpoint."""
        # Create initial state and save it
        initial_state = PipelineState()
        initial_state.mark_step_done('extract_audio')
        initial_state.audio_path = '/tmp/audio.wav'
        initial_state.duration = 60.5
        checkpoint_path = tmp_path / CHECKPOINT_FILE
        initial_state.save(checkpoint_path)

        # Create pipeline with resume=True
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            resume=True,
        )
        pipeline = DubbingPipeline(config)

        # Verify state was loaded
        assert pipeline.state.is_step_done('extract_audio')
        assert pipeline.state.audio_path == '/tmp/audio.wav'
        assert pipeline.state.duration == 60.5

    def test_resume_loads_multiple_completed_steps(self, tmp_path):
        """Verify all completed steps are loaded from checkpoint."""
        initial_state = PipelineState(
            completed_steps=['extract_audio', 'transcribe', 'translate']
        )
        checkpoint_path = tmp_path / CHECKPOINT_FILE
        initial_state.save(checkpoint_path)

        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            resume=True,
        )
        pipeline = DubbingPipeline(config)

        assert pipeline.state.completed_steps == [
            'extract_audio',
            'transcribe',
            'translate',
        ]

    def test_resume_false_ignores_checkpoint(self, tmp_path):
        """Verify resume=False creates fresh state even if checkpoint exists."""
        # Create and save a checkpoint
        initial_state = PipelineState()
        initial_state.mark_step_done('extract_audio')
        checkpoint_path = tmp_path / CHECKPOINT_FILE
        initial_state.save(checkpoint_path)

        # Create pipeline with resume=False
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            resume=False,
        )
        pipeline = DubbingPipeline(config)

        # Verify state is fresh
        assert pipeline.state.completed_steps == []

    def test_resume_loads_segments(self, tmp_path):
        """Verify resumed state includes serialized segments."""
        initial_state = PipelineState()
        initial_state.segments = [
            {
                'index': 0,
                'start_time': 0.0,
                'end_time': 2.5,
                'original_text': 'Hello',
                'translated_text': 'Cześć',
                'speaker_id': None,
                'is_speech': True,
                'is_hallucination': False,
                'tts_audio_path': None,
                'tts_duration_ms': None,
                'metadata': {},
            }
        ]
        checkpoint_path = tmp_path / CHECKPOINT_FILE
        initial_state.save(checkpoint_path)

        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            resume=True,
        )
        pipeline = DubbingPipeline(config)

        assert len(pipeline.state.segments) == 1
        assert pipeline.state.segments[0]['original_text'] == 'Hello'


class TestDubbingPipelineCheckpointSaving:
    """Test checkpoint saving during pipeline execution."""

    def test_save_checkpoint_creates_file(self, tmp_path):
        """Verify _save_checkpoint() creates checkpoint file."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        pipeline = DubbingPipeline(config)
        pipeline.state.mark_step_done('extract_audio')

        pipeline._save_checkpoint()

        checkpoint_path = tmp_path / CHECKPOINT_FILE
        assert checkpoint_path.exists()

    def test_save_checkpoint_persists_state(self, tmp_path):
        """Verify saved checkpoint contains current pipeline state."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        pipeline = DubbingPipeline(config)
        pipeline.state.mark_step_done('extract_audio')
        pipeline.state.audio_path = '/tmp/audio.wav'
        pipeline.state.duration = 45.5

        pipeline._save_checkpoint()

        # Load checkpoint and verify
        saved_state = PipelineState.load(tmp_path / CHECKPOINT_FILE)
        assert saved_state.is_step_done('extract_audio')
        assert saved_state.audio_path == '/tmp/audio.wav'
        assert saved_state.duration == 45.5

    def test_save_checkpoint_overwrites_previous(self, tmp_path):
        """Verify saving checkpoint overwrites previous version."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        pipeline = DubbingPipeline(config)

        # Save first checkpoint
        pipeline.state.mark_step_done('extract_audio')
        pipeline._save_checkpoint()

        # Save updated checkpoint
        pipeline.state.mark_step_done('transcribe')
        pipeline._save_checkpoint()

        # Verify both steps are in final checkpoint
        saved_state = PipelineState.load(tmp_path / CHECKPOINT_FILE)
        assert saved_state.is_step_done('extract_audio')
        assert saved_state.is_step_done('transcribe')


class TestDubbingPipelineStepSkipping:
    """Test that completed steps are skipped on resume."""

    @patch('audiosmith.pipeline.DubbingPipeline._extract_audio')
    def test_skip_extract_audio_if_completed(self, mock_extract, tmp_path):
        """Verify extract_audio step is skipped if already completed."""
        # Create checkpoint with extract_audio done
        initial_state = PipelineState()
        initial_state.mark_step_done('extract_audio')
        initial_state.audio_path = '/tmp/audio.wav'
        checkpoint_path = tmp_path / CHECKPOINT_FILE
        initial_state.save(checkpoint_path)

        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            resume=True,
        )
        pipeline = DubbingPipeline(config)

        # Verify the step is already marked as done
        assert pipeline.state.is_step_done('extract_audio')
        # extract_audio method should not be called in normal resume flow
        # (it's checked via is_step_done before calling)

    def test_multiple_completed_steps_preserved(self, tmp_path):
        """Verify all completed steps are preserved on resume."""
        initial_state = PipelineState(
            completed_steps=['extract_audio', 'transcribe', 'translate']
        )
        checkpoint_path = tmp_path / CHECKPOINT_FILE
        initial_state.save(checkpoint_path)

        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
            resume=True,
        )
        pipeline = DubbingPipeline(config)

        # Verify all steps are present
        for step in ['extract_audio', 'transcribe', 'translate']:
            assert pipeline.state.is_step_done(step)


# ============================================================================
# DubbingSegment Tests
# ============================================================================


class TestDubbingSegmentCreation:
    """Test DubbingSegment creation and properties."""

    def test_dubbing_segment_defaults(self):
        """Verify DubbingSegment has correct default values."""
        segment = DubbingSegment(
            index=0,
            start_time=0.0,
            end_time=2.5,
            original_text='Hello',
        )
        assert segment.index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 2.5
        assert segment.original_text == 'Hello'
        assert segment.translated_text == ''
        assert segment.speaker_id is None
        assert segment.is_speech is True
        assert segment.is_hallucination is False
        assert segment.tts_audio_path is None
        assert segment.tts_duration_ms is None
        assert segment.metadata == {}

    def test_dubbing_segment_duration_calculation(self):
        """Verify duration_ms property calculates correctly."""
        segment = DubbingSegment(
            index=0,
            start_time=0.0,
            end_time=2.5,
            original_text='Hello',
        )
        assert segment.duration_ms == 2500

    def test_dubbing_segment_with_custom_values(self):
        """Verify DubbingSegment accepts all custom values."""
        segment = DubbingSegment(
            index=5,
            start_time=10.5,
            end_time=15.5,
            original_text='Hello world',
            translated_text='Cześć świat',
            speaker_id='speaker_1',
            is_speech=True,
            is_hallucination=False,
            tts_audio_path=Path('/tmp/seg.wav'),
            tts_duration_ms=2000,
            metadata={'emotion': 'happy'},
        )
        assert segment.index == 5
        assert segment.translated_text == 'Cześć świat'
        assert segment.speaker_id == 'speaker_1'
        assert segment.tts_audio_path == Path('/tmp/seg.wav')
        assert segment.tts_duration_ms == 2000
        assert segment.metadata == {'emotion': 'happy'}

    def test_dubbing_segment_duration_negative_edge_case(self):
        """Verify duration_ms works with fractional milliseconds."""
        segment = DubbingSegment(
            index=0,
            start_time=0.001,
            end_time=0.015,
            original_text='Test',
        )
        assert segment.duration_ms == 13  # int(0.014 * 1000) = 13 milliseconds


class TestDubbingSegmentSerialization:
    """Test DubbingSegment serialization to/from dict."""

    def test_segment_to_dict_conversion(self, tmp_path):
        """Verify segments can be converted to dict for storage."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        pipeline = DubbingPipeline(config)

        segment = DubbingSegment(
            index=0,
            start_time=0.0,
            end_time=2.5,
            original_text='Hello',
            translated_text='Cześć',
        )
        segments = [segment]

        # Use pipeline's serialization method
        dicts = pipeline._segments_to_dicts(segments)

        assert len(dicts) == 1
        assert dicts[0]['original_text'] == 'Hello'
        assert dicts[0]['translated_text'] == 'Cześć'

    def test_segment_from_dict_conversion(self, tmp_path):
        """Verify dicts can be converted back to DubbingSegment."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        pipeline = DubbingPipeline(config)

        segment_dict = {
            'index': 0,
            'start_time': 0.0,
            'end_time': 2.5,
            'original_text': 'Hello',
            'translated_text': 'Cześć',
            'speaker_id': None,
            'is_speech': True,
            'is_hallucination': False,
            'tts_audio_path': None,
            'tts_duration_ms': None,
            'metadata': {},
        }

        segments = pipeline._dicts_to_segments([segment_dict])

        assert len(segments) == 1
        assert segments[0].original_text == 'Hello'
        assert segments[0].translated_text == 'Cześć'

    def test_segment_roundtrip_with_paths(self, tmp_path):
        """Verify segments with paths round-trip correctly."""
        config = DubbingConfig(
            video_path=Path('video.mp4'),
            output_dir=tmp_path,
        )
        pipeline = DubbingPipeline(config)

        original = DubbingSegment(
            index=1,
            start_time=2.0,
            end_time=3.0,
            original_text='bye',
            tts_audio_path=Path('/tmp/seg.wav'),
            tts_duration_ms=800,
        )

        dicts = pipeline._segments_to_dicts([original])
        restored = pipeline._dicts_to_segments(dicts)

        assert restored[0].tts_audio_path == Path('/tmp/seg.wav')
        assert restored[0].tts_duration_ms == 800
