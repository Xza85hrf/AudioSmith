"""Tests for audiosmith.emotion_config module."""


from audiosmith.emotion_config import (
    EMOTION_TTS_MAP,
    EMOTION_STYLE_MAP,
    EMOTION_INTENSITY,
    EMOTION_SPECTRAL_INTENSITY,
)


class TestEmotionConfig:
    """Test emotion configuration constants."""

    def test_emotion_tts_map_exists(self):
        """Test that EMOTION_TTS_MAP is defined and accessible."""
        assert EMOTION_TTS_MAP is not None
        assert isinstance(EMOTION_TTS_MAP, dict)

    def test_emotion_tts_map_has_required_emotions(self):
        """Test that EMOTION_TTS_MAP contains expected emotions."""
        required = {'happy', 'sad', 'angry', 'fearful', 'surprised', 'whisper',
                    'sarcastic', 'tender', 'excited', 'determined'}
        assert set(EMOTION_TTS_MAP.keys()) == required

    def test_emotion_tts_map_structure(self):
        """Test that each emotion in EMOTION_TTS_MAP has correct keys."""
        for emotion, params in EMOTION_TTS_MAP.items():
            assert isinstance(params, dict)
            assert 'exaggeration' in params
            assert 'cfg_weight' in params
            assert isinstance(params['exaggeration'], float)
            assert isinstance(params['cfg_weight'], float)
            assert 0 <= params['exaggeration'] <= 1
            assert 0 <= params['cfg_weight'] <= 1

    def test_emotion_style_map_exists(self):
        """Test that EMOTION_STYLE_MAP is defined and accessible."""
        assert EMOTION_STYLE_MAP is not None
        assert isinstance(EMOTION_STYLE_MAP, dict)

    def test_emotion_style_map_has_required_emotions(self):
        """Test that EMOTION_STYLE_MAP contains expected emotions."""
        required = {'neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised',
                    'whisper', 'excited', 'tender', 'sarcastic', 'determined'}
        assert set(EMOTION_STYLE_MAP.keys()) == required

    def test_emotion_style_map_values_in_range(self):
        """Test that EMOTION_STYLE_MAP values are in valid range."""
        for emotion, value in EMOTION_STYLE_MAP.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0

    def test_emotion_intensity_exists(self):
        """Test that EMOTION_INTENSITY is defined and accessible."""
        assert EMOTION_INTENSITY is not None
        assert isinstance(EMOTION_INTENSITY, dict)

    def test_emotion_intensity_has_required_emotions(self):
        """Test that EMOTION_INTENSITY contains expected emotions."""
        required = {'angry', 'excited', 'happy', 'determined', 'surprised',
                    'neutral', 'sarcastic', 'fearful', 'sad', 'tender', 'whisper'}
        assert set(EMOTION_INTENSITY.keys()) == required

    def test_emotion_intensity_values_in_range(self):
        """Test that EMOTION_INTENSITY values are positive floats."""
        for emotion, intensity in EMOTION_INTENSITY.items():
            assert isinstance(intensity, float)
            assert intensity > 0

    def test_emotion_spectral_intensity_exists(self):
        """Test that EMOTION_SPECTRAL_INTENSITY is defined and accessible."""
        assert EMOTION_SPECTRAL_INTENSITY is not None
        assert isinstance(EMOTION_SPECTRAL_INTENSITY, dict)

    def test_emotion_spectral_intensity_values_in_range(self):
        """Test that EMOTION_SPECTRAL_INTENSITY values are in valid range."""
        for emotion, intensity in EMOTION_SPECTRAL_INTENSITY.items():
            assert isinstance(intensity, float)
            assert 0 <= intensity <= 1
