"""Tests for audiosmith.emotion module (no GPU or network required)."""

import sys
from unittest.mock import patch

import pytest

from audiosmith.emotion import Emotion, EmotionResult, EmotionEngine, detect_emotion


class TestEmotion:
    def test_values(self):
        assert len(list(Emotion)) == 11

    def test_neutral_default(self):
        assert Emotion('neutral') == Emotion.NEUTRAL


class TestEmotionResult:
    def test_defaults(self):
        result = EmotionResult(Emotion.NEUTRAL, 0.5)
        assert result.secondary_emotion is None
        assert result.intensity == 0.5
        assert result.context_notes == ''

    def test_custom(self):
        result = EmotionResult(
            primary_emotion=Emotion.HAPPY,
            confidence=0.8,
            secondary_emotion=Emotion.EXCITED,
            intensity=0.7,
            context_notes='test',
        )
        assert result.primary_emotion == Emotion.HAPPY
        assert result.confidence == 0.8
        assert result.secondary_emotion == Emotion.EXCITED
        assert result.intensity == 0.7


class TestRuleBasedAnalysis:
    @pytest.fixture
    def engine(self):
        return EmotionEngine(use_classifier=False)

    def test_happy_text(self, engine):
        result = engine.analyze('I am so happy!', use_context=False)
        assert result.primary_emotion == Emotion.HAPPY

    def test_sad_text(self, engine):
        result = engine.analyze('This is so sad...', use_context=False)
        assert result.primary_emotion == Emotion.SAD

    def test_angry_text(self, engine):
        result = engine.analyze('THIS IS RIDICULOUS!!', use_context=False)
        assert result.primary_emotion == Emotion.ANGRY

    def test_neutral_text(self, engine):
        result = engine.analyze('The meeting is at 3pm.', use_context=False)
        assert result.primary_emotion == Emotion.NEUTRAL

    def test_whisper_text(self, engine):
        result = engine.analyze('(whisper) Keep it a secret', use_context=False)
        assert result.primary_emotion == Emotion.WHISPER

    def test_polish_happy(self, engine):
        result = engine.analyze('Kocham to! Wspaniały dzień!', use_context=False)
        assert result.primary_emotion == Emotion.HAPPY

    def test_intensity_booster(self, engine):
        result = engine.analyze('I am very extremely happy!', use_context=False)
        assert result.intensity > 0.5

    def test_intensity_reducer(self, engine):
        result = engine.analyze('I am a little sad', use_context=False)
        assert result.intensity < 0.5


class TestContextTracking:
    def test_context_inherits(self):
        engine = EmotionEngine(use_classifier=False, context_window=3)
        # First call builds context with strong emotion
        engine.analyze('I am so happy and delighted!', use_context=False)
        # Second call: neutral weak text should inherit from context
        result = engine.analyze('OK.', use_context=True)
        assert result.primary_emotion == Emotion.HAPPY
        assert result.context_notes == 'Inherited from context'

    def test_reset_context(self):
        engine = EmotionEngine(use_classifier=False, context_window=3)
        engine.analyze('I am so happy and delighted!', use_context=False)
        engine.reset_context()
        result = engine.analyze('OK.', use_context=True)
        assert result.primary_emotion == Emotion.NEUTRAL


class TestProsodyParams:
    @pytest.fixture
    def engine(self):
        return EmotionEngine(use_classifier=False)

    def test_neutral_prosody(self, engine):
        params = engine.get_prosody_params(Emotion.NEUTRAL)
        assert params['rate'] == '+0%'
        assert params['pitch'] == '+0Hz'
        assert params['volume'] == '+0%'

    def test_happy_prosody(self, engine):
        params = engine.get_prosody_params(Emotion.HAPPY)
        assert '+' in params['rate']
        assert '+' in params['pitch']
        assert '+' in params['volume']

    def test_whisper_prosody(self, engine):
        params = engine.get_prosody_params(Emotion.WHISPER)
        assert '-' in params['volume']

    def test_intensity_scaling(self, engine):
        default_params = engine.get_prosody_params(Emotion.HAPPY, intensity=0.5)
        scaled_params = engine.get_prosody_params(Emotion.HAPPY, intensity=1.0)
        # Extract numeric value from rate string
        default_rate = int(default_params['rate'].replace('%', '').replace('+', ''))
        scaled_rate = int(scaled_params['rate'].replace('%', '').replace('+', ''))
        assert scaled_rate > default_rate


class TestBatch:
    def test_analyze_batch(self):
        engine = EmotionEngine(use_classifier=False)
        results = engine.analyze_batch(['Happy day!', 'Sad news...'])
        assert len(results) == 2
        assert all(isinstance(r, EmotionResult) for r in results)


class TestDetectEmotion:
    def test_convenience(self):
        name, confidence = detect_emotion('I am happy!')
        assert name == 'happy'
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


class TestClassifierFallback:
    def test_missing_transformers(self):
        """With use_classifier=True but transformers missing, falls back to rules."""
        engine = EmotionEngine(use_classifier=True)
        with patch.dict(sys.modules, {'transformers': None}):
            result = engine.analyze('I am so happy!', use_context=False)
            assert result.primary_emotion == Emotion.HAPPY
