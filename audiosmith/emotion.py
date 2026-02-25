"""Emotion detection from text for TTS enhancement — rule-based with optional ML fallback."""

import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Emotion(Enum):
    """Supported emotions for TTS prosody control."""
    NEUTRAL = 'neutral'
    HAPPY = 'happy'
    SAD = 'sad'
    ANGRY = 'angry'
    FEARFUL = 'fearful'
    SURPRISED = 'surprised'
    WHISPER = 'whisper'
    EXCITED = 'excited'
    TENDER = 'tender'
    SARCASTIC = 'sarcastic'
    DETERMINED = 'determined'


@dataclass
class EmotionResult:
    """Result of emotion analysis."""
    primary_emotion: Emotion
    confidence: float
    secondary_emotion: Optional[Emotion] = None
    intensity: float = 0.5
    context_notes: str = ''


class EmotionEngine:
    """Detect emotions from text using rule-based analysis with optional ML classifier fallback."""

    EMOTION_PATTERNS: Dict[Emotion, Dict[str, List[str]]] = {
        Emotion.HAPPY: {
            'keywords': [
                'happy', 'joy', 'glad', 'wonderful', 'great', 'awesome', 'amazing',
                'love', 'fantastic', 'excited', 'yay', 'hurray', 'delighted',
                'szczęśliwy', 'radość', 'wspaniały', 'cudowny', 'świetny', 'kocham',
                'fantastyczny', 'zachwycony',
            ],
            'patterns': [r'!\s*$', r'(?i)ha\s*ha', r':[\)D]'],
        },
        Emotion.SAD: {
            'keywords': [
                'sad', 'sorry', 'unfortunately', 'miss', 'lost', 'cry', 'tears',
                'disappointed', 'heartbroken', 'grief', 'mourn', 'depressed',
                'smutny', 'przepraszam', 'niestety', 'tęsknię', 'płakać', 'łzy',
                'rozczarowany', 'żałoba',
            ],
            'patterns': [r'\.\.\.\s*$', r':\('],
        },
        Emotion.ANGRY: {
            'keywords': [
                'angry', 'furious', 'hate', 'damn', 'hell', 'stupid', 'idiot',
                'outrageous', 'unacceptable', 'ridiculous', 'enough',
                'wściekły', 'zły', 'nienawidzę', 'cholera', 'głupi', 'idiota',
                'skandal', 'dość',
            ],
            'patterns': [r'!{2,}', r'\b[A-Z]{4,}\b'],
        },
        Emotion.FEARFUL: {
            'keywords': [
                'afraid', 'fear', 'scared', 'terrified', 'horror', 'danger',
                'help', 'run', 'escape', 'threat', 'panic',
                'strach', 'boje się', 'przerażony', 'niebezpieczeństwo', 'pomocy',
                'uciekaj', 'panika',
            ],
            'patterns': [r'\?!'],
        },
        Emotion.SURPRISED: {
            'keywords': [
                'wow', 'really', 'seriously', 'unbelievable', 'incredible',
                'no way', 'oh my', 'shocking', 'unexpected',
                'naprawdę', 'serio', 'niewiarygodne', 'szok', 'nie wierzę',
            ],
            'patterns': [r'\?{2,}', r'(?i)^oh\b'],
        },
        Emotion.WHISPER: {
            'keywords': [
                'whisper', 'quiet', 'secret', 'shh', 'psst', 'hush',
                'between us', "don't tell",
                'szept', 'cicho', 'sekret', 'ciii', 'między nami',
            ],
            'patterns': [r'(?i)\(.*whisper.*\)', r'(?i)\[.*quietly.*\]'],
        },
        Emotion.EXCITED: {
            'keywords': [
                'excited', "can't wait", 'amazing', 'incredible', "let's go",
                'finally', 'woohoo',
                'podekscytowany', 'nie mogę się doczekać', 'nareszcie',
            ],
            'patterns': [r'!{3,}'],
        },
        Emotion.TENDER: {
            'keywords': [
                'dear', 'darling', 'sweetheart', 'gentle', 'soft', 'care',
                'precious', 'beloved',
                'kochanie', 'skarbie', 'delikatny', 'drogi', 'ukochany',
            ],
            'patterns': [],
        },
        Emotion.DETERMINED: {
            'keywords': [
                'must', 'will', 'have to', 'going to', 'definitely', 'absolutely',
                'no matter what', 'whatever it takes',
                'muszę', 'będę', 'na pewno', 'absolutnie', 'bez względu',
            ],
            'patterns': [],
        },
    }

    INTENSITY_BOOSTERS = [
        'very', 'extremely', 'so', 'really', 'truly', 'absolutely',
        'bardzo', 'niezwykle', 'naprawdę',
    ]
    INTENSITY_REDUCERS = [
        'bit', 'little', 'slightly', 'somewhat', 'kind of',
        'trochę', 'nieco', 'jakby',
    ]

    def __init__(self, use_classifier: bool = False, context_window: int = 3):
        self.use_classifier = use_classifier
        self.context_window = context_window
        self._context_history: List[EmotionResult] = []
        self._classifier = None

    def analyze(self, text: str, use_context: bool = True) -> EmotionResult:
        """Analyze text for emotion. Returns EmotionResult."""
        result = self._analyze_rules(text)

        if self.use_classifier and result.confidence < 0.7:
            try:
                classifier_result = self._analyze_classifier(text)
                if classifier_result.confidence > result.confidence:
                    result = classifier_result
            except Exception:
                pass  # Fall back to rule-based result

        if use_context and self._context_history:
            result = self._apply_context(result)

        self._context_history.append(result)
        if len(self._context_history) > self.context_window:
            self._context_history.pop(0)

        return result

    def analyze_batch(self, texts: List[str]) -> List[EmotionResult]:
        """Analyze multiple texts with context continuity."""
        return [self.analyze(text) for text in texts]

    def _analyze_rules(self, text: str) -> EmotionResult:
        """Rule-based emotion detection using keywords and regex patterns."""
        text_lower = text.lower()
        scores: Dict[Emotion, float] = {e: 0.0 for e in Emotion}

        for emotion, patterns in self.EMOTION_PATTERNS.items():
            for kw in patterns['keywords']:
                if kw in text_lower:
                    scores[emotion] += 1.0
            for pattern in patterns['patterns']:
                if re.search(pattern, text):
                    scores[emotion] += 0.5

        # Calculate intensity from modifiers
        intensity = 0.5
        for booster in self.INTENSITY_BOOSTERS:
            if booster in text_lower:
                intensity = min(1.0, intensity + 0.15)
        for reducer in self.INTENSITY_REDUCERS:
            if reducer in text_lower:
                intensity = max(0.2, intensity - 0.1)

        total_score = sum(scores.values())
        if total_score == 0:
            return EmotionResult(Emotion.NEUTRAL, 0.5, intensity=intensity)

        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_emotion, primary_score = sorted_emotions[0]

        secondary_emotion = None
        if len(sorted_emotions) > 1 and sorted_emotions[1][1] > 0:
            secondary_emotion = sorted_emotions[1][0]

        confidence = min(0.9, primary_score / max(total_score, 1) * 0.8 + 0.1)

        return EmotionResult(
            primary_emotion=primary_emotion if primary_score > 0 else Emotion.NEUTRAL,
            confidence=confidence,
            secondary_emotion=secondary_emotion,
            intensity=intensity,
        )

    def _analyze_classifier(self, text: str) -> EmotionResult:
        """Classify emotion using transformers pipeline (lazy-loaded)."""
        from transformers import pipeline as hf_pipeline

        if self._classifier is None:
            self._classifier = hf_pipeline(
                'text-classification',
                model='j-hartmann/emotion-english-distilroberta-base',
                top_k=2,
            )

        results = self._classifier(text)
        if not results:
            return EmotionResult(Emotion.NEUTRAL, 0.5)

        top = results[0][0]
        label_map = {
            'joy': Emotion.HAPPY,
            'sadness': Emotion.SAD,
            'anger': Emotion.ANGRY,
            'fear': Emotion.FEARFUL,
            'surprise': Emotion.SURPRISED,
            'disgust': Emotion.ANGRY,
            'neutral': Emotion.NEUTRAL,
        }
        primary = label_map.get(top['label'], Emotion.NEUTRAL)
        return EmotionResult(primary_emotion=primary, confidence=top['score'])

    def _apply_context(self, result: EmotionResult) -> EmotionResult:
        """Inherit emotion from recent history when current result is weak neutral."""
        if result.primary_emotion == Emotion.NEUTRAL and result.confidence <= 0.5:
            for prev in reversed(self._context_history):
                if prev.primary_emotion != Emotion.NEUTRAL and prev.confidence > 0.6:
                    return EmotionResult(
                        primary_emotion=prev.primary_emotion,
                        confidence=prev.confidence * 0.7,
                        intensity=prev.intensity * 0.8,
                        context_notes='Inherited from context',
                    )
        return result

    def reset_context(self) -> None:
        """Clear emotion context history."""
        self._context_history.clear()

    def get_prosody_params(self, emotion: Emotion, intensity: float = 0.5) -> Dict[str, str]:
        """Get TTS prosody parameters (rate, pitch, volume) for an emotion and intensity."""
        prosody_map = {
            Emotion.NEUTRAL: {'rate': '+0%', 'pitch': '+0Hz', 'volume': '+0%'},
            Emotion.HAPPY: {'rate': '+5%', 'pitch': '+5Hz', 'volume': '+5%'},
            Emotion.SAD: {'rate': '-10%', 'pitch': '-5Hz', 'volume': '-5%'},
            Emotion.ANGRY: {'rate': '+10%', 'pitch': '+10Hz', 'volume': '+10%'},
            Emotion.FEARFUL: {'rate': '+15%', 'pitch': '+15Hz', 'volume': '-5%'},
            Emotion.SURPRISED: {'rate': '+10%', 'pitch': '+8Hz', 'volume': '+5%'},
            Emotion.WHISPER: {'rate': '-10%', 'pitch': '-3Hz', 'volume': '-25%'},
            Emotion.EXCITED: {'rate': '+15%', 'pitch': '+10Hz', 'volume': '+10%'},
            Emotion.TENDER: {'rate': '-5%', 'pitch': '-2Hz', 'volume': '-10%'},
            Emotion.SARCASTIC: {'rate': '-5%', 'pitch': '+3Hz', 'volume': '+0%'},
            Emotion.DETERMINED: {'rate': '+5%', 'pitch': '+0Hz', 'volume': '+10%'},
        }

        base = prosody_map.get(emotion, prosody_map[Emotion.NEUTRAL])

        if intensity == 0.5:
            return base

        scale = intensity / 0.5
        result = {}
        for key, value in base.items():
            match = re.match(r'([+-]?\d+)', value)
            if match:
                num = int(match.group(1))
                scaled = int(num * scale)
                unit = value[match.end():]
                result[key] = f'{scaled:+d}{unit}'
            else:
                result[key] = value
        return result


def detect_emotion(text: str) -> Tuple[str, float]:
    """Quick emotion detection. Returns (emotion_name, confidence)."""
    engine = EmotionEngine(use_classifier=False)
    result = engine.analyze(text, use_context=False)
    return result.primary_emotion.value, result.confidence
