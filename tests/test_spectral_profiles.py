"""Tests for audiosmith.spectral_profiles module."""

import pytest

from audiosmith.spectral_profiles import (
    EmotionProfile,
    BAND_EDGES,
    get_profile,
    get_language_modifier,
    list_emotions,
    LANGUAGE_SPECTRAL_MODIFIERS,
)


class TestBandEdges:
    """Test the band edge definitions."""

    def test_band_edges_count(self):
        """Test that there are exactly 9 band edges."""
        assert len(BAND_EDGES) == 9

    def test_band_edges_ordered(self):
        """Test that band edges are in ascending order."""
        for i in range(len(BAND_EDGES) - 1):
            assert BAND_EDGES[i] < BAND_EDGES[i + 1]

    def test_band_edges_start_at_zero(self):
        """Test that first band edge is 0."""
        assert BAND_EDGES[0] == 0

    def test_band_edges_specific_values(self):
        """Test known band edge values."""
        assert BAND_EDGES[0] == 0
        assert BAND_EDGES[1] == 150
        assert BAND_EDGES[2] == 350
        assert BAND_EDGES[3] == 700
        assert BAND_EDGES[4] == 1400
        assert BAND_EDGES[5] == 2800
        assert BAND_EDGES[6] == 4000
        assert BAND_EDGES[7] == 6000
        assert BAND_EDGES[8] == 8000


class TestEmotionProfileStructure:
    """Test EmotionProfile dataclass structure."""

    def test_emotion_profile_creation(self):
        """Test creating an EmotionProfile."""
        profile = EmotionProfile(
            emotion='test_emotion',
            centroid_hz=2500.0,
            target_rms=0.15,
            target_dynamics_db=16.0,
            band_energies_db=(0.0, -2.0, -3.5, -5.0, -8.0, -12.0, -18.0, -24.0, -30.0),
            brightness=0.6,
        )
        assert profile.emotion == 'test_emotion'
        assert profile.centroid_hz == 2500.0
        assert profile.target_rms == 0.15

    def test_emotion_profile_frozen(self):
        """Test that EmotionProfile is immutable."""
        profile = EmotionProfile(
            emotion='test',
            centroid_hz=2500.0,
            target_rms=0.15,
            target_dynamics_db=16.0,
            band_energies_db=(0.0, -2.0, -3.5, -5.0, -8.0, -12.0, -18.0, -24.0, -30.0),
            brightness=0.6,
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            profile.centroid_hz = 3000.0

    def test_emotion_profile_band_energies_tuple(self):
        """Test that band_energies_db is a tuple with 9 values."""
        profile = EmotionProfile(
            emotion='test',
            centroid_hz=2500.0,
            target_rms=0.15,
            target_dynamics_db=16.0,
            band_energies_db=(0.0, -2.0, -3.5, -5.0, -8.0, -12.0, -18.0, -24.0, -30.0),
            brightness=0.6,
        )
        assert isinstance(profile.band_energies_db, tuple)
        assert len(profile.band_energies_db) == 9


class TestGetProfile:
    """Test the get_profile function."""

    def test_get_angry_profile(self):
        """Test retrieving angry emotion profile."""
        profile = get_profile('angry')
        assert profile.emotion == 'angry'
        assert isinstance(profile.centroid_hz, float)

    def test_get_neutral_profile(self):
        """Test retrieving neutral emotion profile."""
        profile = get_profile('neutral')
        assert profile.emotion == 'neutral'
        assert isinstance(profile.centroid_hz, float)

    def test_get_sad_profile(self):
        """Test retrieving sad emotion profile."""
        profile = get_profile('sad')
        assert profile.emotion == 'sad'
        assert isinstance(profile.centroid_hz, float)

    def test_get_excited_profile(self):
        """Test retrieving excited emotion profile."""
        profile = get_profile('excited')
        assert profile.emotion == 'excited'
        assert isinstance(profile.centroid_hz, float)

    def test_get_whisper_profile(self):
        """Test retrieving whisper emotion profile."""
        profile = get_profile('whisper')
        assert profile.emotion == 'whisper'
        assert isinstance(profile.centroid_hz, float)

    def test_get_happy_profile(self):
        """Test retrieving happy emotion profile."""
        profile = get_profile('happy')
        assert profile.emotion == 'happy'
        assert isinstance(profile.centroid_hz, float)

    def test_get_fearful_profile(self):
        """Test retrieving fearful emotion profile."""
        profile = get_profile('fearful')
        assert profile.emotion == 'fearful'
        assert isinstance(profile.centroid_hz, float)

    def test_get_unknown_emotion_returns_neutral(self):
        """Test that unknown emotion returns neutral profile."""
        profile = get_profile('unknown_emotion')
        assert profile.emotion == 'neutral'

    def test_get_case_sensitive(self):
        """Test that emotion lookup is case-sensitive."""
        # 'ANGRY' should not match 'angry'
        profile = get_profile('ANGRY')
        # Should return neutral (default) since 'ANGRY' != 'angry'
        assert profile.emotion == 'neutral'


class TestEmotionProfileValidation:
    """Test that emotion profiles have valid values."""

    def test_angry_profile_valid(self):
        """Test angry profile has valid ranges."""
        profile = get_profile('angry')
        assert 2000 <= profile.centroid_hz <= 3000
        assert 0.1 <= profile.target_rms <= 0.2
        assert 10 <= profile.target_dynamics_db <= 20
        assert 0.0 <= profile.brightness <= 1.0

    def test_neutral_profile_valid(self):
        """Test neutral profile has valid ranges."""
        profile = get_profile('neutral')
        assert 2000 <= profile.centroid_hz <= 3000
        assert 0.1 <= profile.target_rms <= 0.2
        assert 10 <= profile.target_dynamics_db <= 20
        assert 0.0 <= profile.brightness <= 1.0

    def test_all_profiles_have_9_bands(self):
        """Test all emotion profiles have exactly 9 band energy values."""
        emotions = list_emotions()
        for emotion in emotions:
            profile = get_profile(emotion)
            assert len(profile.band_energies_db) == 9

    def test_all_profiles_band_energies_negative(self):
        """Test that band energies are non-positive (relative dB values)."""
        emotions = list_emotions()
        for emotion in emotions:
            profile = get_profile(emotion)
            # First band is reference (0), rest should be negative
            assert profile.band_energies_db[0] == 0.0
            for i in range(1, 9):
                assert profile.band_energies_db[i] <= 0.0

    def test_whisper_lowest_rms(self):
        """Test that whisper has lower RMS than other emotions."""
        whisper = get_profile('whisper')
        angry = get_profile('angry')
        excited = get_profile('excited')
        happy = get_profile('happy')
        # Whisper should have quieter RMS
        assert whisper.target_rms < angry.target_rms
        assert whisper.target_rms < excited.target_rms
        assert whisper.target_rms < happy.target_rms

    def test_excited_highest_rms(self):
        """Test that excited has higher RMS than calm emotions."""
        excited = get_profile('excited')
        neutral = get_profile('neutral')
        whisper = get_profile('whisper')
        # Excited should be louder
        assert excited.target_rms > neutral.target_rms
        assert excited.target_rms > whisper.target_rms

    def test_brightness_whisper_high(self):
        """Test that whisper has high brightness."""
        whisper = get_profile('whisper')
        neutral = get_profile('neutral')
        # Whisper should be brighter
        assert whisper.brightness > neutral.brightness

    def test_centroid_whisper_high(self):
        """Test that whisper has high spectral centroid."""
        whisper = get_profile('whisper')
        angry = get_profile('angry')
        neutral = get_profile('neutral')
        # Whisper should have higher centroid
        assert whisper.centroid_hz > neutral.centroid_hz


class TestListEmotions:
    """Test the list_emotions function."""

    def test_list_emotions_returns_list(self):
        """Test that list_emotions returns a list."""
        emotions = list_emotions()
        assert isinstance(emotions, list)

    def test_list_emotions_not_empty(self):
        """Test that list_emotions is not empty."""
        emotions = list_emotions()
        assert len(emotions) > 0

    def test_list_emotions_contains_expected(self):
        """Test that list_emotions contains expected emotions."""
        emotions = list_emotions()
        assert 'angry' in emotions
        assert 'neutral' in emotions
        assert 'sad' in emotions
        assert 'excited' in emotions
        assert 'whisper' in emotions
        assert 'happy' in emotions
        assert 'fearful' in emotions

    def test_list_emotions_count(self):
        """Test that exactly 7 emotions are defined."""
        emotions = list_emotions()
        assert len(emotions) == 7

    def test_list_emotions_all_accessible(self):
        """Test that all listed emotions can be retrieved."""
        emotions = list_emotions()
        for emotion in emotions:
            profile = get_profile(emotion)
            assert profile.emotion == emotion


class TestGetLanguageModifier:
    """Test the get_language_modifier function."""

    def test_get_polish_modifier(self):
        """Test retrieving Polish language modifier."""
        modifier = get_language_modifier('pl')
        assert isinstance(modifier, dict)
        assert len(modifier) > 0

    def test_get_unsupported_language_returns_empty(self):
        """Test that unsupported language returns empty dict."""
        modifier = get_language_modifier('fr')
        assert isinstance(modifier, dict)
        assert len(modifier) == 0

    def test_get_none_language_returns_empty(self):
        """Test that None language returns empty dict."""
        modifier = get_language_modifier(None)
        assert isinstance(modifier, dict)
        assert len(modifier) == 0

    def test_polish_modifier_has_band_boosts(self):
        """Test that Polish modifier has band boost values."""
        modifier = get_language_modifier('pl')
        assert 'band_0_boost_db' in modifier
        assert 'band_1_boost_db' in modifier
        assert 'band_2_boost_db' in modifier

    def test_polish_modifier_has_brightness_offset(self):
        """Test that Polish modifier has brightness offset."""
        modifier = get_language_modifier('pl')
        assert 'brightness_offset' in modifier

    def test_polish_modifier_has_centroid_offset(self):
        """Test that Polish modifier has centroid offset."""
        modifier = get_language_modifier('pl')
        assert 'centroid_offset_hz' in modifier

    def test_polish_modifier_values_are_numeric(self):
        """Test that Polish modifier values are numbers."""
        modifier = get_language_modifier('pl')
        for key, value in modifier.items():
            assert isinstance(value, (int, float))


class TestPolishLanguageModifier:
    """Test Polish language modifier specifics."""

    def test_polish_band_cuts(self):
        """Test that Polish modifier has low-band cuts."""
        modifier = get_language_modifier('pl')
        assert modifier['band_0_boost_db'] < 0  # Cut sub-bass
        assert modifier['band_1_boost_db'] < 0  # Cut bass
        assert modifier['band_2_boost_db'] < 0  # Cut low-mid

    def test_polish_band_lifts(self):
        """Test that Polish modifier has high-band lifts."""
        modifier = get_language_modifier('pl')
        assert modifier['band_3_boost_db'] > 0  # Lift mid
        assert modifier['band_4_boost_db'] > 0  # Lift upper-mid
        assert modifier['band_5_boost_db'] > 0  # Lift presence

    def test_polish_brightness_increase(self):
        """Test that Polish modifier increases brightness."""
        modifier = get_language_modifier('pl')
        assert modifier['brightness_offset'] > 0

    def test_polish_centroid_increase(self):
        """Test that Polish modifier shifts centroid up."""
        modifier = get_language_modifier('pl')
        assert modifier['centroid_offset_hz'] > 0

    def test_polish_spectral_intensity_cap(self):
        """Test that Polish modifier caps intensity."""
        modifier = get_language_modifier('pl')
        assert 'spectral_intensity_cap' in modifier
        assert 0 < modifier['spectral_intensity_cap'] <= 1.0

    def test_polish_correction_clip_max(self):
        """Test that Polish modifier has clip limit."""
        modifier = get_language_modifier('pl')
        assert 'correction_clip_max' in modifier
        assert modifier['correction_clip_max'] > 0


class TestLanguageSpectralModifiersStructure:
    """Test LANGUAGE_SPECTRAL_MODIFIERS constant."""

    def test_language_modifiers_is_dict(self):
        """Test that LANGUAGE_SPECTRAL_MODIFIERS is a dict."""
        assert isinstance(LANGUAGE_SPECTRAL_MODIFIERS, dict)

    def test_language_modifiers_contains_polish(self):
        """Test that Polish is in language modifiers."""
        assert 'pl' in LANGUAGE_SPECTRAL_MODIFIERS

    def test_language_modifiers_polish_is_dict(self):
        """Test that Polish entry is a dict."""
        assert isinstance(LANGUAGE_SPECTRAL_MODIFIERS['pl'], dict)


class TestEmotionProfileComparisons:
    """Test comparing different emotion profiles."""

    def test_angry_vs_neutral_centroid(self):
        """Test that angry has higher centroid than neutral."""
        angry = get_profile('angry')
        neutral = get_profile('neutral')
        assert angry.centroid_hz > neutral.centroid_hz

    def test_excited_vs_sad_rms(self):
        """Test that excited has higher RMS than sad."""
        excited = get_profile('excited')
        sad = get_profile('sad')
        assert excited.target_rms > sad.target_rms

    def test_angry_vs_fearful_brightness(self):
        """Test anger vs fear brightness differences."""
        angry = get_profile('angry')
        fearful = get_profile('fearful')
        # Both are high energy, so similar brightness
        assert abs(angry.brightness - fearful.brightness) < 0.15

    def test_sad_vs_happy_dynamics(self):
        """Test sad vs happy dynamics difference."""
        sad = get_profile('sad')
        happy = get_profile('happy')
        # Sad should have higher dynamics (more variable)
        assert sad.target_dynamics_db > happy.target_dynamics_db


class TestEmotionProfileBandEnergyPatterns:
    """Test band energy distribution patterns."""

    def test_band_energies_decreasing(self):
        """Test that band energies generally decrease toward high freq."""
        angry = get_profile('angry')
        bands = angry.band_energies_db
        # Each band should be lower than or equal to previous
        for i in range(1, len(bands)):
            assert bands[i] <= bands[i - 1]

    def test_neutral_band_pattern(self):
        """Test neutral emotion band energy pattern."""
        neutral = get_profile('neutral')
        bands = neutral.band_energies_db
        # Reference band should be 0
        assert bands[0] == 0.0
        # Higher frequencies should be increasingly attenuated
        assert bands[-1] < bands[-2] < bands[-3]

    def test_whisper_band_sibilance_lift(self):
        """Test whisper has emphasize sibilance bands."""
        whisper = get_profile('whisper')
        neutral = get_profile('neutral')
        # Whisper sibilance bands (5, 6) should be less attenuated
        assert whisper.band_energies_db[5] > neutral.band_energies_db[5]
        assert whisper.band_energies_db[6] > neutral.band_energies_db[6]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_get_profile_empty_string(self):
        """Test get_profile with empty string returns neutral."""
        profile = get_profile('')
        assert profile.emotion == 'neutral'

    def test_get_language_modifier_empty_string(self):
        """Test get_language_modifier with empty string."""
        modifier = get_language_modifier('')
        assert isinstance(modifier, dict)

    def test_multiple_get_profile_calls_same_instance(self):
        """Test that get_profile returns consistent profiles."""
        profile1 = get_profile('angry')
        profile2 = get_profile('angry')
        assert profile1.emotion == profile2.emotion
        assert profile1.centroid_hz == profile2.centroid_hz


class TestProfileDataTypes:
    """Test that profile data has correct types."""

    def test_profile_emotion_is_string(self):
        """Test that emotion field is a string."""
        profile = get_profile('angry')
        assert isinstance(profile.emotion, str)

    def test_profile_centroid_is_float(self):
        """Test that centroid_hz is a float."""
        profile = get_profile('angry')
        assert isinstance(profile.centroid_hz, float)

    def test_profile_rms_is_float(self):
        """Test that target_rms is a float."""
        profile = get_profile('angry')
        assert isinstance(profile.target_rms, float)

    def test_profile_dynamics_is_float(self):
        """Test that target_dynamics_db is a float."""
        profile = get_profile('angry')
        assert isinstance(profile.target_dynamics_db, float)

    def test_profile_brightness_is_float(self):
        """Test that brightness is a float."""
        profile = get_profile('angry')
        assert isinstance(profile.brightness, float)

    def test_profile_band_energies_are_floats(self):
        """Test that all band energies are floats."""
        profile = get_profile('angry')
        for value in profile.band_energies_db:
            assert isinstance(value, (int, float))


class TestRangeValidations:
    """Test value ranges across all profiles."""

    def test_all_centroid_in_range(self):
        """Test all emotions have centroid in reasonable range."""
        emotions = list_emotions()
        for emotion in emotions:
            profile = get_profile(emotion)
            assert 1500 <= profile.centroid_hz <= 3500

    def test_all_rms_in_range(self):
        """Test all emotions have RMS in reasonable range."""
        emotions = list_emotions()
        for emotion in emotions:
            profile = get_profile(emotion)
            assert 0.08 <= profile.target_rms <= 0.2

    def test_all_dynamics_in_range(self):
        """Test all emotions have dynamics in reasonable range."""
        emotions = list_emotions()
        for emotion in emotions:
            profile = get_profile(emotion)
            assert 10 <= profile.target_dynamics_db <= 20

    def test_all_brightness_in_range(self):
        """Test all emotions have brightness between 0 and 1."""
        emotions = list_emotions()
        for emotion in emotions:
            profile = get_profile(emotion)
            assert 0.0 <= profile.brightness <= 1.0
