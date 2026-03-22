"""Tests for audiosmith.pipeline_config module."""


from audiosmith.pipeline_config import (
    ENGINE_PP_PRESETS,
    LANGUAGE_PP_OVERRIDES,
)


class TestPipelineConfig:
    """Test pipeline configuration constants."""

    def test_engine_pp_presets_exists(self):
        """Test that ENGINE_PP_PRESETS is defined and accessible."""
        assert ENGINE_PP_PRESETS is not None
        assert isinstance(ENGINE_PP_PRESETS, dict)

    def test_engine_pp_presets_has_required_engines(self):
        """Test that ENGINE_PP_PRESETS contains expected TTS engines."""
        required = {'piper', 'chatterbox', 'fish', 'qwen3', 'f5', 'indextts',
                    'cosyvoice', 'orpheus'}
        assert set(ENGINE_PP_PRESETS.keys()) == required

    def test_engine_pp_presets_structure(self):
        """Test that each engine preset is a dict with boolean/numeric values."""
        for engine, preset in ENGINE_PP_PRESETS.items():
            assert isinstance(preset, dict)
            # Check that values are bools or floats
            for key, value in preset.items():
                assert isinstance(value, (bool, int, float, str))

    def test_language_pp_overrides_exists(self):
        """Test that LANGUAGE_PP_OVERRIDES is defined and accessible."""
        assert LANGUAGE_PP_OVERRIDES is not None
        assert isinstance(LANGUAGE_PP_OVERRIDES, dict)

    def test_language_pp_overrides_has_polish(self):
        """Test that LANGUAGE_PP_OVERRIDES contains Polish config."""
        assert 'pl' in LANGUAGE_PP_OVERRIDES

    def test_polish_override_structure(self):
        """Test that Polish overrides have expected keys."""
        pl_config = LANGUAGE_PP_OVERRIDES['pl']
        assert isinstance(pl_config, dict)
        assert 'spectral_intensity' in pl_config
        assert 'enable_spectral_matching' in pl_config
        assert 'enable_dynamics' in pl_config
        assert 'enable_breath' in pl_config
        assert 'enable_normalize' in pl_config
        assert 'target_rms_adaptive' in pl_config
        assert 'target_rms' in pl_config
