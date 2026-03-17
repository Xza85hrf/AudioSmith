"""Tests for audiosmith.f5_tts module."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiosmith.exceptions import TTSError
from audiosmith.f5_tts import F5_SAMPLE_RATE, F5TTS


def _make_wav_bytes(duration: float = 1.0, sr: int = 24000) -> bytes:
    """Create valid WAV bytes for testing."""
    import soundfile as sf

    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestConstants:
    def test_sample_rate_constant(self):
        """F5TTS should have sample rate of 24000."""
        assert F5_SAMPLE_RATE == 24000


class TestF5TTS_Init:
    def test_defaults(self):
        """Test default initialization parameters."""
        tts = F5TTS()
        assert tts.model_name == "f5-tts"
        assert tts.device == "cuda"
        assert tts._checkpoint_path is None
        assert tts._vocab_path is None
        assert tts._model is None
        assert tts._vocoder is None
        assert tts._voice_cache == {}
        assert tts.initialized is False

    def test_custom_model_name(self):
        """Test custom model_name parameter."""
        tts = F5TTS(model_name="f5-polish")
        assert tts.model_name == "f5-polish"

    def test_custom_device(self):
        """Test custom device parameter."""
        tts = F5TTS(device="cpu")
        assert tts.device == "cpu"

    def test_custom_checkpoint_path(self):
        """Test custom checkpoint_path parameter."""
        tts = F5TTS(checkpoint_path="/path/to/checkpoint")
        assert tts._checkpoint_path == "/path/to/checkpoint"

    def test_custom_vocab_path(self):
        """Test custom vocab_path parameter."""
        tts = F5TTS(vocab_path="/path/to/vocab.txt")
        assert tts._vocab_path == "/path/to/vocab.txt"

    def test_init_does_not_load_model(self):
        """Test that model is NOT loaded on init (lazy loading)."""
        tts = F5TTS(checkpoint_path="/path/ckpt")
        assert tts._model is None
        assert tts._vocoder is None
        assert tts.initialized is False

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    @patch("audiosmith.f5_tts._get_dit")
    def test_init_lazy_loads_without_checkpoint(
        self, mock_get_dit, mock_get_f5_infer, mock_get_torch
    ):
        """Test that model is NOT loaded on init without checkpoint."""
        mock_get_torch.return_value = MagicMock()
        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_get_dit.return_value = MagicMock()

        F5TTS()
        # Should not load model yet - lazy loading
        mock_get_f5_infer.return_value[0].assert_not_called()


class TestF5TTS_Synthesize:
    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_synthesize_success(self, mock_get_f5_infer, mock_get_torch):
        """Test successful synthesis returns audio."""
        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_load_model = MagicMock()
        mock_load_vocoder = MagicMock()
        mock_infer_process = MagicMock()

        mock_wav = np.zeros(24000, dtype=np.float32)
        mock_infer_process.return_value = (mock_wav, 24000, None)

        mock_get_f5_infer.return_value = (
            mock_load_model,
            mock_load_vocoder,
            mock_infer_process,
            MagicMock(),
        )

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS(checkpoint_path="/fake/ckpt")
            tts._model = MagicMock()
            tts._vocoder = MagicMock()

            audio, sr = tts.synthesize("Hello world")

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 24000
        assert len(audio) > 0

    def test_synthesize_empty_text(self):
        """Test empty text raises error."""
        tts = F5TTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("")

    def test_synthesize_whitespace_text(self):
        """Test whitespace-only text raises error."""
        tts = F5TTS()
        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("   \n  ")

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_synthesize_with_voice(self, mock_get_f5_infer, mock_get_torch):
        """Test synthesis with voice reference."""
        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_infer_process = MagicMock()
        mock_wav = np.zeros(24000, dtype=np.float32)
        mock_infer_process.return_value = (mock_wav, 24000, None)

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            mock_infer_process,
            MagicMock(),
        )

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS(checkpoint_path="/fake/ckpt")
            tts._model = MagicMock()
            tts._vocoder = MagicMock()
            tts._voice_cache = {
                "my-voice": (np.zeros(24000, dtype=np.float32), "reference text")
            }
            tts.synthesize("Hello", voice="my-voice")

        mock_infer_process.assert_called_once()

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_synthesize_with_language(self, mock_get_f5_infer, mock_get_torch):
        """Test synthesis with language parameter (passed but unused)."""
        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_infer_process = MagicMock()
        mock_wav = np.zeros(24000, dtype=np.float32)
        mock_infer_process.return_value = (mock_wav, 24000, None)

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            mock_infer_process,
            MagicMock(),
        )

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS(checkpoint_path="/fake/ckpt")
            tts._model = MagicMock()
            tts._vocoder = MagicMock()
            audio, sr = tts.synthesize("Hello", language="en")

        assert sr == 24000

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_synthesize_with_speed(self, mock_get_f5_infer, mock_get_torch):
        """Test synthesis with speed parameter."""
        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_infer_process = MagicMock()
        mock_wav = np.zeros(24000, dtype=np.float32)
        mock_infer_process.return_value = (mock_wav, 24000, None)

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            mock_infer_process,
            MagicMock(),
        )

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS(checkpoint_path="/fake/ckpt")
            tts._model = MagicMock()
            tts._vocoder = MagicMock()
            tts.synthesize("Hello", speed=1.5)

        call_kwargs = mock_infer_process.call_args.kwargs
        assert call_kwargs.get("speed") == 1.5

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_synthesize_api_error(self, mock_get_f5_infer, mock_get_torch):
        """Test API errors are properly wrapped."""
        mock_torch = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_infer_process = MagicMock()
        mock_infer_process.side_effect = RuntimeError("Model inference failed")

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            mock_infer_process,
            MagicMock(),
        )

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS(checkpoint_path="/fake/ckpt")
            tts._model = MagicMock()
            tts._vocoder = MagicMock()

            with pytest.raises(TTSError, match="inference failed"):
                tts.synthesize("Hello")

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_synthesize_voice_not_found_still_works(self, mock_get_f5_infer, mock_get_torch):
        """Test that missing voice logs warning but synthesis still succeeds."""
        mock_torch = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_infer_process = MagicMock()
        mock_wav = np.zeros(24000, dtype=np.float32)
        mock_infer_process.return_value = (mock_wav, 24000, None)

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            mock_infer_process,
            MagicMock(),
        )

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS()
            tts._model = MagicMock()
            tts._vocoder = MagicMock()
            # Voice not in cache should warn but continue
            audio, sr = tts.synthesize("Hello", voice="nonexistent")

        assert isinstance(audio, np.ndarray)
        assert sr == 24000


class TestF5TTS_VoiceClone:
    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_clone_voice_from_file(self, mock_get_f5_infer, mock_get_torch, tmp_path):
        """Test cloning voice from audio file."""
        mock_torch = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_preprocess = MagicMock()
        mock_processed_audio = np.zeros(24000 * 5, dtype=np.float32)
        mock_preprocess.return_value = (mock_processed_audio, "reference text")

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_preprocess,
        )

        # Create test audio file
        audio_path = tmp_path / "ref.wav"
        import soundfile as sf

        audio = np.random.randn(24000 * 5).astype(np.float32) * 0.3  # 5s
        sf.write(str(audio_path), audio, 24000)

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS()
            tts._model = MagicMock()
            tts._vocoder = MagicMock()

            result = tts.clone_voice("test-voice", str(audio_path))

        assert result == "test-voice"
        assert "test-voice" in tts._voice_cache

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_clone_voice_with_ref_text(self, mock_get_f5_infer, mock_get_torch):
        """Test cloning voice with reference text."""
        mock_torch = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_preprocess = MagicMock()
        mock_processed_audio = np.zeros(24000 * 5, dtype=np.float32)
        mock_preprocess.return_value = (mock_processed_audio, "custom ref text")

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_preprocess,
        )

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS()
            tts._model = MagicMock()
            tts._vocoder = MagicMock()

            audio = np.random.randn(24000 * 5).astype(np.float32) * 0.3
            result = tts.clone_voice("test-voice", audio, ref_text="custom ref text")

        assert result == "test-voice"
        assert "test-voice" in tts._voice_cache
        call_args = mock_preprocess.call_args
        assert "custom ref text" in str(call_args)

    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_clone_voice_file_not_found(self, mock_get_f5_infer):
        """Test error when audio file not found."""
        mock_get_f5_infer.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        tts = F5TTS()
        with patch.object(F5TTS, "_ensure_model"):
            with pytest.raises(TTSError, match="not found"):
                tts.clone_voice("test", "/nonexistent/path.wav")

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_clone_voice_updates_cache(self, mock_get_f5_infer, mock_get_torch, tmp_path):
        """Test that cloned voice is stored in cache."""
        mock_torch = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_preprocess = MagicMock()
        mock_audio = np.zeros(24000 * 5, dtype=np.float32)
        mock_preprocess.return_value = (mock_audio, "ref text")

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_preprocess,
        )

        audio_path = tmp_path / "ref.wav"
        import soundfile as sf

        sf.write(str(audio_path), mock_audio, 24000)

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS()
            tts._model = MagicMock()
            tts._vocoder = MagicMock()

            tts.clone_voice("my-voice", str(audio_path))

        assert "my-voice" in tts._voice_cache
        cached_audio, cached_text = tts._voice_cache["my-voice"]
        assert isinstance(cached_audio, np.ndarray)
        assert cached_text == "ref text"


class TestF5TTS_GetAvailableVoices:
    def test_get_available_voices_empty(self):
        """Test empty voices returns empty dict."""
        tts = F5TTS()
        tts._voice_cache = {}
        voices = tts.get_available_voices()
        assert voices == {}

    def test_get_available_voices_returns_dict(self):
        """Test returns dict with cloned voices."""
        tts = F5TTS()
        tts._voice_cache = {
            "voice1": (np.zeros(24000, dtype=np.float32), "text1"),
            "voice2": (np.zeros(48000, dtype=np.float32), "text2"),
        }
        voices = tts.get_available_voices()
        assert len(voices) == 2
        assert "voice1" in voices
        assert "voice2" in voices
        assert voices["voice1"]["type"] == "cloned"
        assert voices["voice1"]["engine"] == "f5-tts"


class TestF5TTS_SampleRate:
    def test_sample_rate_property(self):
        """Test sample_rate property returns correct value."""
        tts = F5TTS()
        assert tts.sample_rate == 24000

    def test_sample_rate_matches_constant(self):
        """Test sample_rate property matches F5_SAMPLE_RATE constant."""
        tts = F5TTS()
        assert tts.sample_rate == F5_SAMPLE_RATE


class TestF5TTS_SaveAudio:
    @patch("audiosmith.f5_tts._get_soundfile")
    def test_save_audio_basic(self, mock_get_sf, tmp_path):
        """Test saving audio to file."""
        mock_sf = MagicMock()
        mock_get_sf.return_value = mock_sf

        tts = F5TTS()
        audio = np.zeros(24000, dtype=np.float32)
        output_path = tmp_path / "output.wav"

        result = tts.save_audio(audio, str(output_path))

        mock_sf.write.assert_called_once()
        assert result.suffix == ".wav"

    @patch("audiosmith.f5_tts._get_soundfile")
    def test_save_audio_adds_wav_suffix(self, mock_get_sf, tmp_path):
        """Test save_audio adds .wav suffix if missing."""
        mock_sf = MagicMock()
        mock_get_sf.return_value = mock_sf

        tts = F5TTS()
        audio = np.zeros(24000, dtype=np.float32)
        output_path = tmp_path / "output"

        result = tts.save_audio(audio, str(output_path))

        assert result.suffix == ".wav"

    @patch("audiosmith.f5_tts._get_soundfile")
    def test_save_audio_custom_sample_rate(self, mock_get_sf, tmp_path):
        """Test save_audio with custom sample rate."""
        mock_sf = MagicMock()
        mock_get_sf.return_value = mock_sf

        tts = F5TTS()
        audio = np.zeros(24000, dtype=np.float32)
        output_path = tmp_path / "output.wav"

        tts.save_audio(audio, str(output_path), sample_rate=48000)

        call_args = mock_sf.write.call_args
        assert call_args[0][2] == 48000


class TestF5TTS_Lifecycle:
    def test_cleanup(self):
        """Test cleanup resets state."""
        tts = F5TTS()
        tts._model = MagicMock()
        tts._voice_cache = {"v1": (np.zeros(100, dtype=np.float32), "text")}
        tts._vocoder = MagicMock()
        tts.initialized = True

        with patch("audiosmith.f5_tts.gc"):
            tts.cleanup()

        assert tts._model is None
        assert tts._vocoder is None
        assert tts._voice_cache == {}
        assert tts.initialized is False

    def test_context_manager(self):
        """Test context manager protocol."""
        tts = F5TTS()
        tts._model = MagicMock()

        with tts:
            pass

        assert tts._model is None

    def test_context_manager_with_exception(self):
        """Test context manager cleans up even on exception."""
        tts = F5TTS()
        tts._model = MagicMock()

        try:
            with tts:
                raise ValueError("test")
        except ValueError:
            pass

        assert tts._model is None

    @patch("audiosmith.f5_tts._get_torch")
    def test_device_fallback_cpu(self, mock_get_torch):
        """Test device fallback when cuda unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_get_torch.return_value = mock_torch

        tts = F5TTS(device="cuda")
        assert tts.device == "cuda"


class TestF5TTS_LazyLoading:
    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_model_not_reloaded_on_subsequent_calls(
        self, mock_get_f5_infer, mock_get_torch
    ):
        """Test model is not reloaded on subsequent synthesize calls."""
        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_infer_process = MagicMock()
        mock_wav = np.zeros(24000, dtype=np.float32)
        mock_infer_process.return_value = (mock_wav, 24000, None)

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            mock_infer_process,
            MagicMock(),
        )

        tts = F5TTS(checkpoint_path="/fake/ckpt")
        tts._model = MagicMock()
        tts._vocoder = MagicMock()

        tts.synthesize("Hello")
        first_model = tts._model

        tts.synthesize("World")
        assert tts._model is first_model


class TestF5TTS_ImportErrors:
    def test_torch_import_error(self):
        """Test proper error when torch import fails."""
        with patch(
            "audiosmith.f5_tts._get_torch",
            side_effect=TTSError("torch not installed", error_code="TORCH_IMPORT_ERR"),
        ):
            tts = F5TTS()
            with pytest.raises(TTSError, match="torch not installed"):
                tts._ensure_model()

    def test_f5_infer_import_error(self):
        """Test proper error when f5-tts import fails."""
        with patch("audiosmith.f5_tts._get_torch", return_value=MagicMock()):
            with patch(
                "audiosmith.f5_tts._get_f5_infer",
                side_effect=TTSError(
                    "f5-tts not installed", error_code="F5_IMPORT_ERR",
                ),
            ):
                tts = F5TTS(checkpoint_path="/fake")
                with pytest.raises(TTSError, match="f5-tts not installed"):
                    tts._ensure_model()

    def test_dit_import_error(self):
        """Test proper error when DiT import fails."""
        with patch("audiosmith.f5_tts._get_torch", return_value=MagicMock()):
            with patch(
                "audiosmith.f5_tts._get_f5_infer",
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()),
            ):
                with patch(
                    "audiosmith.f5_tts._get_dit",
                    side_effect=TTSError(
                        "DiT not found", error_code="F5_IMPORT_ERR",
                    ),
                ):
                    tts = F5TTS(checkpoint_path="/fake")
                    with pytest.raises(TTSError, match="DiT not found"):
                        tts._ensure_model()


class TestF5TTS_Additional:
    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    def test_synthesize_returns_correct_sample_rate(
        self, mock_get_f5_infer, mock_get_torch
    ):
        """Test synthesize always returns 24000 sample rate."""
        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_get_torch.return_value = mock_torch

        mock_infer_process = MagicMock()
        mock_wav = np.zeros(48000, dtype=np.float32)
        mock_infer_process.return_value = (mock_wav, 24000, None)

        mock_get_f5_infer.return_value = (
            MagicMock(),
            MagicMock(),
            mock_infer_process,
            MagicMock(),
        )

        with patch.object(F5TTS, "_ensure_model"):
            tts = F5TTS(checkpoint_path="/fake/ckpt")
            tts._model = MagicMock()
            tts._vocoder = MagicMock()
            audio, sr = tts.synthesize("Test")

        assert sr == 24000

    def test_voice_cache_initialized_empty(self):
        """Test voice cache is initialized as empty dict."""
        tts = F5TTS()
        assert tts._voice_cache == {}
        assert tts.initialized is False

    def test_model_name_attribute(self):
        """Test model_name attribute is set correctly."""
        tts = F5TTS(model_name="CustomModel")
        assert tts.model_name == "CustomModel"

    @patch("audiosmith.f5_tts._get_torch")
    @patch("audiosmith.f5_tts._get_f5_infer")
    @patch("audiosmith.f5_tts._get_dit")
    def test_vocoder_loaded(self, mock_get_dit, mock_get_f5_infer, mock_get_torch):
        """Test that vocoder is loaded alongside model."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_get_torch.return_value = mock_torch

        mock_load_model = MagicMock()
        mock_load_vocoder = MagicMock()

        mock_get_f5_infer.return_value = (
            mock_load_model,
            mock_load_vocoder,
            MagicMock(),
            MagicMock(),
        )

        mock_dit = MagicMock()
        mock_get_dit.return_value = mock_dit

        with patch("audiosmith.f5_tts.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            mock_vocab_path = MagicMock()
            mock_vocab_path.exists.return_value = False
            mock_path_instance.with_suffix.return_value = mock_vocab_path
            type(mock_path_instance).suffix = property(lambda self: "")

            tts = F5TTS(checkpoint_path="/fake/ckpt", vocab_path="/fake/vocab")
            tts._ensure_model()

        mock_load_model.assert_called_once()
        mock_load_vocoder.assert_called_once()
        assert tts._vocoder is not None
