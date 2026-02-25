"""Tests for audiosmith.retry module."""

import pytest
from unittest.mock import patch, MagicMock
from audiosmith.retry import retry, with_fallback, is_transient_error, RetryError


class TestRetryDecorator:
    @patch('audiosmith.retry.time.sleep')
    def test_success_first_attempt(self, mock_sleep):
        @retry(max_retries=3)
        def func():
            return "success"
        assert func() == "success"
        mock_sleep.assert_not_called()

    @patch('audiosmith.retry.time.sleep')
    def test_success_after_failures(self, mock_sleep):
        counter = [0]

        @retry(max_retries=2, jitter=False)
        def func():
            counter[0] += 1
            if counter[0] < 3:
                raise IOError("fail")
            return "ok"

        assert func() == "ok"
        assert counter[0] == 3

    @patch('audiosmith.retry.time.sleep')
    def test_exhausted_raises_retry_error(self, mock_sleep):
        @retry(max_retries=2, jitter=False)
        def func():
            raise IOError("always fails")

        with pytest.raises(RetryError) as exc_info:
            func()
        assert exc_info.value.attempt_count == 3

    @patch('audiosmith.retry.time.sleep')
    def test_non_matching_exception_propagates(self, mock_sleep):
        @retry(max_retries=2, exceptions=(IOError,))
        def func():
            raise ValueError("wrong type")

        with pytest.raises(ValueError):
            func()

    @patch('audiosmith.retry.time.sleep')
    def test_backoff_timing(self, mock_sleep):
        counter = [0]

        @retry(max_retries=2, delay=1.0, backoff=2.0, jitter=False)
        def func():
            counter[0] += 1
            if counter[0] < 3:
                raise IOError("fail")
            return "ok"

        func()
        assert mock_sleep.call_count == 2
        # First retry: delay * backoff^0 = 1.0
        # Second retry: delay * backoff^1 = 2.0
        calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert calls[0] == pytest.approx(1.0)
        assert calls[1] == pytest.approx(2.0)

    def test_retry_error_has_original(self):
        err = RetryError("msg", attempt_count=3, original_error=IOError("orig"))
        assert err.attempt_count == 3
        assert isinstance(err.original_error, IOError)


class TestWithFallback:
    def test_primary_succeeds(self):
        result = with_fallback(lambda: "primary", lambda: "fallback")
        assert result == "primary"

    def test_uses_fallback(self):
        def primary():
            raise IOError("fail")
        result = with_fallback(primary, lambda: "fallback")
        assert result == "fallback"

    def test_both_fail(self):
        def primary():
            raise IOError("primary fail")
        def fallback():
            raise ValueError("fallback fail")
        with pytest.raises(ValueError):
            with_fallback(primary, fallback)

    def test_passes_args(self):
        def primary(x, y=1):
            return x + y
        result = with_fallback(primary, lambda x, y=1: 0, 5, y=10)
        assert result == 15


class TestIsTransientError:
    def test_timeout(self):
        assert is_transient_error(TimeoutError()) is True

    def test_connection(self):
        assert is_transient_error(ConnectionError()) is True

    def test_io_error(self):
        assert is_transient_error(IOError()) is True

    def test_value_error(self):
        assert is_transient_error(ValueError()) is False

    def test_file_not_found(self):
        assert is_transient_error(FileNotFoundError()) is False

    def test_type_error(self):
        assert is_transient_error(TypeError()) is False
