"""Retry decorator with exponential backoff and error classification."""

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

from audiosmith.exceptions import ProcessingError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryError(ProcessingError):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, attempt_count: int = 0, original_error: Optional[Exception] = None):
        super().__init__(message, error_code="RETRY_ERR", original_error=original_error)
        self.attempt_count = attempt_count


def is_transient_error(exc: Exception) -> bool:
    """Classify whether an error is transient (worth retrying).

    Returns True for TimeoutError, ConnectionError, IOError, OSError.
    Returns False for ValueError, TypeError, FileNotFoundError.
    """
    permanent_types = (FileNotFoundError, PermissionError, ValueError, TypeError)
    if isinstance(exc, permanent_types):
        return False
    transient_types = (
        TimeoutError,
        ConnectionError,
        BrokenPipeError,
        IOError,
        OSError,
        InterruptedError,
    )
    return isinstance(exc, transient_types)


def retry(
    max_retries: int = 2,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        wait = delay * (backoff ** attempt)
                        if jitter:
                            wait += random.uniform(0, wait * 0.1)
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.2fs...",
                            attempt + 1, max_retries + 1, func.__name__, e, wait,
                        )
                        time.sleep(wait)
            raise RetryError(
                f"Failed after {max_retries + 1} attempts",
                attempt_count=max_retries + 1,
                original_error=last_exc,
            )
        return wrapper
    return decorator


def with_fallback(
    primary_fn: Callable[..., T],
    fallback_fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute primary function, falling back to secondary on failure."""
    try:
        return primary_fn(*args, **kwargs)
    except Exception as e:
        logger.warning("Primary %s failed: %s. Using fallback.", primary_fn.__name__, e)
        return fallback_fn(*args, **kwargs)
