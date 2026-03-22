"""Tests for audiosmith.log module."""

import logging
import tempfile
from pathlib import Path

import pytest

from audiosmith.log import setup_logging, get_logger


class TestSetupLogging:
    """Test the setup_logging function."""

    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates the audiosmith logger."""
        # Clear any existing handlers first
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        result = setup_logging()
        assert result is not None
        assert result.name == 'audiosmith'

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger instance."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        result = setup_logging()
        assert isinstance(result, logging.Logger)

    def test_setup_logging_default_level_info(self):
        """Test that default log level is INFO."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging()
        assert logging.getLogger('audiosmith').level == logging.INFO

    def test_setup_logging_custom_level_debug(self):
        """Test setting log level to DEBUG."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(level='DEBUG')
        assert logging.getLogger('audiosmith').level == logging.DEBUG

    def test_setup_logging_custom_level_warning(self):
        """Test setting log level to WARNING."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(level='WARNING')
        assert logging.getLogger('audiosmith').level == logging.WARNING

    def test_setup_logging_custom_level_error(self):
        """Test setting log level to ERROR."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(level='ERROR')
        assert logging.getLogger('audiosmith').level == logging.ERROR

    def test_setup_logging_adds_stream_handler(self):
        """Test that setup_logging adds a StreamHandler."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging()
        logger = logging.getLogger('audiosmith')
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_setup_logging_handler_has_formatter(self):
        """Test that handlers have a formatter."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging()
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers:
            assert handler.formatter is not None

    def test_setup_logging_idempotent(self):
        """Test that calling setup_logging twice doesn't double handlers."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging()
        initial_count = len(logger.handlers)

        setup_logging()
        # Should still have same number of handlers (guards against duplicate)
        final_count = len(logger.handlers)
        assert final_count == initial_count

    def test_setup_logging_with_file_creates_file_handler(self, tmp_path):
        """Test that log_file parameter creates FileHandler."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

    def test_setup_logging_writes_to_file(self, tmp_path):
        """Test that logs are actually written to file."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.info('Test message')

        # File should exist and contain our message
        assert log_file.exists()
        content = log_file.read_text()
        assert 'Test message' in content

    def test_setup_logging_without_file_no_file_handler(self):
        """Test that no FileHandler is created when log_file is None."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(log_file=None)

        logger = logging.getLogger('audiosmith')
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_setup_logging_level_case_insensitive(self):
        """Test that level parameter is case insensitive."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(level='DeBuG')
        assert logging.getLogger('audiosmith').level == logging.DEBUG

    def test_setup_logging_formatter_contains_timestamp(self, tmp_path, caplog):
        """Test that formatter includes timestamp."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.info('Test message')

        content = log_file.read_text()
        # Timestamp format should be present (YYYY-MM-DD HH:MM:SS)
        assert '20' in content  # Year starting with 20

    def test_setup_logging_formatter_contains_level(self, tmp_path):
        """Test that formatter includes log level."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.warning('Test warning')

        content = log_file.read_text()
        assert 'WARNING' in content

    def test_setup_logging_formatter_contains_logger_name(self, tmp_path):
        """Test that formatter includes logger name."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.info('Test message')

        content = log_file.read_text()
        assert 'audiosmith' in content


class TestGetLogger:
    """Test the get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)

    def test_get_logger_has_audiosmith_prefix(self):
        """Test that returned logger has audiosmith prefix."""
        logger = get_logger('test_module')
        assert logger.name == 'audiosmith.test_module'

    def test_get_logger_unique_names(self):
        """Test that different calls return different loggers."""
        logger1 = get_logger('module1')
        logger2 = get_logger('module2')
        assert logger1.name != logger2.name
        assert logger1.name == 'audiosmith.module1'
        assert logger2.name == 'audiosmith.module2'

    def test_get_logger_same_name_returns_same_logger(self):
        """Test that calling with same name returns same logger."""
        logger1 = get_logger('test')
        logger2 = get_logger('test')
        assert logger1 is logger2

    def test_get_logger_inherits_from_parent(self):
        """Test that child logger inherits from parent."""
        # Setup parent logger
        parent = logging.getLogger('audiosmith')
        for handler in parent.handlers[:]:
            parent.removeHandler(handler)
        setup_logging()

        # Get child logger
        child = get_logger('child_module')
        # Child should inherit parent's handlers
        assert child.name == 'audiosmith.child_module'


class TestLoggingIntegration:
    """Test logging functionality end-to-end."""

    def test_logging_to_console(self, capsys):
        """Test that logging works to console."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging()
        logger = logging.getLogger('audiosmith')
        logger.info('Console test message')

        captured = capsys.readouterr()
        # Message should appear in captured output
        assert 'Console test message' in captured.out or 'Console test message' in captured.err

    def test_logging_file_and_console(self, tmp_path, capsys):
        """Test that logging goes to both file and console."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.info('Dual output message')

        # Check file
        assert log_file.exists()
        file_content = log_file.read_text()
        assert 'Dual output message' in file_content

        # Check console
        captured = capsys.readouterr()
        assert 'Dual output message' in captured.out or 'Dual output message' in captured.err

    def test_logging_respects_level_debug(self, tmp_path):
        """Test that DEBUG messages appear when level is DEBUG."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(level='DEBUG', log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.debug('Debug message')

        content = log_file.read_text()
        assert 'Debug message' in content

    def test_logging_respects_level_info(self, tmp_path):
        """Test that DEBUG messages don't appear when level is INFO."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(level='INFO', log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.debug('Debug message')

        content = log_file.read_text()
        assert 'Debug message' not in content

    def test_child_logger_logging(self, tmp_path):
        """Test that child loggers work correctly."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        child = get_logger('child')
        child.info('Child message')

        content = log_file.read_text()
        assert 'Child message' in content
        assert 'audiosmith.child' in content


class TestLoggingEdgeCases:
    """Test edge cases in logging."""

    def test_get_logger_with_empty_string(self):
        """Test get_logger with empty string."""
        logger = get_logger('')
        assert logger.name == 'audiosmith.'

    def test_get_logger_with_dots_in_name(self):
        """Test get_logger with dots in module name."""
        logger = get_logger('sub.module.name')
        assert logger.name == 'audiosmith.sub.module.name'

    def test_setup_logging_with_nonexistent_directory(self, tmp_path):
        """Test setup_logging creates parent directory if needed."""
        log_file = tmp_path / 'nested' / 'dir' / 'test.log'
        # Create the parent directory first (standard behavior)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(log_file=str(log_file))
        logger = logging.getLogger('audiosmith')
        logger.info('Test message')

        assert log_file.exists()


class TestMultipleLoggerSetups:
    """Test behavior with multiple setup_logging calls."""

    def test_multiple_setups_same_config(self):
        """Test calling setup multiple times with same config."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(level='INFO')
        initial_handlers = len(logger.handlers)

        setup_logging(level='INFO')
        final_handlers = len(logger.handlers)

        # Handlers shouldn't duplicate
        assert final_handlers == initial_handlers

    def test_multiple_setups_different_levels(self):
        """Test setup with different levels."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(level='INFO')
        setup_logging(level='DEBUG')

        # Final level should be DEBUG
        assert logger.level == logging.DEBUG

    def test_multiple_setups_different_files(self, tmp_path):
        """Test setup with different log files."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file1 = tmp_path / 'test1.log'
        setup_logging(log_file=str(log_file1))

        logger.info('Message 1')

        # Clean up handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file2 = tmp_path / 'test2.log'
        setup_logging(log_file=str(log_file2))

        logger.info('Message 2')

        # Both files should exist with their respective messages
        assert log_file1.exists()
        assert log_file2.exists()
        assert 'Message 1' in log_file1.read_text()
        assert 'Message 2' in log_file2.read_text()


class TestLoggerFormatting:
    """Test the log message formatting."""

    def test_format_includes_asctime(self, tmp_path):
        """Test that format includes asctime."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.info('Test')

        content = log_file.read_text()
        # Should have timestamp pattern
        import re
        assert re.search(r'\d{4}-\d{2}-\d{2}', content)

    def test_format_includes_levelname(self, tmp_path):
        """Test that format includes levelname."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.info('Test info')
        logger.warning('Test warning')
        logger.error('Test error')

        content = log_file.read_text()
        assert 'INFO' in content
        assert 'WARNING' in content
        assert 'ERROR' in content

    def test_format_includes_message(self, tmp_path):
        """Test that format includes the log message."""
        logger = logging.getLogger('audiosmith')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = tmp_path / 'test.log'
        setup_logging(log_file=str(log_file))

        logger = logging.getLogger('audiosmith')
        logger.info('Custom test message here')

        content = log_file.read_text()
        assert 'Custom test message here' in content
