"""
Logger utility for CollabGPT.

This module provides a consistent logging interface for the entire application.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import sys

from ..config import settings

# Create a custom logger
logger = logging.getLogger('collabgpt')

# Only configure the logger once
if not logger.handlers:
    # Set the log level
    log_level = getattr(logging, settings.LOGGING['level'].upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if a log file path is specified
    log_file_path = settings.LOGGING.get('file_path')
    if log_file_path:
        try:
            # Create log directory if needed
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Configure rotating file handler
            max_size_bytes = settings.LOGGING.get('max_file_size_mb', 10) * 1024 * 1024
            backup_count = settings.LOGGING.get('backup_count', 5)
            
            file_handler = RotatingFileHandler(
                log_file_path, 
                maxBytes=max_size_bytes, 
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to configure file logging: {e}")


def get_logger(module_name: str = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        module_name: The name of the module (optional)
        
    Returns:
        A configured logger instance
    """
    if module_name:
        return logging.getLogger(f"collabgpt.{module_name}")
    return logger


# Convenience functions for logging in different levels
def debug(msg: str, *args, **kwargs):
    """Log a debug message."""
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log an info message."""
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log a warning message."""
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log an error message."""
    logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log a critical message."""
    logger.critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs):
    """Log an exception (includes traceback)."""
    logger.exception(msg, *args, **kwargs)


class LogContext:
    """Context manager for grouped log messages."""
    
    def __init__(self, context_name: str, log_level: str = 'info'):
        self.context_name = context_name
        self.log_level = log_level
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        log_func = getattr(logger, self.log_level.lower())
        log_func(f"Starting: {self.context_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        
        if exc_type:
            logger.error(f"Failed: {self.context_name} - {exc_val} (took {duration.total_seconds():.2f}s)")
        else:
            log_func = getattr(logger, self.log_level.lower())
            log_func(f"Completed: {self.context_name} (took {duration.total_seconds():.2f}s)")