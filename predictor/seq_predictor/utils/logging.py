import logging
import os
import sys


def init_logger(output_dir: str = None) -> logging.Logger:
    """
    Initialize a logger with file and line information.
    
    Args:
        output_dir: Optional directory to save log file. If None, only console output.
    
    Returns:
        Configured logger instance
    """
    # Create a custom formatter that includes file and line information
    log_format = "[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
    log_dateformat = "%Y-%m-%d %H:%M:%S"

    # Get the logger for the calling module
    logger = logging.getLogger(__name__)
    
    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Set the logger level
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, log_dateformat)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(log_format, log_dateformat)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the logger instance."""
    return logging.getLogger(__name__)