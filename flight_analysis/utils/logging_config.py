# utils/logging_config.py
import logging
import sys
from pathlib import Path
from datetime import datetime
from ..config import LOG_DIR

class LogManager:
    LOGGERS = {}

    @staticmethod
    def setup_logger(name: str, log_file: str = None, level=logging.INFO):
        """Set up a new logger or return existing one"""
        if name in LogManager.LOGGERS:
            return LogManager.LOGGERS[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Ensure log directory exists
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Console handler (for errors only)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(
                LOG_DIR / f"{log_file}_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

        LogManager.LOGGERS[name] = logger
        return logger

# Common loggers
def get_processing_logger():
    return LogManager.setup_logger('processing', 'processing')

def get_database_logger():
    return LogManager.setup_logger('database', 'database')

def get_analysis_logger():
    return LogManager.setup_logger('analysis', 'analysis')

def get_cleaning_logger():
    return LogManager.setup_logger('cleaning', 'cleaning')

def get_api_logger():
    return LogManager.setup_logger('api', 'api')