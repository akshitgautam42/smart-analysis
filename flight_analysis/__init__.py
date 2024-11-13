from .analysis.analyst import DataAnalyst
from .processor import DataProcessor
from .data_cleaning.auto_cleaner import AutoDataCleaner
from .data_cleaning.column_renamer import ColumnRenamer
from .data_cleaning.data_merger import SmartDataMerger
from .database.sqlite_manager import SQLiteManager
from .utils.logging_config import LogManager

from .utils.logging_config import (
    LogManager,
    get_processing_logger,
    get_database_logger,
    get_analysis_logger,
    get_cleaning_logger,
    get_api_logger
)
__all__ = [
    'DataAnalyst',
    'DataProcessor',
    'AutoDataCleaner',
    'ColumnRenamer',
    'SmartDataMerger',
    'SQLiteManager',
    'LogManager'
]

# Version info
__version__ = '1.0.0'

# Initialize logging
from .utils.logging_config import get_processing_logger
logger = get_processing_logger()
logger.info(f"Initializing flight_analysis package v{__version__}")