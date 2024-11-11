from typing import Dict, List
import logging
import numpy as np
import pandas as pd

from .database.sqlite_manager import SQLiteManager
from .data_cleaning.auto_cleaner import AutoDataCleaner
from .data_cleaning.column_renamer import ColumnRenamer
from .data_cleaning.data_merger import SmartDataMerger
from .config import (
    DEFAULT_DB_PATH, BATCH_SIZE, LOG_FILE,
    LOG_FORMAT
)

class DataProcessor:
    def __init__(self, input_paths: List[str]):
        self.input_paths = input_paths
        self.merger = SmartDataMerger()
        self.cleaner = AutoDataCleaner()
        self.renamer = ColumnRenamer()
        self.db_manager = SQLiteManager()
        self.batch_size = BATCH_SIZE
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_and_merge_data(self) -> pd.DataFrame:
        try:
            self.logger.info("Loading data files for merging...")
            dataframes = []
            
            for file_path in self.input_paths:
                try:
                    df = pd.read_csv(file_path)
                    self.logger.info(f"Loaded {len(df)} rows from {file_path}")
                    dataframes.append(df)
                except Exception as e:
                    self.logger.error(f"Error loading file {file_path}: {str(e)}")
                    raise
            
            if not dataframes:
                raise ValueError("No data files were successfully loaded")
            
            if len(dataframes) == 1:
                return dataframes[0]
            
            self.logger.info("Merging dataframes...")
            merged_df = self.merger.merge_dataframes(dataframes)
            self.logger.info(f"Successfully merged data: {len(merged_df)} rows")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error in data merging process: {str(e)}")
            raise

    def process(self, table_name: str = 'flight_data') -> Dict:
        try:
            self.logger.info("Starting data merge process...")
            merged_df = self.load_and_merge_data()
            
            # Use a small sample to get column names
            self.logger.info("Determining column names from sample...")
            sample_chunk = merged_df.iloc[:100]
            renamed_sample = self.renamer.rename_columns(sample_chunk)
            column_mapping = dict(zip(sample_chunk.columns, renamed_sample.columns))
            
            self.logger.info(f"Applying column mapping: {column_mapping}")
            merged_df = merged_df.rename(columns=column_mapping)
            
            self.logger.info("Initializing database with first chunk...")
            first_chunk = merged_df.iloc[:1]
            cleaned_first_chunk = self.cleaner.clean_dataset(first_chunk)
            self.db_manager.initialize_database(cleaned_first_chunk, table_name)
            
            total_rows = 0
            chunks = np.array_split(merged_df, max(1, len(merged_df) // self.batch_size))
            
            for chunk_num, chunk in enumerate(chunks):
                try:
                    self.logger.info(f"Processing chunk {chunk_num + 1}/{len(chunks)}, size: {len(chunk)} rows...")
                    cleaned_chunk = self.cleaner.clean_dataset(chunk)
                    
                    self.db_manager.write_data_chunk(cleaned_chunk, table_name)
                    total_rows += len(cleaned_chunk)
                    self.logger.info(f"Successfully processed chunk {chunk_num + 1}, total rows: {total_rows}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_num + 1}: {str(e)}")
                    continue
            
            self.logger.info("Performing final data validation...")
            validation_results = self.db_manager.validate_data(table_name)
            self.log_validation_results(validation_results)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {str(e)}")
            raise

    def log_validation_results(self, validation_results: Dict) -> None:
        self.logger.info("\n=== Data Validation Results ===")
        self.logger.info(f"Total Rows Processed: {validation_results['total_rows']}")
        self.logger.info("\nNull Counts by Column:")
        for column, null_count in validation_results['null_counts'].items():
            percentage = (null_count / validation_results['total_rows']) * 100
            self.logger.info(f"{column}: {null_count} nulls ({percentage:.2f}%)")