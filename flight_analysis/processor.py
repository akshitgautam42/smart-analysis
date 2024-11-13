from typing import Dict, List
import logging
import numpy as np
import pandas as pd
import traceback

from .database.sqlite_manager import SQLiteManager
from .data_cleaning.auto_cleaner import AutoDataCleaner
from .data_cleaning.column_renamer import ColumnRenamer
from .data_cleaning.data_merger import SmartDataMerger
from .config import (
    DEFAULT_DB_PATH, BATCH_SIZE, LOG_FILE,
    LOG_FORMAT
)
from .utils.logging_config import get_processing_logger

class DataProcessor:
    def __init__(self, input_paths: List[str]):
        self.input_paths = input_paths
        self.merger = SmartDataMerger()
        self.cleaner = AutoDataCleaner()
        self.renamer = ColumnRenamer()
        self.db_manager = SQLiteManager()
        self.batch_size = BATCH_SIZE
        self.logger = get_processing_logger()
        self.logger.info(f"Initialized DataProcessor with {len(input_paths)} input files")
      

    def load_and_merge_data(self) -> pd.DataFrame:
        """Load and merge data from input paths."""
        try:
            self.logger.info("Starting data loading process")
            dataframes = []
            total_rows = 0
            
            for file_path in self.input_paths:
                try:
                    self.logger.info(f"Loading file: {file_path}")
                    df = pd.read_csv(file_path)
                    rows = len(df)
                    total_rows += rows
                    self.logger.info(f"Successfully loaded {rows:,} rows from {file_path}")
                    self.logger.debug(f"Columns found in {file_path}: {', '.join(df.columns)}")
                    dataframes.append(df)
                except Exception as e:
                    self.logger.error(f"Failed to load file {file_path}")
                    self.logger.error(f"Error details: {str(e)}")
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
                    raise
            
            if not dataframes:
                self.logger.error("No data files were successfully loaded")
                raise ValueError("No data files were successfully loaded")
            
            if len(dataframes) == 1:
                self.logger.info("Single dataframe detected, skipping merge step")
                return dataframes[0]
            
            self.logger.info(f"Starting merge of {len(dataframes)} dataframes")
            merged_df = self.merger.merge_dataframes(dataframes)
            self.logger.info(f"Successfully merged data: {len(merged_df):,} rows from {total_rows:,} input rows")
            self.logger.debug(f"Final merged columns: {', '.join(merged_df.columns)}")
            
            return merged_df
            
        except Exception as e:
            self.logger.error("Critical error in data merging process")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def process(self, table_name: str = 'flight_data') -> Dict:
        """Process the data through cleaning and loading pipeline."""
        try:
            self.logger.info(f"Starting processing pipeline for table: {table_name}")
            merged_df = self.load_and_merge_data()
            
            # Column renaming process
            self.logger.info("Starting column renaming process")
            sample_size = min(100, len(merged_df))
            self.logger.debug(f"Using sample size of {sample_size} for column name determination")
            sample_chunk = merged_df.iloc[:sample_size]
            renamed_sample = self.renamer.rename_columns(sample_chunk)
            column_mapping = dict(zip(sample_chunk.columns, renamed_sample.columns))
            
            self.logger.info("Applying column mapping to full dataset")
            self.logger.debug(f"Column mapping: {column_mapping}")
            merged_df = merged_df.rename(columns=column_mapping)
            
            # Database initialization
            self.logger.info("Initializing database with schema")
            first_chunk = merged_df.iloc[:1]
            cleaned_first_chunk = self.cleaner.clean_dataset(first_chunk)
            self.db_manager.initialize_database(cleaned_first_chunk, table_name)
            
            # Chunk processing
            total_rows = 0
            num_chunks = max(1, len(merged_df) // self.batch_size)
            chunks = np.array_split(merged_df, num_chunks)
            self.logger.info(f"Processing {len(chunks):,} chunks of approximately {self.batch_size:,} rows each")
            
            for chunk_num, chunk in enumerate(chunks, 1):
                try:
                    self.logger.info(f"Processing chunk {chunk_num}/{len(chunks)}")
                    self.logger.debug(f"Chunk {chunk_num} size: {len(chunk):,} rows")
                    
                    cleaned_chunk = self.cleaner.clean_dataset(chunk)
                    self.db_manager.write_data_chunk(cleaned_chunk, table_name)
                    
                    total_rows += len(cleaned_chunk)
                    progress = (chunk_num / len(chunks)) * 100
                    self.logger.info(f"Chunk {chunk_num} completed. Progress: {progress:.1f}% ({total_rows:,} total rows)")
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_num}/{len(chunks)}")
                    self.logger.error(f"Error details: {str(e)}")
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # Validation
            self.logger.info("Starting final data validation")
            validation_results = self.db_manager.validate_data(table_name)
            self.log_validation_results(validation_results)
            
            return validation_results
            
        except Exception as e:
            self.logger.error("Critical error in processing pipeline")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def log_validation_results(self, validation_results: Dict) -> None:
        """Log the results of data validation checks."""
        self.logger.info("=== Data Validation Results ===")
        self.logger.info(f"Total Rows Processed: {validation_results['total_rows']:,}")
        
        self.logger.info("Null Counts by Column:")
        max_nulls = 0
        max_null_column = None
        
        for column, null_count in validation_results['null_counts'].items():
            percentage = (null_count / validation_results['total_rows']) * 100
            self.logger.info(f"{column}: {null_count:,} nulls ({percentage:.2f}%)")
            
            if null_count > max_nulls:
                max_nulls = null_count
                max_null_column = column
        
        if max_null_column:
            self.logger.warning(f"Highest null count in column '{max_null_column}' with {max_nulls:,} nulls")