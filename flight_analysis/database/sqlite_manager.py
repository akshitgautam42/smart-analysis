from contextlib import contextmanager
import logging
import sqlite3
import re
import pandas as pd
import traceback
from typing import List, Tuple, Dict, Any
from ..config import DEFAULT_DB_PATH, LOG_FORMAT, DB_LOG_FILE, SQLITE_TIMEOUT
from ..utils.logging_config import get_database_logger

class SQLiteManager:
    def __init__(self):
        self.logger = get_database_logger()
        self.db_path = DEFAULT_DB_PATH
        self.logger.info(f"Initializing SQLiteManager with database: {self.db_path}")
        self._verify_database_access()

    def _verify_database_access(self):
        """Verify database accessibility on initialization"""
        try:
            with self.get_connection() as conn:
                self.logger.debug("Successfully verified database access")
        except Exception as e:
            self.logger.error("Failed to verify database access")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a database connection with proper error handling"""
        conn = None
        self.logger.debug(f"Attempting to connect to database: {self.db_path}")
        try:
            conn = sqlite3.connect(self.db_path, timeout=SQLITE_TIMEOUT)
            self.logger.debug("Database connection established")
            yield conn
        except sqlite3.Error as e:
            self.logger.error("SQLite error while connecting to database")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
        except Exception as e:
            self.logger.error("Unexpected error while connecting to database")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed")

    def initialize_database(self, sample_df: pd.DataFrame, table_name: str):
        """Initialize database with table schema based on DataFrame"""
        self.logger.info(f"Initializing database table: {table_name}")
        self.logger.debug(f"Sample DataFrame shape: {sample_df.shape}")
        
        create_table_sql = self._generate_create_table_sql(sample_df, table_name)
        self.logger.debug(f"Generated SQL: {create_table_sql}")
        
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                self.logger.debug(f"Dropping existing table if exists: {table_name}")
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                self.logger.debug("Creating new table")
                cursor.execute(create_table_sql)
                conn.commit()
                self.logger.info(f"Successfully initialized table: {table_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize table: {table_name}")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise

    def _generate_create_table_sql(self, df: pd.DataFrame, table_name: str) -> str:
        """Generate SQL for table creation with proper type mapping"""
        self.logger.debug(f"Generating CREATE TABLE SQL for {table_name}")
        
        type_mapping = {
            'int64': 'INTEGER',
            'float64': 'REAL',
            'datetime64[ns]': 'TEXT',
            'bool': 'INTEGER',
            'object': 'TEXT'
        }
        
        # Sanitize table name
        safe_table_name = self._sanitize_identifier(table_name)
        self.logger.debug(f"Sanitized table name: {safe_table_name}")
        
        columns = []
        for col, dtype in df.dtypes.items():
            safe_col = self._sanitize_identifier(col)
            sql_type = type_mapping.get(str(dtype), 'TEXT')
            columns.append(f'"{safe_col}" {sql_type}')
            self.logger.debug(f"Column mapping: {col} -> {safe_col} ({sql_type})")
        
        create_table_sql = f"""
        CREATE TABLE {safe_table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {',\n            '.join(columns)}
        )
        """
        return create_table_sql

    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifiers"""
        safe_id = re.sub(r'[^\w]', '_', str(identifier))
        safe_id = safe_id.strip('_')
        if safe_id[0].isdigit():
            safe_id = f'{"col" if len(safe_id) < 20 else "c"}_{safe_id}'
        return safe_id

    def write_data_chunk(self, df_chunk: pd.DataFrame, table_name: str):
        """Write DataFrame chunk to database with proper error handling"""
        self.logger.info(f"Writing data chunk to table {table_name}")
        self.logger.debug(f"Chunk shape: {df_chunk.shape}")
        
        df_to_write = df_chunk.copy()
        safe_table_name = self._sanitize_identifier(table_name)
        
        # Clean column names
        rename_dict = {col: self._sanitize_identifier(col) for col in df_to_write.columns}
        df_to_write = df_to_write.rename(columns=rename_dict)
        self.logger.debug(f"Column name mapping: {rename_dict}")
        
        with self.get_connection() as conn:
            try:
                # Handle datetime columns
                datetime_cols = df_to_write.select_dtypes(include=['datetime64']).columns
                if datetime_cols.any():
                    self.logger.debug(f"Converting datetime columns: {list(datetime_cols)}")
                    for col in datetime_cols:
                        df_to_write[col] = df_to_write[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_to_write.to_sql(safe_table_name, conn, if_exists='append', index=False)
                self.logger.info(f"Successfully wrote {len(df_to_write):,} rows to {safe_table_name}")
            except Exception as e:
                self.logger.error(f"Failed to write chunk to table: {safe_table_name}")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise

    def validate_data(self, table_name: str) -> Dict[str, Any]:
        """Validate data in table with comprehensive checks"""
        self.logger.info(f"Starting data validation for table: {table_name}")
        
        safe_table_name = self._sanitize_identifier(table_name)
        
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Get total row count
                cursor.execute(f"SELECT COUNT(*) FROM {safe_table_name}")
                total_rows = cursor.fetchone()[0]
                self.logger.info(f"Total rows in table: {total_rows:,}")
                
                # Get column information
                cursor.execute(f"SELECT * FROM {safe_table_name} LIMIT 1")
                columns = [description[0] for description in cursor.description]
                self.logger.debug(f"Found {len(columns)} columns: {columns}")
                
                # Calculate null counts and additional statistics
                null_counts = {}
                column_stats = {}
                
                for col in columns:
                    if col != 'id':
                        self.logger.debug(f"Analyzing column: {col}")
                        
                        # Null count
                        cursor.execute(f'SELECT COUNT(*) FROM {safe_table_name} WHERE "{col}" IS NULL')
                        null_count = cursor.fetchone()[0]
                        null_counts[col] = null_count
                        
                        # Basic statistics
                        cursor.execute(f'SELECT COUNT(DISTINCT "{col}") FROM {safe_table_name}')
                        unique_count = cursor.fetchone()[0]
                        
                        column_stats[col] = {
                            'unique_values': unique_count,
                            'null_percentage': (null_count / total_rows) * 100 if total_rows > 0 else 0
                        }
                        
                        self.logger.debug(f"Column '{col}' stats: {null_count:,} nulls, {unique_count:,} unique values")
                
                validation_results = {
                    'total_rows': total_rows,
                    'null_counts': null_counts,
                    'column_stats': column_stats
                }
                
                # Log validation summary
                self.logger.info("Data Validation Summary:")
                self.logger.info(f"Total Rows: {total_rows:,}")
                self.logger.info("Column Statistics:")
                for col, stats in column_stats.items():
                    self.logger.info(f"  {col}:")
                    self.logger.info(f"    Null Count: {null_counts[col]:,} ({stats['null_percentage']:.2f}%)")
                    self.logger.info(f"    Unique Values: {stats['unique_values']:,}")
                
                return validation_results
                
            except Exception as e:
                self.logger.error(f"Failed to validate table: {safe_table_name}")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise

    def list_tables(self) -> List[str]:
        """Get list of all tables in the database"""
        self.logger.debug("Retrieving list of tables")
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                self.logger.info(f"Found {len(tables)} tables in database")
                self.logger.debug(f"Tables: {tables}")
                return tables
            except Exception as e:
                self.logger.error("Failed to retrieve table list")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise

    def get_table_schema(self, table_name: str) -> List[Tuple]:
        """Get schema information for a specific table"""
        self.logger.debug(f"Retrieving schema for table: {table_name}")
        safe_table_name = self._sanitize_identifier(table_name)
        
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({safe_table_name})")
                schema = cursor.fetchall()
                self.logger.debug(f"Schema columns: {[col[1] for col in schema]}")
                return schema
            except Exception as e:
                self.logger.error(f"Failed to retrieve schema for table: {safe_table_name}")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise

    def drop_table(self, table_name: str):
        """Drop a specific table from the database"""
        self.logger.info(f"Attempting to drop table: {table_name}")
        safe_table_name = self._sanitize_identifier(table_name)
        
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {safe_table_name}")
                conn.commit()
                self.logger.info(f"Successfully dropped table: {safe_table_name}")
            except Exception as e:
                self.logger.error(f"Failed to drop table: {safe_table_name}")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise