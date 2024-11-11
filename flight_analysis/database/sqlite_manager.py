from contextlib import contextmanager
import logging
import sqlite3
import re
import pandas as pd
from ..config import DEFAULT_DB_PATH, LOG_FORMAT, DB_LOG_FILE, SQLITE_TIMEOUT


class SQLiteManager:
    def __init__(self):
        self.db_path = DEFAULT_DB_PATH
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(DB_LOG_FILE),
                logging.StreamHandler()
            ]
        )

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def initialize_database(self, sample_df: pd.DataFrame, table_name: str):
        create_table_sql = self._generate_create_table_sql(sample_df, table_name)
        
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                cursor.execute(create_table_sql)
                conn.commit()
                self.logger.info(f"Database initialized successfully with table {table_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize database: {str(e)}")
                raise

    def _generate_create_table_sql(self, df: pd.DataFrame, table_name: str) -> str:
        type_mapping = {
            'int64': 'INTEGER',
            'float64': 'REAL',
            'datetime64[ns]': 'TEXT',
            'bool': 'INTEGER',
            'object': 'TEXT'
        }
        
        # Ensure table name is SQL safe
        safe_table_name = re.sub(r'[^\w]', '_', str(table_name))
        safe_table_name = safe_table_name.strip('_')
        if safe_table_name[0].isdigit():
            safe_table_name = f'tbl_{safe_table_name}'
        
        columns = []
        for col, dtype in df.dtypes.items():
            safe_col = re.sub(r'[^\w]', '_', str(col))
            safe_col = safe_col.strip('_')
            if safe_col[0].isdigit():
                safe_col = f'col_{safe_col}'
            
            sql_type = type_mapping.get(str(dtype), 'TEXT')
            columns.append(f'"{safe_col}" {sql_type}')
        
        create_table_sql = f"""
        CREATE TABLE {safe_table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {',\n            '.join(columns)}
        )
        """
        
        return create_table_sql

    def write_data_chunk(self, df_chunk: pd.DataFrame, table_name: str):
        df_to_write = df_chunk.copy()
        
        # Ensure table name is SQL safe
        safe_table_name = re.sub(r'[^\w]', '_', str(table_name))
        safe_table_name = safe_table_name.strip('_')
        if safe_table_name[0].isdigit():
            safe_table_name = f'tbl_{safe_table_name}'
        
        # Clean column names
        rename_dict = {}
        for col in df_to_write.columns:
            safe_col = re.sub(r'[^\w]', '_', str(col))
            safe_col = safe_col.strip('_')
            if safe_col[0].isdigit():
                safe_col = f'col_{safe_col}'
            rename_dict[col] = safe_col
        
        df_to_write = df_to_write.rename(columns=rename_dict)
        
        with self.get_connection() as conn:
            try:
                # Convert datetime columns to string format
                for col in df_to_write.select_dtypes(include=['datetime64']):
                    df_to_write[col] = df_to_write[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_to_write.to_sql(safe_table_name, conn, if_exists='append', index=False)
                self.logger.info(f"Successfully wrote {len(df_to_write)} rows to {safe_table_name}")
            except Exception as e:
                self.logger.error(f"Failed to write chunk to database: {str(e)}")
                raise

    def validate_data(self, table_name: str):
        # Ensure table name is SQL safe
        safe_table_name = re.sub(r'[^\w]', '_', str(table_name))
        safe_table_name = safe_table_name.strip('_')
        if safe_table_name[0].isdigit():
            safe_table_name = f'tbl_{safe_table_name}'
            
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Get total row count
                cursor.execute(f"SELECT COUNT(*) FROM {safe_table_name}")
                total_rows = cursor.fetchone()[0]
                
                # Get column information
                cursor.execute(f"SELECT * FROM {safe_table_name} LIMIT 1")
                columns = [description[0] for description in cursor.description]
                
                # Calculate null counts for each column
                null_counts = {}
                for col in columns:
                    if col != 'id':
                        cursor.execute(f'SELECT COUNT(*) FROM {safe_table_name} WHERE "{col}" IS NULL')
                        null_counts[col] = cursor.fetchone()[0]
                
                validation_results = {
                    'total_rows': total_rows,
                    'null_counts': null_counts
                }
                
                # Log validation results
                self.logger.info(f"Database Validation Results for {safe_table_name}:")
                self.logger.info(f"Total Rows: {total_rows}")
                self.logger.info("Null Counts by Column:")
                for col, count in null_counts.items():
                    self.logger.info(f"{col}: {count}")
                
                return validation_results
                
            except Exception as e:
                self.logger.error(f"Failed to validate database: {str(e)}")
                raise

    def list_tables(self) -> list:
        """Get list of all tables in the database"""
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                self.logger.error(f"Failed to get table list: {str(e)}")
                raise

    def get_table_schema(self, table_name: str) -> list:
        """Get schema information for a specific table"""
        safe_table_name = re.sub(r'[^\w]', '_', str(table_name))
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({safe_table_name})")
                return cursor.fetchall()
            except Exception as e:
                self.logger.error(f"Failed to get schema for {table_name}: {str(e)}")
                raise

    def drop_table(self, table_name: str):
        """Drop a specific table from the database"""
        safe_table_name = re.sub(r'[^\w]', '_', str(table_name))
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {safe_table_name}")
                conn.commit()
                self.logger.info(f"Successfully dropped table {safe_table_name}")
            except Exception as e:
                self.logger.error(f"Failed to drop table {table_name}: {str(e)}")
                raise