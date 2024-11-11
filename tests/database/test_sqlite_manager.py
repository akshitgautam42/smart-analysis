import pytest
import pandas as pd
import sqlite3
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
from flight_analysis.database.sqlite_manager import SQLiteManager

class TestSQLiteManager:
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)  # Clean up after test

    @pytest.fixture
    def manager(self, temp_db_path):
        """Create SQLiteManager instance with temporary database"""
        manager = SQLiteManager()
        manager.db_path = temp_db_path
        return manager

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'flight_id': [1, 2, 3],
            'airline': ['AA', 'UA', 'DL'],
            'departure_time': ['2024-01-01 10:00', '2024-01-01 11:00', '2024-01-01 12:00'],
            'price': [100.50, 200.75, 150.25],
            'is_cancelled': [False, True, False]
        })

    def test_initialize_database(self, manager, sample_df):
        """Test database initialization with sample data"""
        manager.initialize_database(sample_df)
        
        with manager.get_connection() as conn:
            cursor = conn.cursor()
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='flight_data'")
            assert cursor.fetchone() is not None
            
            # Check column names and types
            cursor.execute("PRAGMA table_info(flight_data)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            assert 'flight_id' in columns
            assert 'airline' in columns
            assert 'price' in columns
            assert columns['price'] == 'REAL'
            assert columns['is_cancelled'] == 'INTEGER'

    def test_write_data_chunk(self, manager, sample_df):
        """Test writing data chunks to database"""
        manager.initialize_database(sample_df)
        manager.write_data_chunk(sample_df)
        
        with manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM flight_data")
            assert cursor.fetchone()[0] == len(sample_df)
            
            cursor.execute("SELECT * FROM flight_data")
            data = cursor.fetchall()
            assert len(data) == len(sample_df)

    def test_validate_data(self, manager, sample_df):
        """Test data validation functionality"""
        manager.initialize_database(sample_df)
        manager.write_data_chunk(sample_df)
        
        # Add some null values
        df_with_nulls = sample_df.copy()
        df_with_nulls.loc[0, 'airline'] = None
        manager.write_data_chunk(df_with_nulls)
        
        validation_results = manager.validate_data()
        
        assert 'total_rows' in validation_results
        assert 'null_counts' in validation_results
        assert validation_results['total_rows'] == len(sample_df) * 2
        assert validation_results['null_counts']['airline'] == 1

    def test_connection_error_handling(self, manager):
        """Test database connection error handling"""
        manager.db_path = "nonexistent/path/to/db.sqlite"
        
        with pytest.raises(Exception) as exc_info:
            with manager.get_connection():
                pass
        assert "database" in str(exc_info.value).lower()

    def test_handle_special_column_names(self, manager):
        """Test handling of special characters in column names"""
        df_special = pd.DataFrame({
            'Column With Spaces': [1, 2, 3],
            'Special@#Characters': ['a', 'b', 'c'],
            '123NumericStart': [4, 5, 6]
        })
        
        manager.initialize_database(df_special)
        manager.write_data_chunk(df_special)
        
        with manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(flight_data)")
            columns = [row[1].lower() for row in cursor.fetchall()]
            
            # Check if special characters are properly handled (case insensitive)
            assert any('column_with_spaces' in col for col in columns)
            # assert any('special_characters' in col.replace('@', '_').replace('#', '_') for col in columns)
            assert any('numeric_start' in col or col.startswith('col_123') for col in columns)

    def test_datetime_handling(self, manager):
        """Test handling of datetime values"""
        df_dates = pd.DataFrame({
            'departure': pd.date_range(start='2024-01-01', periods=3),
            'arrival': pd.date_range(start='2024-01-02', periods=3)
        })
        
        manager.initialize_database(df_dates)
        manager.write_data_chunk(df_dates)
        
        with manager.get_connection() as conn:
            df_read = pd.read_sql("SELECT * FROM flight_data", conn)
            assert pd.to_datetime(df_read['departure']).dt.date.equals(df_dates['departure'].dt.date)
            assert pd.to_datetime(df_read['arrival']).dt.date.equals(df_dates['arrival'].dt.date)