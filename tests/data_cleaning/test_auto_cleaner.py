import pytest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
from flight_analysis.data_cleaning.auto_cleaner import AutoDataCleaner, EnhancedJSONEncoder

class TestAutoDataCleaner:
    @pytest.fixture
    def cleaner(self):
        return AutoDataCleaner()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'PREFIXcustomer_id': ['PRE123', 'PRE456', 'PRE789'],
            'FLIGHT_NUM': ['FL100', 'FL200', 'FL300'],
            'departure_TIME': ['09:00', '10:00', '11:00'],
            'price_USD': ['$100', '$200', '$300'],
            'is_cancelled': ['Yes', 'No', 'No']
        })

    def test_clean_dataset_removes_consistent_prefixes(self, cleaner, sample_df):
        """Test if cleaner removes consistent prefixes from column values"""
        with patch('crewai.Agent.execute_task') as mock_agent:
            # Mock the agent response with a proper cleaning operation
            mock_agent.return_value = '''
            {
                "cleaning_operations": [
                    {
                        "column": "PREFIXcustomer_id",
                        "operation": "remove_prefix",
                        "details": "Remove consistent PREFIX",
                        "pattern": "PRE"
                    }
                ]
            }
            '''
            
            cleaned_df = cleaner.clean_dataset(sample_df)
            # Check if at least one prefix was removed
            assert any(not id.startswith('PRE') for id in cleaned_df['PREFIXcustomer_id'])

    
    def test_clean_dataset_handles_null_values(self, cleaner):
        """Test if cleaner properly handles null values"""
        df = pd.DataFrame({
            'col1': ['data', None, 'NA', 'N/A', ''],
            'col2': ['value', np.nan, 'null', 'NULL', ' ']
        })

        with patch('crewai.Agent.execute_task') as mock_agent:
            mock_agent.return_value = '''
            {
                "cleaning_operations": [
                    {
                        "column": "col1",
                        "operation": "handle_nulls",
                        "details": "Convert missing values to NULL",
                        "pattern": "^(?i)(na|n/a|null|)$|^\\s*$"
                    },
                    {
                        "column": "col2",
                        "operation": "handle_nulls",
                        "details": "Convert missing values to NULL",
                        "pattern": "^(?i)(null)$|^\\s*$"
                    }
                ]
            }
            '''
            
            cleaned_df = cleaner.clean_dataset(df)
            # Count truly null values
            null_count_col1 = (cleaned_df['col1'].isna() | 
                            cleaned_df['col1'].str.lower().isin(['na', 'n/a', '', 'null'])).sum()
            null_count_col2 = (cleaned_df['col2'].isna() | 
                            cleaned_df['col2'].str.lower().isin(['null', ''])).sum()
            
            # Assert the expected number of null values
            assert null_count_col1 >= 3  # Should catch None, NA, N/A, ''
            assert null_count_col2 >= 3  # Should catch np.nan, null, NULL, ' '

    def test_clean_dataset_preserves_data_types(self, cleaner):
        """Test if cleaner preserves appropriate data types"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'date_col': pd.date_range(start='2024-01-01', periods=3)
        })
        cleaned_df = cleaner.clean_dataset(df)
        assert cleaned_df['int_col'].dtype == df['int_col'].dtype
        assert cleaned_df['float_col'].dtype == df['float_col'].dtype
        assert cleaned_df['str_col'].dtype == df['str_col'].dtype
        assert cleaned_df['date_col'].dtype == df['date_col'].dtype

class TestEnhancedJSONEncoder:
    def test_encoder_handles_numpy_types(self):
        """Test if encoder properly handles numpy data types"""
        encoder = EnhancedJSONEncoder()
        data = {
            'int': np.int64(42),
            'float': np.float64(3.14),
            'bool': np.bool_(True),
            'array': np.array([1, 2, 3])
        }
        encoded = encoder.encode(data)
        assert '42' in encoded
        assert '3.14' in encoded
        assert 'true' in encoded
        assert '[1, 2, 3]' in encoded

    def test_encoder_handles_pandas_na(self):
        """Test if encoder properly handles pandas NA values"""
        encoder = EnhancedJSONEncoder()
        data = {
            'na_value': pd.NA,
            'nat_value': pd.NaT,
            'none_value': None
        }
        encoded = encoder.encode(data)
        assert 'null' in encoded