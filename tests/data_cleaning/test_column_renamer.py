import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
from flight_analysis.data_cleaning.column_renamer import ColumnRenamer

class TestColumnRenamer:
    @pytest.fixture
    def renamer(self):
        return ColumnRenamer()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'Customer ID': [1, 2, 3],
            'Flight#': ['A1', 'B2', 'C3'],
            'Price($)': [100, 200, 300]
        })

    def test_rename_columns_standard_format(self, renamer, sample_df):
        with patch('openai.ChatCompletion.create') as mock_openai:
            # Match the actual implementation's column naming
            mock_openai.return_value.choices[0].message.content = '''
            {
                "Customer ID": "customer_id",
                "Flight#": "flight_number",
                "Price($)": "price_usd"
            }
            '''
            
            renamed_df = renamer.rename_columns(sample_df)
            expected_columns = ['customer_id', 'flight_number', 'price_usd']
            assert list(renamed_df.columns) == expected_columns

    def test_rename_columns_fallback(self, renamer, sample_df):
        with patch('openai.ChatCompletion.create', side_effect=Exception):
            renamed_df = renamer.rename_columns(sample_df)
            # Test that columns are lowercase
            assert all(col.islower() for col in renamed_df.columns)
            
            # Test that special characters are replaced with underscores
            for col in renamed_df.columns:
                cleaned_col = col.lower().replace(' ', '_').replace('$', '').replace('#', '')
                assert '_' in col or not any(char in col for char in ' $#()') 

    def test_rename_columns_preserves_data(self, renamer, sample_df):
        original_values = sample_df.values.copy()
        renamed_df = renamer.rename_columns(sample_df)
        assert np.array_equal(renamed_df.values, original_values)

    def test_rename_columns_handles_empty_df(self, renamer):
        empty_df = pd.DataFrame()
        renamed_df = renamer.rename_columns(empty_df)
        assert renamed_df.empty

    def test_rename_columns_handles_numeric_columns(self, renamer):
        df_with_numeric = pd.DataFrame({
            '1st Column': [1, 2, 3],
            '2nd_data': [4, 5, 6]
        })
        renamed_df = renamer.rename_columns(df_with_numeric)
        assert all(not col[0].isdigit() for col in renamed_df.columns)

    def test_rename_columns_special_characters(self, renamer):
        df_with_special = pd.DataFrame({
            'Column@#$%': [1, 2, 3],
            'Data&&**': [4, 5, 6]
        })
        renamed_df = renamer.rename_columns(df_with_special)
        assert all(col.replace('_', '').isalnum() for col in renamed_df.columns)