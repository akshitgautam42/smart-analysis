# column_renamer.py
from typing import Dict
import pandas as pd
from openai import OpenAI
import logging
import re
from ..config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE

class ColumnRenamer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def suggest_column_names(self, columns: list) -> Dict[str, str]:
        """Get better column name suggestions from OpenAI"""
        prompt = f"""
        Suggest standardized names for these database columns: {columns}

        Follow these exact rules:
        1. Use clear, descriptive names
        2. Convert to snake_case
        3. Remove special characters and spaces
        4. Keep standard abbreviations (id, url, etc.)
        5. Maintain semantic meaning
        6. Fix spelling errors
        7. Keep common prefixes/suffixes (pre_, post_, _id, _date, etc.)
        8. Make plurals consistent
        9. No abbreviations unless standard (num -> number, qty -> quantity)
        10. Boolean columns should start with 'is_' or 'has_'
        11. Date columns should end with '_date' or '_datetime'
        12. Maximum length of 64 characters

        Return a Python dictionary mapping original names to suggested names.
        Example format: {{"original_col": "new_col_name", "another_col": "another_name"}}
        """

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=TEMPERATURE
            )

            # Get the response content
            suggestion_text = response.choices[0].message.content

            # Find the dictionary in the response
            dict_start = suggestion_text.find('{')
            dict_end = suggestion_text.rfind('}') + 1
            if dict_start >= 0 and dict_end > dict_start:
                suggestion_dict = eval(suggestion_text[dict_start:dict_end])
                return suggestion_dict
            else:
                raise ValueError("No valid dictionary found in response")

        except Exception as e:
            self.logger.error(f"Error getting column suggestions: {str(e)}")
            # Fall back to basic cleaning if API call fails
            return self._basic_clean_column_names(columns)

    def _basic_clean_column_names(self, columns: list) -> Dict[str, str]:
        """Fallback method for basic column name cleaning"""
        clean_names = {}
        for col in columns:
            # Convert to lowercase
            clean_name = col.lower()
            
            # Replace special characters and spaces with underscore
            clean_name = re.sub(r'[^\w\s]', '_', clean_name)
            clean_name = re.sub(r'\s+', '_', clean_name)
            
            # Remove duplicate underscores
            clean_name = re.sub(r'_+', '_', clean_name)
            
            # Remove leading/trailing underscores
            clean_name = clean_name.strip('_')
            
            # If column starts with number, prefix with 'col_'
            if clean_name[0].isdigit():
                clean_name = f'col_{clean_name}'
                
            clean_names[col] = clean_name
            
        return clean_names

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename DataFrame columns using suggestions or fallback to basic cleaning"""
        try:
            # Get column name suggestions
            suggestions = self.suggest_column_names(df.columns.tolist())
            
            # Apply the renaming
            renamed_df = df.rename(columns=suggestions)
            
            # Log the changes
            self.logger.info("Column renaming changes:")
            for old_name, new_name in suggestions.items():
                if old_name != new_name:
                    self.logger.info(f"  {old_name} -> {new_name}")
            
            return renamed_df
            
        except Exception as e:
            self.logger.error(f"Error in column renaming: {str(e)}")
            # If anything fails, return original DataFrame
            return df