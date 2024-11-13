from typing import Dict
import pandas as pd
from openai import OpenAI
import logging
import re
import traceback
from ..config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE
from ..utils.logging_config import get_cleaning_logger

class ColumnRenamer:
    def __init__(self):
        self.logger = get_cleaning_logger()
        self.logger.info("Initializing ColumnRenamer")
        try:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.logger.info(f"OpenAI client initialized with model: {OPENAI_MODEL}")
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI client")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def suggest_column_names(self, columns: list) -> Dict[str, str]:
        """Get better column name suggestions from OpenAI"""
        self.logger.info(f"Requesting column name suggestions for {len(columns)} columns")
        self.logger.debug(f"Original columns: {', '.join(columns)}")
        
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
            self.logger.debug(f"Sending request to OpenAI API with temperature: {TEMPERATURE}")
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
            self.logger.debug(f"Received response from OpenAI: {suggestion_text[:200]}...")

            # Find the dictionary in the response
            dict_start = suggestion_text.find('{')
            dict_end = suggestion_text.rfind('}') + 1
            
            if dict_start >= 0 and dict_end > dict_start:
                suggestion_dict = eval(suggestion_text[dict_start:dict_end])
                
                # Validate suggestions
                self.logger.debug("Validating suggested column names")
                for orig, suggested in suggestion_dict.items():
                    if len(suggested) > 64:
                        self.logger.warning(f"Suggested name '{suggested}' exceeds 64 characters")
                    if not all(c.isalnum() or c == '_' for c in suggested):
                        self.logger.warning(f"Suggested name '{suggested}' contains invalid characters")
                
                self.logger.info(f"Successfully received {len(suggestion_dict)} column suggestions")
                return suggestion_dict
            else:
                self.logger.error("No valid dictionary found in OpenAI response")
                self.logger.debug(f"Full response: {suggestion_text}")
                raise ValueError("No valid dictionary found in response")

        except Exception as e:
            self.logger.error("Failed to get column suggestions from OpenAI")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            self.logger.info("Falling back to basic column name cleaning")
            return self._basic_clean_column_names(columns)

    def _basic_clean_column_names(self, columns: list) -> Dict[str, str]:
        """Fallback method for basic column name cleaning"""
        self.logger.info(f"Performing basic cleaning for {len(columns)} columns")
        clean_names = {}
        
        for col in columns:
            self.logger.debug(f"Cleaning column: '{col}'")
            
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
                self.logger.debug(f"Adding 'col_' prefix to numeric column: '{clean_name}'")
                clean_name = f'col_{clean_name}'
                
            clean_names[col] = clean_name
            
            if col != clean_name:
                self.logger.debug(f"Column cleaned: '{col}' -> '{clean_name}'")
            
        self.logger.info(f"Basic cleaning completed with {sum(1 for k, v in clean_names.items() if k != v)} changes")
        return clean_names

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename DataFrame columns using suggestions or fallback to basic cleaning"""
        try:
            self.logger.info(f"Starting column renaming for DataFrame with {len(df.columns)} columns")
            
            # Get column name suggestions
            suggestions = self.suggest_column_names(df.columns.tolist())
            
            # Validate all suggested names are unique
            if len(set(suggestions.values())) != len(suggestions):
                self.logger.error("Suggested column names contain duplicates")
                duplicate_names = [name for name in suggestions.values() 
                                 if list(suggestions.values()).count(name) > 1]
                self.logger.debug(f"Duplicate names: {duplicate_names}")
                raise ValueError("Suggested column names must be unique")
            
            # Apply the renaming
            renamed_df = df.rename(columns=suggestions)
            
            # Log the changes
            changes = [(old, new) for old, new in suggestions.items() if old != new]
            self.logger.info(f"Applied {len(changes)} column name changes:")
            for old_name, new_name in changes:
                self.logger.info(f"  {old_name} -> {new_name}")
            
            return renamed_df
            
        except Exception as e:
            self.logger.error("Failed to rename columns")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            self.logger.warning("Returning DataFrame with original column names")
            return df