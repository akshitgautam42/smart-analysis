from typing import Any, Dict, List, Tuple
import json
import numpy as np
import pandas as pd
from crewai import Agent, Crew, Task
from langchain.chat_models import ChatOpenAI
import re
import traceback
from ..config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE
from ..utils.logging_config import get_cleaning_logger

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

class AutoDataCleaner:
    def __init__(self):
        self.logger = get_cleaning_logger()
        self.logger.info("Initializing AutoDataCleaner")
        
        try:
            self.logger.debug(f"Setting up ChatOpenAI with model: {OPENAI_MODEL}, temperature: {TEMPERATURE}")
            self.llm = ChatOpenAI(
                temperature=TEMPERATURE,
                model_name=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            self.setup_agent()
        except Exception as e:
            self.logger.error("Failed to initialize AutoDataCleaner")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def setup_agent(self):
        """Set up the cleaner agent with specific roles and goals."""
        try:
            self.logger.info("Setting up cleaner agent")
            self.cleaner_agent = Agent(
                role='Autonomous Data Cleaner',
                goal='Identify patterns and clean data automatically',
                backstory="""Expert in data cleaning who can identify patterns and
                automatically apply appropriate transformations. Specializes in
                discovering and removing unnecessary prefixes, suffixes, and
                standardizing data formats.""",
                verbose=True,
                llm=self.llm
            )
            self.logger.debug("Cleaner agent successfully initialized")
        except Exception as e:
            self.logger.error("Failed to setup cleaner agent")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset using the configured agent."""
        self.logger.info(f"Starting data cleaning process for DataFrame with shape: {df.shape}")
        cleaned_df = df.copy()

        try:
            # Log initial dataset characteristics
            self.logger.debug(f"Column types: {df.dtypes.to_dict()}")
            self.logger.debug(f"Null counts: {df.isnull().sum().to_dict()}")

            # Create and execute cleaning task
            self.logger.info("Creating cleaning task")
            cleaning_task = self._create_cleaning_task(df)
            
            self.logger.info("Executing cleaning task")
            result = self.cleaner_agent.execute_task(cleaning_task)
            self.logger.debug(f"Raw cleaning task result: {result[:200]}...")  # Log truncated result

            try:
                cleaning_data = json.loads(result)
                self.logger.info(f"Identified {len(cleaning_data.get('cleaning_operations', []))} cleaning operations")

                for operation in cleaning_data.get('cleaning_operations', []):
                    self._apply_cleaning_operation(cleaned_df, operation)

            except json.JSONDecodeError:
                self.logger.error("Failed to parse cleaning instructions JSON")
                self.logger.debug(f"Invalid JSON: {result}")

            # Validate and log results
            changes = self._validate_results(df, cleaned_df)
            self._log_cleaning_results(changes)
            
            self.logger.info("Data cleaning completed successfully")
            
        except Exception as e:
            self.logger.error("Error during cleaning process")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
        return cleaned_df

    def _create_cleaning_task(self, df: pd.DataFrame) -> Task:
        """Create a cleaning task with dataset analysis."""
        self.logger.debug("Creating cleaning task description")
        return Task(
            description=f"""
            Analyze and clean this dataset:
            Dataset Sample: {self._format_sample_data(df)}
            Instructions:

            Look for and identify:
            Consistent prefixes that appear in ALL rows of any column
            Date/time formats that need standardization
            Unnecessary formatting patterns that are repeated

            Rules for prefix/pattern removal:
            Only remove prefixes that are purely decorative and appear in ALL rows
            Do not remove prefixes that are part of meaningful words or standard terminology
            Preserve any prefix that, if removed, would change the semantic meaning
            Keep all status indicators, negations, and meaningful modifiers like (No,Non-,etc.)
            Maintain all standardized codes and identifiers(Eg:, CAPITALIZED LETTERS)

            If the whole word is same like (Yes/No ) then don't remove it.
            Format standardization:
            Identify any date/time fields that need consistent formatting
            Look for numeric or categorical fields that need standardization""",

            expected_output="""Return a JSON string with this exact structure:
                            {
                            "cleaning_operations": [
                                {
                                    "column": "column_name",
                                    "operation": "operation_type",
                                    "details": "specific_details",
                                    "pattern": "pattern_to_remove"
                                }
                            ]
                            }""",
            agent=self.cleaner_agent
        )

    def _apply_cleaning_operation(self, df: pd.DataFrame, operation: Dict) -> None:
        """Apply a single cleaning operation to the DataFrame."""
        column = operation.get('column')
        op_type = operation.get('operation')
        pattern = operation.get('pattern')
        
        if column not in df.columns:
            self.logger.warning(f"Column '{column}' not found in DataFrame")
            return
            
        self.logger.info(f"Applying {op_type} operation to column '{column}'")
        self.logger.debug(f"Operation details: {operation}")

        try:
            if op_type == 'remove_prefix' and pattern:
                df[column] = df[column].astype(str).str.replace(f"^{pattern}", '', regex=True)
                self.logger.debug(f"Removed prefix '{pattern}' from column '{column}'")
            elif op_type == 'remove_suffix' and pattern:
                df[column] = df[column].astype(str).str.replace(f"{pattern}$", '', regex=True)
                self.logger.debug(f"Removed suffix '{pattern}' from column '{column}'")
            elif op_type == 'standardize':
                df[column] = self._standardize_column(df[column], pattern)
                self.logger.debug(f"Standardized column '{column}' with pattern '{pattern}'")
        except Exception as e:
            self.logger.error(f"Failed to apply {op_type} operation to column '{column}'")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

    def _format_sample_data(self, df: pd.DataFrame) -> str:
        """Format sample data for analysis."""
        self.logger.debug("Formatting sample data for analysis")
        samples = {}
        
        for col in df.columns:
            unique_values = df[col].unique()
            sample_values = [self._convert_to_python_type(val) for val in unique_values[:5]]
            
            samples[col] = {
                "sample_values": sample_values,
                "unique_count": int(len(df[col].unique())),
                "has_nulls": bool(df[col].isnull().any()),
                "dtype": str(df[col].dtype)
            }
            
        formatted_data = json.dumps(samples, cls=EnhancedJSONEncoder)
        self.logger.debug(f"Sample data formatted successfully for {len(df.columns)} columns")
        return formatted_data

    def _standardize_column(self, series: pd.Series, pattern: str = None) -> pd.Series:
        """Standardize a column's values."""
        self.logger.debug(f"Standardizing column with pattern: {pattern}")
        
        try:
            if pattern:
                standardized = series.astype(str).str.replace(pattern, '', regex=True)
                self.logger.debug("Applied explicit pattern standardization")
                return standardized
            
            values = series.dropna().astype(str)
            starts = [re.match(r'^([A-Za-z]+(?=[A-Z0-9]))', str(x)) for x in values]
            starts = [s.group(1) for s in starts if s]

            if starts and len(set(starts)) == 1:
                prefix = starts[0]
                self.logger.info(f"Detected common prefix '{prefix}' for standardization")
                return series.astype(str).str.replace(f"^{prefix}", '', regex=True)

            self.logger.debug("No common pattern found for standardization")
            return series

        except Exception as e:
            self.logger.error("Error in column standardization")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return series

    def _validate_results(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        """Validate and collect changes made during cleaning."""
        self.logger.info("Validating cleaning results")
        changes = {}

        for col in original_df.columns:
            changed_mask = original_df[col] != cleaned_df[col]
            if changed_mask.any():
                changes[col] = {
                    'count': int(changed_mask.sum()),
                    'examples': {
                        'before': original_df[col][changed_mask].head().tolist(),
                        'after': cleaned_df[col][changed_mask].head().tolist()
                    }
                }
                
        return changes

    def _log_cleaning_results(self, changes: Dict) -> None:
        """Log the results of the cleaning process."""
        if not changes:
            self.logger.info("No changes were made to the dataset")
            return
            
        self.logger.info(f"Cleaning completed with changes in {len(changes)} columns:")
        for col, details in changes.items():
            self.logger.info(f"Column '{col}': {details['count']} values modified")
            self.logger.debug("Example changes (first 5):")
            for before, after in zip(details['examples']['before'],
                                   details['examples']['after']):
                self.logger.debug(f"  {before} → {after}")