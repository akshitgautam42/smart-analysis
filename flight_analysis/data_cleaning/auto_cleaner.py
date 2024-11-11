from typing import Any
import json
import numpy as np
import pandas as pd
from crewai import Agent, Crew, Task
from langchain.chat_models import ChatOpenAI
import re
import traceback
from ..config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE

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
        self.llm = ChatOpenAI(
        temperature=TEMPERATURE,
        model_name=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
        self.setup_agent()

    def setup_agent(self):
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

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()

        try:
            cleaning_task = Task(
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

            result = self.cleaner_agent.execute_task(cleaning_task)

            try:
                cleaning_data = json.loads(result)

                for operation in cleaning_data.get('cleaning_operations', []):
                    column = operation.get('column')
                    op_type = operation.get('operation')
                    pattern = operation.get('pattern')

                    if column in cleaned_df.columns:
                        if op_type == 'remove_prefix' and pattern:
                            cleaned_df[column] = cleaned_df[column].astype(str).str.replace(f"^{pattern}", '', regex=True)
                        elif op_type == 'remove_suffix' and pattern:
                            cleaned_df[column] = cleaned_df[column].astype(str).str.replace(f"{pattern}$", '', regex=True)
                        elif op_type == 'standardize':
                            cleaned_df[column] = self._standardize_column(cleaned_df[column], pattern)

            except json.JSONDecodeError:
                print("Error parsing cleaning instructions")

            self._validate_results(df, cleaned_df)

        except Exception as e:
            print(f"Error during cleaning process: {e}")
            traceback.print_exc()

        return cleaned_df

    def _format_sample_data(self, df: pd.DataFrame) -> str:
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

        return json.dumps(samples, cls=EnhancedJSONEncoder)

    def _convert_to_python_type(self, val: Any) -> Any:
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        elif isinstance(val, np.bool_):
            return bool(val)
        elif pd.isna(val):
            return None
        elif isinstance(val, (str, int, float, bool)):
            return val
        return str(val)

    def _standardize_column(self, series: pd.Series, pattern: str = None) -> pd.Series:
        try:
            if pattern:
                return series.astype(str).str.replace(pattern, '', regex=True)
            else:
                values = series.dropna().astype(str)

                starts = [re.match(r'^([A-Za-z]+(?=[A-Z0-9]))', str(x)) for x in values]
                starts = [s.group(1) for s in starts if s]

                if starts and len(set(starts)) == 1:
                    return series.astype(str).str.replace(f"^{starts[0]}", '', regex=True)

                return series

        except Exception as e:
            print(f"Error in standardization: {e}")
            return series

    def _validate_results(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        changes = {}

        for col in original_df.columns:
            changed_mask = original_df[col] != cleaned_df[col]
            if changed_mask.any():
                changes[col] = {
                    'count': changed_mask.sum(),
                    'examples': {
                        'before': original_df[col][changed_mask].head().tolist(),
                        'after': cleaned_df[col][changed_mask].head().tolist()
                    }
                }

        if changes:
            print("\nChanges Made:")
            for col, details in changes.items():
                print(f"\n{col}:")
                print(f"Modified {details['count']} values")
                print("Example changes (first 5):")
                for before, after in zip(details['examples']['before'],
                                      details['examples']['after']):
                    print(f"  {before} → {after}")