from typing import Dict, List
import json
import logging
import pandas as pd
import traceback
from crewai import Agent, Crew, Task
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from ..config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE
from ..utils.logging_config import get_cleaning_logger

class SmartDataMerger:
    def __init__(self):
        self.logger = get_cleaning_logger()
        self.logger.info("Initializing SmartDataMerger")
        
        try:
            self.logger.debug(f"Setting up ChatOpenAI with model: {OPENAI_MODEL}")
            self.llm = ChatOpenAI(
                temperature=TEMPERATURE,
                model_name=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            self.setup_tools()
            self.setup_agents()
            self.logger.info("SmartDataMerger initialization complete")
        except Exception as e:
            self.logger.error("Failed to initialize SmartDataMerger")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def setup_tools(self):
        """Setup tools for the agents"""
        self.logger.debug("Setting up agent tools")
        self.tools = [
            StructuredTool.from_function(
                func=self.analyze_dataframe,
                name="analyze_dataframe",
                description="Analyze DataFrame structure and content"
            ),
            StructuredTool.from_function(
                func=self.try_merge,
                name="try_merge",
                description="Attempt to merge DataFrames with given strategy"
            )
        ]
        self.logger.debug("Agent tools setup complete")

    def setup_agents(self):
        """Setup specialized agents"""
        self.logger.debug("Setting up specialized agents")
        try:
            self.schema_analyst = Agent(
                role='Data Schema Analyst',
                goal='Analyze and understand data structures',
                backstory="""Expert in analyzing data schemas and identifying relationships 
                between different datasets. Specializes in understanding column meanings 
                and potential join keys.""",
                verbose=True,
                llm=self.llm,
                tools=self.tools
            )

            self.merge_strategist = Agent(
                role='Merge Strategy Expert',
                goal='Determine optimal merge strategy',
                backstory="""Expert in determining how to merge complex datasets. 
                Specializes in identifying common keys, handling edge cases, and 
                ensuring data integrity during merges.""",
                verbose=True,
                llm=self.llm,
                tools=self.tools
            )
            self.logger.debug("Specialized agents setup complete")
        except Exception as e:
            self.logger.error("Failed to setup agents")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def analyze_dataframe(self, df_info: str) -> str:
        """Tool for analyzing DataFrame structure"""
        self.logger.debug("Analyzing DataFrame structure")
        try:
            df_dict = json.loads(df_info)
            analysis = {
                "columns": df_dict["columns"],
                "sample_values": df_dict.get("sample_values", {}),
                "potential_keys": [],
                "column_types": df_dict.get("dtypes", {})
            }
            
            # Identify potential key columns
            for col in df_dict["columns"]:
                if any(key_term in col.lower() for key_term in ['id', 'key', 'code', 'number']):
                    analysis["potential_keys"].append(col)
            
            self.logger.debug(f"Found {len(analysis['potential_keys'])} potential key columns")
            return json.dumps(analysis)
        except Exception as e:
            self.logger.error("Failed to analyze DataFrame")
            self.logger.error(f"Error details: {str(e)}")
            return json.dumps({"error": str(e)})

    def try_merge(self, merge_strategy: str) -> str:
        """Tool for attempting merge with given strategy"""
        self.logger.debug("Simulating merge with provided strategy")
        try:
            strategy = json.loads(merge_strategy)
            return json.dumps({
                "status": "simulated_success",
                "merged_rows": "simulation_only",
                "strategy_validated": True
            })
        except Exception as e:
            self.logger.error("Failed to simulate merge")
            self.logger.error(f"Error details: {str(e)}")
            return json.dumps({"error": str(e)})

    def determine_merge_strategy(self, dataframes: List[pd.DataFrame]) -> Dict:
        """Use agents to analyze and determine merge strategy"""
        self.logger.info(f"Determining merge strategy for {len(dataframes)} DataFrames")
        
        df_infos = []
        for i, df in enumerate(dataframes):
            df_info = {
                "index": i,
                "columns": df.columns.tolist(),
                "sample_values": df.head(3).to_dict(orient='records'),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "shape": df.shape
            }
            df_infos.append(df_info)
            self.logger.debug(f"DataFrame {i} info prepared: {df.shape} shape, {len(df.columns)} columns")

        try:
            crew = Crew(
                agents=[self.schema_analyst, self.merge_strategist],
                tasks=[
                    Task(
                        description=f"""Analyze these datasets and identify potential merge keys:
                        
                        Datasets: {json.dumps(df_infos)}
                        
                        Return a JSON with EXACTLY this structure:
                        {{
                            "similar_columns": [
                                {{
                                    "df1_index": 0,
                                    "df1_column": "column_name",
                                    "df2_index": 1,
                                    "df2_column": "column_name",
                                    "similarity_type": "exact/partial/semantic",
                                    "data_types": ["type1", "type2"]
                                }}
                            ],
                            "potential_keys": [
                                {{
                                    "columns": ["col1", "col2"],
                                    "match_type": "exact/similar",
                                    "requires_transformation": true/false
                                }}
                            ],
                            "data_type_issues": [
                                {{
                                    "df_index": 0,
                                    "column": "column_name",
                                    "current_type": "type1",
                                    "required_type": "type2"
                                }}
                            ]
                        }}
                        """,
                        expected_output="JSON string with detailed schema analysis",
                        agent=self.schema_analyst
                    ),
                    Task(
                        description=f"""Based on the schema analysis, create a merge strategy.
                        
                        Return a JSON with EXACTLY this structure:
                        {{
                            "transformations": [
                                {{
                                    "df_index": 0,
                                    "column": "column_name",
                                    "action": "rename/type_convert/format_date/standardize_values",
                                    "new_name": "new_column_name",
                                    "new_type": "desired_type",
                                    "format_pattern": "date_format_if_needed"
                                }}
                            ],
                            "merge_keys": ["key1", "key2"],
                            "merge_type": "inner/left/right/outer",
                            "merge_order": [0, 1, 2],
                            "post_merge_cleanup": [
                                {{
                                    "action": "drop_columns/rename/combine",
                                    "columns": ["col1", "col2"],
                                    "new_name": "if_needed"
                                }}
                            ]
                        }}
                        """,
                        expected_output="JSON string with comprehensive merge strategy",
                        agent=self.merge_strategist
                    )
                ]
            )

            self.logger.info("Executing crew tasks for merge strategy determination")
            results = crew.kickoff()
            self.logger.debug(f"Raw strategy results type: {type(results.raw)}")
            return results.raw
            
        except Exception as e:
            self.logger.error("Failed to determine merge strategy")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Execute merge based on determined strategy"""
        try:
            if not dataframes:
                self.logger.error("No DataFrames provided for merge")
                raise ValueError("No DataFrames provided")

            self.logger.info(f"Starting merge process for {len(dataframes)} DataFrames")
            
            # Get merge strategy
            strategy = self.determine_merge_strategy(dataframes)
            self.logger.debug(f"Raw strategy type: {type(strategy)}, content length: {len(str(strategy))}")
            
            # Parse strategy
            if isinstance(strategy, str):
                self.logger.debug("Parsing strategy string")
                strategy = strategy.replace("```json", "").replace("```", "").strip()
                strategy = json.loads(strategy)
            
            # Extract merge details
            merge_keys = strategy.get("merge_keys", [])
            merge_type = strategy.get("merge_type", "inner")
            transformations = strategy.get("transformations", [])
            
            self.logger.info(f"Merge strategy details: {merge_type} merge on {len(merge_keys)} keys")
            self.logger.debug(f"Transformations to apply: {len(transformations)}")
            
            # Process DataFrames
            processed_dfs = []
            for i, df in enumerate(dataframes):
                self.logger.info(f"Processing DataFrame {i}")
                processed_df = df.copy()
                df_transforms = [t for t in transformations if t.get("df_index") == i]
                
                for transform in df_transforms:
                    col = transform["column"]
                    action = transform["action"]
                    self.logger.debug(f"Applying {action} transform to column '{col}'")
                    
                    try:
                        if action == "rename":
                            new_name = transform["new_name"]
                            processed_df.rename(columns={col: new_name}, inplace=True)
                            self.logger.debug(f"Renamed column '{col}' to '{new_name}'")
                        elif action == "to_string":
                            processed_df[col] = processed_df[col].astype(str)
                            self.logger.debug(f"Converted column '{col}' to string type")
                        elif action == "to_numeric":
                            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                            self.logger.debug(f"Converted column '{col}' to numeric type")
                    except Exception as e:
                        self.logger.error(f"Failed to apply {action} transform to column '{col}'")
                        self.logger.error(f"Error details: {str(e)}")
                        raise
                
                processed_dfs.append(processed_df)
                self.logger.info(f"DataFrame {i} processing complete")
            
            # Execute merge
            self.logger.info("Starting DataFrame merge sequence")
            result_df = processed_dfs[0]
            for i, df in enumerate(processed_dfs[1:], 1):
                self.logger.info(f"Merging DataFrame {i}")
                
                # Validate merge keys
                keys_in_first = all(k in result_df.columns for k in merge_keys)
                keys_in_second = all(k in df.columns for k in merge_keys)
                
                if not (keys_in_first and keys_in_second):
                    self.logger.error("Missing merge keys in DataFrames")
                    self.logger.error(f"First DF columns: {result_df.columns.tolist()}")
                    self.logger.error(f"Second DF columns: {df.columns.tolist()}")
                    self.logger.error(f"Required keys: {merge_keys}")
                    raise ValueError("Missing merge keys in DataFrames")
                
                result_df = pd.merge(
                    result_df,
                    df,
                    on=merge_keys,
                    how=merge_type
                )
                self.logger.info(f"Merge {i} complete - Result shape: {result_df.shape}")
            
            self.logger.info("Merge process completed successfully")
            return result_df
                
        except Exception as e:
            self.logger.error("Critical error during merge process")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise