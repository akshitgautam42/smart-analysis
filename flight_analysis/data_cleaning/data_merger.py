from typing import Dict, List
import json
import logging
import pandas as pd
from crewai import Agent, Crew, Task
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from ..config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE


class SmartDataMerger:
    def __init__(self):
        self.llm = ChatOpenAI(
        temperature=TEMPERATURE,
        model_name=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
        self.setup_tools()
        self.setup_agents()
        self.logger = logging.getLogger(__name__)

    def setup_tools(self):
        """Setup tools for the agents"""
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

    def setup_agents(self):
        """Setup specialized agents"""
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

    def analyze_dataframe(self, df_info: str) -> str:
        """Tool for analyzing DataFrame structure"""
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
            
            return json.dumps(analysis)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def try_merge(self, merge_strategy: str) -> str:
        """Tool for attempting merge with given strategy"""
        try:
            strategy = json.loads(merge_strategy)
            return json.dumps({
                "status": "simulated_success",
                "merged_rows": "simulation_only",
                "strategy_validated": True
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def determine_merge_strategy(self, dataframes: List[pd.DataFrame]) -> Dict:
      """Use agents to analyze and determine merge strategy"""
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

      results = crew.kickoff()
      return results.raw

    def merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Execute merge based on determined strategy"""
        try:
            if not dataframes:
                raise ValueError("No DataFrames provided")

            # Get merge strategy from agents and print raw output
            strategy = self.determine_merge_strategy(dataframes)
            print("\n=== Raw Strategy Output ===")
            print(f"Type: {type(strategy)}")
            print(f"Content: {strategy}")

            # Parse strategy and print result
            # Replace with this updated parsing logic:
            if isinstance(strategy, str):
                print("\n=== Parsing Strategy String ===")
                print(f"Strategy string content: {strategy}")
                # Remove markdown code block markers if present
                strategy = strategy.replace("```json", "").replace("```", "").strip()
                strategy = json.loads(strategy)
                
            print("\n=== Parsed Strategy ===")
            print(f"Type: {type(strategy)}")
            print(f"Content: {strategy}")
            
            # Print extracted merge details
            merge_keys = strategy.get("merge_keys", [])
            merge_type = strategy.get("merge_type", "inner")
            transformations = strategy.get("transformations", [])
            
            print("\n=== Merge Details ===")
            print(f"Merge Keys: {merge_keys}")
            print(f"Merge Type: {merge_type}")
            print(f"Transformations: {transformations}")
            
            # Print DataFrame info before processing
            print("\n=== DataFrame Info ===")
            for i, df in enumerate(dataframes):
                print(f"\nDataFrame {i} columns: {df.columns.tolist()}")
                print(f"DataFrame {i} shape: {df.shape}")

            # Rest of your existing code...
            processed_dfs = []
            for i, df in enumerate(dataframes):
                processed_df = df.copy()
                for transform in transformations:
                    if transform.get("df_index") == i:
                        col = transform["column"]
                        action = transform["action"]
                        if action == "rename":
                            processed_df.rename(columns={col: transform["new_name"]}, inplace=True)
                        elif action == "to_string":
                            processed_df[col] = processed_df[col].astype(str)
                        elif action == "to_numeric":
                            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_dfs.append(processed_df)
                
            # Print processed DataFrame info
            print("\n=== Processed DataFrame Info ===")
            for i, df in enumerate(processed_dfs):
                print(f"\nProcessed DataFrame {i} columns: {df.columns.tolist()}")
            
            # Execute merge
            result_df = processed_dfs[0]
            for i, df in enumerate(processed_dfs[1:], 1):
                print(f"\n=== Attempting Merge {i} ===")
                print(f"Merge keys present in first DF: {all(k in result_df.columns for k in merge_keys)}")
                print(f"Merge keys present in second DF: {all(k in df.columns for k in merge_keys)}")
                result_df = pd.merge(
                    result_df,
                    df,
                    on=merge_keys,
                    how=merge_type
                )
                
            return result_df
                
        except Exception as e:
            print(f"Error during merge: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
