from typing import Dict, Any, List, Tuple
from contextlib import contextmanager
import sqlite3
import logging
import traceback
from datetime import datetime
import pandas as pd
from crewai import Agent, Crew, Task
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from tabulate import tabulate
from fuzzywuzzy import fuzz, process
import re
from ..config import (
    OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE,
    DEFAULT_DB_PATH
)
from ..utils.logging_config import get_analysis_logger

class DataAnalyst:
    def __init__(self):
        self.logger = get_analysis_logger()
        self.logger.info("Initializing DataAnalyst")
        
        try:
            self.db_path = DEFAULT_DB_PATH
            self.logger.debug(f"Database path: {self.db_path}")
            
            self.logger.debug(f"Initializing ChatOpenAI with model: {OPENAI_MODEL}")
            self.llm = ChatOpenAI(
                temperature=TEMPERATURE,
                model_name=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            
            self.similarity_threshold = 80
            self.setup_tools()
            self.setup_agents()
            self.current_table = None
            self.logger.info("DataAnalyst initialization completed successfully")
        except Exception as e:
            self.logger.error("Failed to initialize DataAnalyst")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    @contextmanager
    def get_connection(self):
        """Get database connection with enhanced logging"""
        conn = None
        try:
            self.logger.debug(f"Establishing database connection to: {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            conn.create_function('julianday', 1, lambda x: pd.to_datetime(x).toordinal() - pd.Timestamp('1970-01-01').toordinal())
            self.logger.debug("Database connection established successfully")
            yield conn
        except sqlite3.Error as e:
            self.logger.error("SQLite error while connecting to database")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed")

    def get_table_schema(self) -> str:
        """Get schema for current table with logging"""
        self.logger.debug(f"Retrieving schema for table: {self.current_table}")
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", 
                            (self.current_table,))
                schema = cursor.fetchone()[0]
                self.logger.debug(f"Schema retrieved successfully")
                return schema
            except Exception as e:
                self.logger.error(f"Failed to retrieve schema for table: {self.current_table}")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise

    def find_similar_values(self, column: str, search_term: str) -> List[Tuple[str, int]]:
        """Find similar values using fuzzy matching with enhanced logging"""
        self.logger.debug(f"Searching for values similar to '{search_term}' in column '{column}'")
        
        with self.get_connection() as conn:
            try:
                query = f'SELECT DISTINCT "{column}" FROM {self.current_table} WHERE "{column}" IS NOT NULL'
                self.logger.debug(f"Executing query: {query}")
                
                df = pd.read_sql_query(query, conn)
                unique_values = df[column].astype(str).tolist()
                
                if not unique_values:
                    self.logger.warning(f"No values found in column {column}")
                    return []

                matches = process.extract(
                    search_term.lower(),
                    unique_values,
                    scorer=fuzz.ratio,
                    limit=5
                )

                good_matches = [(match, score) for match, score in matches 
                               if score >= self.similarity_threshold]
                
                self.logger.info(f"Found {len(good_matches)} matches above threshold {self.similarity_threshold}")
                for match, score in good_matches:
                    self.logger.debug(f"Match: '{match}' with score {score}")
                
                return good_matches

            except Exception as e:
                self.logger.error(f"Error finding similar values")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                return []

    def enhance_query_with_fuzzy_match(self, query: str) -> str:
        """Enhance SQL query with fuzzy matching and logging"""
        self.logger.debug("Starting query enhancement with fuzzy matching")
        self.logger.debug(f"Original query: {query}")
        
        if not query:
            return query
            
        try:
            where_start = query.lower().find('where')
            if where_start == -1:
                self.logger.debug("No WHERE clause found, returning original query")
                return query

            pre_where = query[:where_start]
            where_clause = query[where_start:]
            
            self.logger.debug(f"Processing WHERE clause: {where_clause}")
            conditions = re.finditer(r'(["\w]+)\s*=\s*["\']([^"\']+)["\']', where_clause)

            modified_where = where_clause
            for match in conditions:
                column, value = match.groups()
                self.logger.debug(f"Processing condition: {column} = {value}")
                
                similar_values = self.find_similar_values(column, value)
                if similar_values:
                    new_conditions = [
                        f"LOWER({column}) = LOWER('{val}')"
                        for val, score in similar_values
                    ]
                    new_condition = f"({' OR '.join(new_conditions)})"
                    
                    modified_where = modified_where.replace(match.group(0), new_condition)
                    self.logger.debug(f"Modified condition to: {new_condition}")

            enhanced_query = pre_where + modified_where
            self.logger.debug(f"Enhanced query: {enhanced_query}")
            return enhanced_query

        except Exception as e:
            self.logger.error("Failed to enhance query with fuzzy matching")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return query

    def execute_sql(self, query: str) -> str:
        """Execute SQL query with enhanced logging"""
        self.logger.info("Executing SQL query")
        self.logger.debug(f"Original query: {query}")
        
        try:
            enhanced_query = self.enhance_query_with_fuzzy_match(query)
            self.logger.debug(f"Enhanced query: {enhanced_query}")
                
            with self.get_connection() as conn:
                enhanced_query = enhanced_query.replace('flight_data', self.current_table)
                self.logger.debug(f"Final query: {enhanced_query}")
                
                df = pd.read_sql_query(enhanced_query, conn)
                self.logger.info(f"Query executed successfully. Result shape: {df.shape}")
                
                if df.empty:
                    self.logger.warning("Query returned no results")
                    return "No data found for this query."
                
                # Process date/time and delay columns
                for col in df.columns:
                    try:
                        if 'date' in col.lower() or 'time' in col.lower():
                            self.logger.debug(f"Converting datetime column: {col}")
                            df[col] = pd.to_datetime(df[col])
                            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M')
                        
                        if 'delay' in col.lower():
                            self.logger.debug(f"Formatting delay column: {col}")
                            df[col] = df[col].apply(lambda x: f"{int(x//60)}h {int(x%60)}m" if pd.notnull(x) else "N/A")
                    except Exception as e:
                        self.logger.warning(f"Failed to process column {col}")
                        self.logger.warning(f"Error details: {str(e)}")
                
                formatted_result = tabulate(df, headers='keys', tablefmt='pretty', showindex=False)
                return formatted_result
                
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return error_msg

    def setup_tools(self):
        """Setup analysis tools with logging"""
        self.logger.debug("Setting up analysis tools")
        try:
            self.tools = [
                StructuredTool.from_function(
                    func=self.get_table_schema,
                    name="get_schema",
                    description="Get the database schema for the table"
                ),
                StructuredTool.from_function(
                    func=self.execute_sql,
                    name="execute_sql",
                    description="Execute an SQL query and return results"
                )
            ]
            self.logger.debug("Tools setup completed successfully")
        except Exception as e:
            self.logger.error("Failed to setup tools")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def setup_agents(self):
        """Setup analysis agents with logging"""
        self.logger.debug("Setting up analysis agents")
        try:
            self.query_expert = Agent(
                role='Flight Data Analyst',
                goal='Create insightful SQL queries for business analysis',
                backstory="""Expert in analyzing flight data and creating business-friendly insights. 
                Focuses on clear, actionable metrics that matter to airlines.""",
                verbose=True,
                llm=self.llm,
                tools=self.tools
            )

            self.interpreter = Agent(
                role='Business Intelligence Specialist',
                goal='Translate technical data into plain English insights',
                backstory="""Expert in translating complex data into clear, actionable insights.
                Specializes in explaining airline metrics in simple terms that any stakeholder can understand.
                Focuses on practical implications and clear recommendations.""",
                verbose=True,
                llm=self.llm,
                tools=self.tools
            )
            self.logger.debug("Agents setup completed successfully")
        except Exception as e:
            self.logger.error("Failed to setup agents")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def analyze(self, question: str, table_name: str = 'flight_data') -> Dict[str, Any]:
        """Analyze data with comprehensive logging"""
        self.logger.info(f"Starting analysis for question: {question}")
        self.logger.debug(f"Using table: {table_name}")
        
        self.current_table = table_name
        
        try:
            analysis_crew = Crew(
                agents=[self.query_expert, self.interpreter],
                tasks=[
                    Task(
                        description=f"""Analyze this business question: {question}
                        Table name: {table_name}
                        
                        Guidelines:
                        1. Use get_schema to understand the data structure.
                        2. Create SQL queries using the correct table name: {table_name}
                        3. Format dates as 'YYYY-MM-DD'
                        4. Convert time differences to hours and minutes
                        5. Use proper column aliases for readability
                        6. Round numerical values appropriately
                        7. Include relevant business context
                        
                        Return results as a JSON object with raw data and key metrics.
                        """,
                        expected_output="A JSON string containing analysis results",
                        agent=self.query_expert
                    ),
                    Task(
                        description=f"""Interpret the analysis results in plain English. Take the technical data and create:

                        1. A brief executive summary (2-3 sentences)
                        2. Key findings in simple bullet points
                        3. Data in form of a table
                        
                        Avoid technical jargon and focus on what this means for the business.
                        Write as if explaining to someone with no technical background.
                        """,
                        expected_output="A string with plain English interpretation",
                        agent=self.interpreter
                    )
                ]
            )

            self.logger.info("Executing analysis crew tasks")
            crew_output = analysis_crew.kickoff()
            insights = crew_output.raw
            
            self.logger.info("Analysis completed successfully")
            return {
                'question': question,
                'insights': insights
            }
        except Exception as e:
            self.logger.error("Analysis failed")
            self.logger.error(f"Error details: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {
                'question': question,
                'error': str(e)
            }

    def format_results(self, analysis_results: Dict[str, Any]) -> str:
        """Format analysis results with logging"""
        self.logger.debug("Formatting analysis results")
        
        if 'error' in analysis_results:
            error_msg = f"\nError in analysis: {analysis_results['error']}"
            self.logger.error(error_msg)
            return error_msg

        try:
            output = [
                f"DATA ANALYSIS REPORT\n",
            ]
            
            insights = analysis_results['insights']
            self.logger.debug(f"Processing insights type: {type(insights)}")
            
            if isinstance(insights, dict):
                if 'Executive Summary' in insights:
                    self.logger.debug("Adding executive summary")
                    output.extend([
                        f"{insights['Executive Summary']}",
                        ""
                    ])
                
                if 'Key Findings' in insights:
                    self.logger.debug("Adding key findings")
                    output.extend([
                        f"\nKEY FINDINGS:",
                        "------------"
                    ])
                    if isinstance(insights['Key Findings'], list):
                        for finding in insights['Key Findings']:
                            output.append(f"• {finding}")
                    else:
                        output.append(insights['Key Findings'])
                    output.append("")
                
            else:
                self.logger.debug("Adding raw insights string")
                output.append(str(insights))
            
            formatted_output = '\n'.join(output)
            self.logger.debug("Results formatting completed successfully")
            return formatted_output
            
        except Exception as e:
            error_msg = f"Error formatting results: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return error_msg