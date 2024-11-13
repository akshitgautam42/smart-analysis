from typing import Dict, Any, List, Tuple
from contextlib import contextmanager
import sqlite3
import logging
from datetime import datetime
import pandas as pd
from crewai import Agent, Crew, Task
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from tabulate import tabulate
from ..config import (
    OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE,
    DEFAULT_DB_PATH
)
from fuzzywuzzy import fuzz, process

from typing import Dict, Any, List, Tuple
from contextlib import contextmanager
import sqlite3
import logging
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

class DataAnalyst:
    def __init__(self):
        self.db_path = DEFAULT_DB_PATH
        self.llm = ChatOpenAI(
            temperature=TEMPERATURE,
            model_name=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self.similarity_threshold = 80  # Threshold for fuzzy matching
        self.setup_tools()
        self.setup_agents()
        self.logger = logging.getLogger(__name__)
        self.current_table = None

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.create_function('julianday', 1, lambda x: pd.to_datetime(x).toordinal() - pd.Timestamp('1970-01-01').toordinal())
            yield conn
        finally:
            if conn:
                conn.close()

    def get_table_schema(self) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", 
                        (self.current_table,))
            return cursor.fetchone()[0]

    def find_similar_values(self, column: str, search_term: str) -> List[Tuple[str, int]]:
        """Find similar values using simple case-insensitive matching"""
        with self.get_connection() as conn:
            try:
                # Get unique values from the column
                query = f'SELECT DISTINCT "{column}" FROM {self.current_table} WHERE "{column}" IS NOT NULL'
                df = pd.read_sql_query(query, conn)
                unique_values = df[column].astype(str).tolist()

                if not unique_values:
                    self.logger.warning(f"No values found in column {column}")
                    return []

                # Simple case-insensitive matching
                matches = process.extract(
                    search_term.lower(),
                    unique_values,
                    scorer=fuzz.ratio,
                    limit=5
                )

                # Filter matches above threshold
                good_matches = [
                    (match, score) for match, score in matches 
                    if score >= self.similarity_threshold
                ]
                
                # Log matches
                if good_matches:
                    self.logger.info(f"Found {len(good_matches)} matches for '{search_term}' in {column}")
                    for match, score in good_matches:
                        self.logger.info(f"  - '{match}' (score: {score})")
                else:
                    self.logger.info(f"No matches found for '{search_term}' in {column}")

                return good_matches

            except Exception as e:
                self.logger.error(f"Error in find_similar_values: {str(e)}")
                return []

    def enhance_query_with_fuzzy_match(self, query: str) -> str:
        """Add simple fuzzy matching to WHERE clauses"""
        if not query:
            return query
            
        try:
            where_start = query.lower().find('where')
            if where_start == -1:
                return query

            pre_where = query[:where_start]
            where_clause = query[where_start:]

            # Find exact match conditions
            conditions = re.finditer(
                r'(["\w]+)\s*=\s*["\']([^"\']+)["\']', 
                where_clause
            )

            modified_where = where_clause
            for match in conditions:
                column, value = match.groups()
                similar_values = self.find_similar_values(column, value)

                if similar_values:
                    # Create simple OR conditions with LOWER() for case-insensitivity
                    new_conditions = [
                        f"LOWER({column}) = LOWER('{val}')"
                        for val, score in similar_values
                    ]
                    new_condition = f"({' OR '.join(new_conditions)})"
                    
                    # Replace the original condition
                    modified_where = modified_where.replace(
                        match.group(0),
                        new_condition
                    )

            enhanced_query = pre_where + modified_where
            if enhanced_query != query:
                self.logger.info(f"Enhanced query from:\n{query}\nto:\n{enhanced_query}")
            return enhanced_query

        except Exception as e:
            self.logger.error(f"Error in enhance_query_with_fuzzy_match: {str(e)}")
            return query

    def execute_sql(self, query: str) -> str:
        try:
            enhanced_query = self.enhance_query_with_fuzzy_match(query)
                
            # Log query changes
            if enhanced_query != query:
                self.logger.info("Original query: " + query)
                self.logger.info("Enhanced query: " + enhanced_query)
                
            with self.get_connection() as conn:
                enhanced_query = enhanced_query.replace('flight_data', self.current_table)
                df = pd.read_sql_query(enhanced_query, conn)
                
                if df.empty:
                    return "No data found for this query."
                
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            pass
                    
                    if 'delay' in col.lower():
                        df[col] = df[col].apply(lambda x: f"{int(x//60)}h {int(x%60)}m" if pd.notnull(x) else "N/A")
                
                return tabulate(df, headers='keys', tablefmt='pretty', showindex=False)
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def setup_tools(self):
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

    def setup_agents(self):
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

    def analyze(self, question: str, table_name: str = 'flight_data') -> Dict[str, Any]:
        self.current_table = table_name
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
            ],
            process_orders=[
                [0],  # First task runs alone
                [1]   # Second task runs after first
            ]
        )

        try:
            crew_output = analysis_crew.kickoff()
            insights = crew_output.raw
            
            return {
                'question': question,
                'insights': insights
            }
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return {
                'question': question,
                'error': str(e)
            }

    def format_results(self, analysis_results: Dict[str, Any]) -> str:
        if 'error' in analysis_results:
            return f"\nError in analysis: {analysis_results['error']}"

        output = [
            f"DATA ANALYSIS REPORT\n",
        ]
        
        insights = analysis_results['insights']
        if isinstance(insights, dict):
            if 'Executive Summary' in insights:
                output.extend([
                    f"{insights['Executive Summary']}",
                    ""
                ])
            
            if 'Key Findings' in insights:
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
            output.append(str(insights))
        
        return '\n'.join(output)