from typing import Dict, Any
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

class DataAnalyst:
    def __init__(self):
        self.db_path = DEFAULT_DB_PATH
        self.llm = ChatOpenAI(
            temperature=TEMPERATURE,
            model_name=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
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

    def execute_sql(self, query: str) -> str:
        with self.get_connection() as conn:
            try:
                query = query.replace('flight_data', self.current_table)
                df = pd.read_sql_query(query, conn)
                
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
                    1. Use get_schema to understand the data structure
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
            f"\n{'='*60}",
            f"FLIGHT DATA ANALYSIS REPORT",
            f"{'='*60}",
            f"\nBUSINESS QUESTION:",
            f"{analysis_results['question']}",
            f"\n{'='*60}"
        ]
        
        insights = analysis_results['insights']
        if isinstance(insights, dict):
            if 'Executive Summary' in insights:
                output.extend([
                    f"\nEXECUTIVE SUMMARY:",
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
            
            if 'Business Implications' in insights:
                output.extend([
                    f"\nBUSINESS IMPLICATIONS:",
                    "--------------------"
                ])
                if isinstance(insights['Business Implications'], list):
                    for implication in insights['Business Implications']:
                        output.append(f"• {implication}")
                else:
                    output.append(insights['Business Implications'])
                output.append("")
            
            if 'Actionable Recommendations' in insights:
                output.extend([
                    f"\nACTIONABLE RECOMMENDATIONS:",
                    "------------------------"
                ])
                if isinstance(insights['Actionable Recommendations'], list):
                    for recommendation in insights['Actionable Recommendations']:
                        output.append(f"• {recommendation}")
                else:
                    output.append(insights['Actionable Recommendations'])
        else:
            output.append(str(insights))
        
        return '\n'.join(output)