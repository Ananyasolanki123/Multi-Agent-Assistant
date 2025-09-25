import pandas as pd
import io
import re
from typing import Dict, Any, Optional, Union
import os
import plotly.express as px
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy import inspect
import json
import csv

class DataIntelligenceAgent:
    """
    Agent to handle data queries on CSV/Excel files using an LLM to generate natural language answers.
    """

    def __init__(self):
        """Initializes the agent, database connection, and LLM."""
        self.db_path = "backend\data\database.db"
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.table_name = None
        # Initialize the ChatGroq client using the API key from the environment
        # The agent relies on this LLM for intelligent query parsing.
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        # Load the latest data from the database on init if it exists
        self.df = self.load_latest_data_from_db()

    def load_latest_data_from_db(self) -> Optional[pd.DataFrame]:
        """
        Loads the most recently created table from the SQLite database into a DataFrame.
        This provides persistence across agent sessions.
        """
        try:
            if not os.path.exists(self.db_path):
                return None
            
            with self.engine.connect() as conn:
                inspector = inspect(self.engine)
                tables = inspector.get_table_names()
                if not tables:
                    return None
                
                # Assume the last table created is the one to use
                self.table_name = tables[-1]
                df = pd.read_sql_table(self.table_name, self.engine)
                return self._clean_data(df)

        except Exception as e:
            print(f"Error loading data from database: {e}")
            return None

    def load_data(self, file_path: str, file_type: str, file_name: str) -> str:
        """
        Loads, cleans, and stores a CSV or Excel file in a SQLite database.
        
        Args:
            file_path: The path to the uploaded file.
            file_type: The type of file ('csv' or 'xlsx').
            file_name: The original name of the uploaded file.
        
        Returns:
            A status message indicating the result of the operation.
        """
        self.original_file_name = file_name
        try:
            if file_type == 'csv':
                # Automatically detect the delimiter for robustness
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        sample = f.read(1024)
                        dialect = csv.Sniffer().sniff(sample)
                        self.df = pd.read_csv(file_path, delimiter=dialect.delimiter, on_bad_lines='skip')
                except (csv.Error, pd.errors.ParserError):
                    self.df = pd.read_csv(file_path, on_bad_lines='skip')

            elif file_type == 'xlsx':
                self.df = pd.read_excel(file_path)
            else:
                return "Unsupported file type. Please upload a CSV or Excel file."
            
            # Check if the dataframe is empty after loading
            if self.df.empty:
                return "The uploaded file is empty or contains no valid data. Please check your file and try again."
                
            self.df = self._clean_data(self.df)
            
            # Save the DataFrame to a new, timestamped table in the SQLite database
            self.table_name = f"data_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            self.df.to_sql(self.table_name, self.engine, index=False, if_exists='replace')
            
            return "File loaded and cleaned successfully. Data has been stored in the database."
        except Exception as e:
            print(f"Error loading or cleaning file: {e}")
            return f"Error loading or cleaning file: {e}"

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs a series of cleaning operations on the DataFrame.
        This includes lowercasing column names, converting numerical columns,
        and filling missing values.
        """
        df.columns = df.columns.str.lower()
        
        for col in df.columns:
            if 'sales' in col or 'revenue' in col or 'price' in col or 'cost' in col:
                df.loc[:, col] = pd.to_numeric(
                    df[col].astype(str).str.replace(' ', '').str.replace(',', ''),
                    errors='coerce'
                )
        
        for col in df.select_dtypes(include=['number']).columns:
            df.loc[:, col] = df[col].fillna(df[col].mean())
            
        for col in df.select_dtypes(include=['object']).columns:
            df.loc[:, col] = df[col].fillna('unknown')
            
        for col in df.columns:
            if df[col].dtype == 'object':
                df.loc[:, col] = df[col].astype(str).str.lower()
                df.loc[:, col] = df[col].apply(
                    lambda x: re.sub(r'[^a-z0-9\s]+', ' ', x)
                )
                df.loc[:, col] = df[col].apply(
                    lambda x: re.sub(r'\s+', ' ', x)
                )
                df.loc[:, col] = df[col].str.strip()
        
        return df

    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a natural language query by routing it to the appropriate function.
        """
        if self.df is None:
            return {
                "type": "text",
                "message": "No data has been loaded. Please upload a file first."
            }

        query_lower = query.lower()

        # Check for plotting queries
        if "plot" in query_lower or "chart" in query_lower or "graph" in query_lower:
            return self._generate_dynamic_plot(query)
        
        # All other queries are sent to the LLM for a numerical extraction or general answer
        return self._execute_numerical_query(query)

    def _execute_numerical_query(self, user_query: str) -> Dict[str, Any]:
        """
        Performs a precise numerical calculation based on the user's query using an LLM to parse the query.
        """
        class NumericalQueryInfo(BaseModel):
            query_type: str = Field(description="The type of user query, either 'numerical', 'categorical', or 'general'.")
            operation: Optional[str] = Field(description="The mathematical operation to perform, e.g., 'sum', 'mean', 'count', 'min', 'max'.")
            column: Optional[str] = Field(description="The name of the column on which to perform the operation.")
            error: Optional[str] = Field(description="An error message if a suitable operation and column cannot be identified. Must be populated if parsing fails.")

        parser = PydanticOutputParser(pydantic_object=NumericalQueryInfo)

        prompt_template = PromptTemplate(
            template="""
            You are a data analyst expert in understanding user queries about data.
            The user wants to perform an operation on a pandas DataFrame with the following columns: {columns}.
            The user's query is: "{query}".
            
            Based on the query and available columns, identify the most suitable operation and the column to apply it to.
            Determine the query type:
            - Use 'numerical' for mathematical operations like sum, mean, min, max, or count on a numerical column.
            - Use 'categorical' for operations on categorical data, such as counting unique values or listing them.
            - Use 'general' if the query is not a data operation but a general question about the dataset.

            If you cannot identify a valid operation or column, populate the 'error' field with a clear, concise error message.

            {format_instructions}
            
            DataFrame columns: {columns}
            User query: "{query}"
            """,
            input_variables=["columns", "query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        columns = self.df.columns.tolist()
        chain = prompt_template | self.llm | parser
        
        query_info = chain.invoke({"columns": columns, "query": user_query})

        if query_info.error:
            return {"type": "text", "message": query_info.error}
            
        # Route based on the identified query type
        if query_info.query_type == 'numerical':
            operation_map = {
                "sum": "sum", "mean": "mean", "average": "mean", "avg": "mean",
                "count": "count", "min": "min", "lowest": "min", "max": "max",
                "highest": "max"
            }
            
            operation = operation_map.get(query_info.operation.lower(), query_info.operation)
            column_to_query = self._find_column(query_info.column)
            
            if not column_to_query or operation not in operation_map.values():
                return {"type": "text", "message": "Could not identify a valid numerical operation or column from your query."}

            if column_to_query not in self.df.columns:
                return {"type": "text", "message": f"Column '{column_to_query}' not found in the DataFrame."}
            
            # Perform the calculation using pandas
            try:
                if operation == "sum":
                    result = self.df[column_to_query].sum()
                elif operation == "mean":
                    result = self.df[column_to_query].mean()
                elif operation == "min":
                    result = self.df[column_to_query].min()
                elif operation == "max":
                    result = self.df[column_to_query].max()
                elif operation == "count":
                    result = self.df[column_to_query].count()
                
                return {
                    "type": "text",
                    "message": f"The {operation} of {column_to_query} is: {result}"
                }
            except Exception as e:
                return {"type": "text", "message": f"An error occurred while performing the numerical operation: {e}"}

        elif query_info.query_type == 'categorical':
            column_to_query = self._find_column(query_info.column)
            if not column_to_query:
                return {"type": "text", "message": "Could not identify a valid categorical column from your query."}
            
            try:
                unique_values = self.df[column_to_query].unique().tolist()
                count_unique = len(unique_values)
                return {
                    "type": "text",
                    "message": f"The column '{column_to_query}' has {count_unique} unique values: {', '.join(unique_values)}"
                }
            except Exception as e:
                return {"type": "text", "message": f"An error occurred while processing the categorical query: {e}"}

        else: # General query type
            prompt = self._create_llm_prompt(user_query)
            llm_response = self.llm.invoke(prompt)
            return {
                "type": "text",
                "message": llm_response.content
            }

    def _create_llm_prompt(self, user_query: str) -> str:
        """
        Creates the prompt for the LLM, providing context about the DataFrame.
        This is a fallback for general questions that aren't numerical or plotting queries.
        """
        df_head = self.df.head().to_string()
        
        prompt = f"""
        You are a data analysis Coach expert in precise analysis without any error. Your task is to analyze the provided DataFrame and answer the user's query in natural language without including any metadata and instructions in the response.
        
        Here is the head of the DataFrame:
        {df_head}
        You should analyze each column and row of the data as an expert.
        User's query: {user_query}
        
        Answer the query based on the data provided. Be concise and helpful with no preamble.
        """
        return prompt

    def _generate_dynamic_plot(self, user_query: str) -> Dict[str, Any]:
        """
        Generates a dynamic interactive plot based on the user's query.
        It uses an LLM to parse the query into plot parameters.
        """
        class PlotInfo(BaseModel):
            x_axis: str = Field(description="The name of the column for the x-axis.")
            y_axis: Optional[str] = Field(description="The name of the column for the y-axis. Can be null if a single variable is being plotted.")
            plot_type: str = Field(description="The suggested type of plot, e.g., 'line', 'bar', 'scatter', 'histogram'.")
            error: Optional[str] = Field(description="An error message if a suitable plot cannot be identified.")

        parser = PydanticOutputParser(pydantic_object=PlotInfo)

        try:
            prompt_template = PromptTemplate(
                template="""
                You are a plotting expert. A user wants to create a plot from a pandas DataFrame.
                The DataFrame has the following columns: {columns}.
                The user's query is: "{query}".
                
                Based on the query and available columns, identify the most suitable columns for the x-axis and y-axis, and suggest a suitable plot type.
                
                If the query mentions a date or time-related column, use a line plot.
                If the query asks for a count or distribution of a single categorical column, use a bar or histogram.
                If the query compares two numerical values, use a scatter plot.

                {format_instructions}
                
                DataFrame columns: {columns}
                User query: "{query}"
                """,
                input_variables=["columns", "query"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            chain = prompt_template | self.llm | parser
            
            columns = self.df.columns.tolist()
            plot_info = chain.invoke({"columns": columns, "query": user_query})
            
            if plot_info.error:
                return {
                    "type": "text",
                    "message": plot_info.error
                }
            
            x_col_llm = plot_info.x_axis
            y_col_llm = plot_info.y_axis
            plot_type = plot_info.plot_type
            
            # Normalize column names for robust lookup
            x_col = self._find_column(x_col_llm)
            y_col = self._find_column(y_col_llm) if y_col_llm else None

            # Verify normalized columns exist
            if x_col is None or (y_col_llm and y_col is None):
                return {
                    "type": "text",
                    "message": f"Could not find the requested columns ({x_col_llm}, {y_col_llm}) in the data."
                }

            df_to_plot = self.df.copy()

            if plot_type == 'line':
                if pd.api.types.is_numeric_dtype(df_to_plot[x_col]):
                    df_to_plot[x_col] = pd.to_datetime(df_to_plot[x_col], errors='coerce')
                    df_to_plot.dropna(subset=[x_col], inplace=True)
                
                fig = px.line(df_to_plot, 
                                 x=x_col, 
                                 y=y_col, 
                                 title=f'{y_col.capitalize()} Trends over {x_col.capitalize()}',
                                 labels={x_col: x_col.capitalize(), y_col: y_col.capitalize()})
            elif plot_type == 'bar':
                if y_col and y_col in self.df.columns:
                    grouped_data = df_to_plot.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.bar(grouped_data, 
                                     x=x_col, 
                                     y=y_col, 
                                     title=f'Total {y_col.capitalize()} by {x_col.capitalize()}',
                                     labels={x_col: x_col.capitalize(), y_col: y_col.capitalize()})
                else:
                    counts = df_to_plot[x_col].value_counts().reset_index()
                    counts.columns = [x_col, 'count']
                    fig = px.bar(counts, 
                                     x=x_col, 
                                     y='count', 
                                     title=f'Count of {x_col.capitalize()}',
                                     labels={x_col: x_col.capitalize(), 'count': 'Count'})
            elif plot_type == 'scatter':
                fig = px.scatter(df_to_plot, 
                                     x=x_col, 
                                     y=y_col, 
                                     title=f'{y_col.capitalize()} vs {x_col.capitalize()}',
                                     labels={x_col: x_col.capitalize(), y_col: y_col.capitalize()})
            elif plot_type == 'histogram':
                fig = px.histogram(df_to_plot, 
                                         x=x_col, 
                                         title=f'Distribution of {x_col.capitalize()}',
                                         labels={x_col: x_col.capitalize()})
            else:
                return {
                    "type": "text",
                    "message": f"Unsupported plot type: {plot_type}. Please try again."
                }
            
            # Return the plot as a JSON object, which can be rendered by a frontend
            return {
                "type": "plot",
                "message": f"Here is an interactive plot of the {x_col} and {y_col} data.",
                "data": fig.to_json()
            }

        except Exception as e:
            return {
                "type": "text",
                "message": f"An error occurred while generating the chart: {e}"
            }

    def _find_column(self, name: str) -> Optional[str]:
        """
        Finds a column in the DataFrame that matches a given name, ignoring case and using fuzzy matching.
        """
        if not name or self.df is None:
            return None
        
        name_lower = name.lower()
        for col in self.df.columns:
            if name_lower in col.lower():
                return col
        
        return None
