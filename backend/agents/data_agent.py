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
    def __init__(self):
        self.db_path = "backend/data/database.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.table_name = None
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        self.df = self.load_latest_data_from_db()

    def load_latest_data_from_db(self) -> Optional[pd.DataFrame]:
        # Tries to load the last saved DataFrame from the SQLite DB.
        try:
            if not os.path.exists(self.db_path):
                return None
            
            with self.engine.connect() as conn:
                inspector = inspect(self.engine)
                tables = inspector.get_table_names()
                if not tables:
                    return None
                
                # We'll just grab the most recently created table
                self.table_name = tables[-1]
                df = pd.read_sql_table(self.table_name, self.engine)
                return self._clean_data(df)

        except Exception as e:
            print(f"Error loading data from database: {e}")
            return None

    def load_data(self, file_path: str, file_type: str, file_name: str) -> str:
        """Loads and cleans a CSV or Excel file, then stores it in a SQLite db."""
        self.original_file_name = file_name
        try:
            if file_type == 'csv':
                # Trying to auto-detect the delimiter
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        sample = f.read(1024)
                        dialect = csv.Sniffer().sniff(sample)
                        self.df = pd.read_csv(file_path, delimiter=dialect.delimiter, on_bad_lines='skip')
                except (csv.Error, pd.errors.ParserError):
                    # Fallback if auto-detection fails
                    self.df = pd.read_csv(file_path, on_bad_lines='skip')

            elif file_type == 'xlsx':
                self.df = pd.read_excel(file_path)
            else:
                return "Unsupported file type. Please upload a CSV or Excel file."
            
            # Make sure the dataframe isn't empty
            if self.df.empty:
                return "The uploaded file is empty or contains no valid data. Please check your file and try again."
            
            self.df = self._clean_data(self.df)
            
            # Save the DataFrame to a new table with a timestamp
            self.table_name = f"data_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            self.df.to_sql(self.table_name, self.engine, index=False, if_exists='replace')
            
            return "File loaded and cleaned successfully. Data has been stored in the database."
        except Exception as e:
            print(f"Error loading or cleaning file: {e}")
            return f"Error loading or cleaning file: {e}"

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """A quick function to clean up the dataframe."""
        df.columns = df.columns.str.lower()
        
        # Clean up numerical columns
        for col in df.columns:
            if 'sales' in col or 'revenue' in col or 'price' in col or 'cost' in col:
                df.loc[:, col] = pd.to_numeric(
                    df[col].astype(str).str.replace(' ', '').str.replace(',', ''),
                    errors='coerce'
                )
        
        # Fill missing values
        for col in df.select_dtypes(include=['number']).columns:
            df.loc[:, col] = df[col].fillna(df[col].mean())
            
        for col in df.select_dtypes(include=['object']).columns:
            df.loc[:, col] = df[col].fillna('unknown')
            
        # Clean up text columns
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
        """Routes the user's query to the right function."""
        if self.df is None:
            return {
                "type": "text",
                "message": "No data has been loaded. Please upload a file first."
            }

        query_lower = query.lower()

        # Check for plotting queries first
        if any(keyword in query_lower for keyword in ["plot", "chart", "graph"]):
            return self._generate_dynamic_plot(query)
        
        # Otherwise, assume it's a numerical query
        return self._execute_numerical_query(query)

    def _execute_numerical_query(self, query: str) -> Dict[str, Any]:
        """Performs a precise numerical calculation based on the user's query."""
        
        # A Pydantic model to help the LLM structure its output
        class NumericalQueryInfo(BaseModel):
            query_type: str = Field(description="The type of user query, either 'numerical', 'categorical', or 'general'.")
            operation: Optional[str] = Field(description="The mathematical operation to perform, e.g., 'sum', 'mean', 'count', 'min', 'max'.")
            column: Optional[str] = Field(description="The name of the column on which to perform the operation.")
            error: Optional[str] = Field(description="An error message if a suitable operation and column cannot be identified.")

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

            If you can't figure out a valid operation or column, fill the 'error' field with a clear, concise error message.

            {format_instructions}
            
            DataFrame columns: {columns}
            User query: "{query}"
            """,
            input_variables=["columns", "query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        columns = self.df.columns.tolist()
        chain = prompt_template | self.llm | parser
        
        query_info = chain.invoke({"columns": columns, "query": query})

        if query_info.error:
            return {"type": "text", "message": query_info.error}
            
        # Route the request based on the query type
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
            
            # Perform the calculation
            try:
                result = getattr(self.df[column_to_query], operation)()
                
                return {
                    "type": "text", 
                    "message": f"The {operation} of {column_to_query} is: {result}"
                }
            except Exception as e:
                return {
                    "type": "text", 
                    "message": f"An error occurred while performing the numerical operation: {e}"
                }

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
                return {
                    "type": "text",
                    "message": f"An error occurred while processing the categorical query: {e}"
                }

        else: # General query type
            prompt = self._create_llm_prompt(query)
            llm_response = self.llm.invoke(prompt)
            return {"type": "text", "message": llm_response.content.strip()}


    def _create_llm_prompt(self, query: str) -> str:
        """Sets up the prompt for the LLM for general, non-numerical questions."""
        df_head = self.df.to_string()
        
        prompt = f"""
        You are a data analysis Coach expert in precise analysis without any error. Your task is to analyze the provided DataFrame and answer the user's query in natural language without including any metadata and instructions in the response.
        Here is the head of the DataFrame:
        {df_head}
        You should analyze each column and row of the data as an expert.
        User's query: {query}
        
        Answer the query based on the data provided. Be concise and helpful with no preamble.
        """
        return prompt

    def _generate_dynamic_plot(self, query: str) -> Dict[str, Any]:
        """Generates an interactive plot based on the user's query."""
        
        # A Pydantic model to help the LLM structure its plot output
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
                
                - If the query mentions a date or time-related column, use a line plot.
                - If the query asks for a count or distribution of a single categorical column, use a bar or histogram.
                - If the query compares two numerical values, use a scatter plot.

                {format_instructions}
                
                DataFrame columns: {columns}
                User query: "{query}"
                """,
                input_variables=["columns", "query"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            chain = prompt_template | self.llm | parser
            
            columns = self.df.columns.tolist()
            plot_info = chain.invoke({"columns": columns, "query": query})
            
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
            
            # Convert plot to JSON for the frontend
            return {"type": "plot", "plot": fig.to_json(), "caption": f"Here is an interactive plot of {x_col} and {y_col}."}

        except Exception as e:
            return {
                "type": "text",
                "message": f"An error occurred while generating the chart: {e}"
            }

    def _find_column(self, name: str) -> Optional[str]:
        if not name or self.df is None:
            return None
        
        name_lower = name.lower()
        for col in self.df.columns:
            if name_lower in col.lower():
                return col
        
        return None
