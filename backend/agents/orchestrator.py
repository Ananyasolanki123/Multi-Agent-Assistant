import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

load_dotenv()

class RouteQuery(BaseModel):
    destination: Literal["research", "data"] = Field(
        ...,
        description=(
            "The agent to route the query to. "
            "'research' = document-based queries (summary, keywords, methods, literature, explanations). "
            "'data' = dataset-based queries (numerical analysis, stats, trends, visualization)."
        )
    )

class OrchestrationAgent:
    """
    Routes incoming user queries to the appropriate agent (ResearchAgent or DataIntelligenceAgent).
    Uses Groq's LLM (via LangChain) to classify the query.
    """
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0 
        )
        self.parser = PydanticOutputParser(pydantic_object=RouteQuery)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a router. Classify a user's query as either 'research' or 'data'.\n"
             "- Use 'research' for document-based queries (e.g., summarize a paper, extract methods, generate keywords).\n"
             "- Use 'data' for dataset queries and if query involves summarize or research based query return this type of query is not supported here. (e.g., analyze sales trends, calculate averages, plot a chart).\n"
             "Return ONLY in JSON format that matches this schema:\n{format_instructions}"
             ),
            ("human", "{query}")
        ]).partial(format_instructions=self.parser.get_format_instructions())

    def route_query(self, query: str) -> str:
        chain = self.prompt | self.llm | self.parser
        response: RouteQuery = chain.invoke({"query": query})
        print(f"[OrchestrationAgent] Query: {query}")
        print(f"[OrchestrationAgent] Routed to: {response.destination}")
        return response.destination
