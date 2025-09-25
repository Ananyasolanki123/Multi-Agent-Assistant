import os
from typing import Literal
from dotenv import load_dotenv

# LangChain + Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ------------------------------
# Structured Output Schema
# ------------------------------
class RouteQuery(BaseModel):
    """Route a user query to the appropriate agent."""
    destination: Literal["research", "data"] = Field(
        ...,
        description=(
            "The agent to route the query to. "
            "'research' = document-based queries (summary, keywords, methods, literature, explanations). "
            "'data' = dataset-based queries (numerical analysis, stats, trends, visualization)."
        )
    )


# ------------------------------
# Orchestration Agent
# ------------------------------
class OrchestrationAgent:
    """
    Routes incoming user queries to the appropriate agent (ResearchAgent or DataIntelligenceAgent).
    Uses Groq's LLM (via LangChain) to classify the query.
    """

    def __init__(self):
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",  # Fast & lightweight router
            temperature=0  # Deterministic for consistent routing
        )

        # Router prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a router. Classify a user's query as either 'research' or 'data'.\n"
             "- Use 'research' for document-based queries (e.g., summarize a paper, extract methods, generate keywords).\n"
             "- Use 'data' for dataset queries (e.g., analyze sales trends, calculate averages, plot a chart).\n"
             "Return ONLY the classification in JSON format."),
            ("human", "{query}")
        ])

    def route_query(self, query: str) -> str:
        """
        Takes a user query string, runs it through the LLM router,
        and returns 'research' or 'data'.
        """
        structured_llm = self.llm.with_structured_output(RouteQuery)
        router_chain = self.prompt | structured_llm

        response: RouteQuery = router_chain.invoke({"query": query})

        # Debugging logs (optional, can remove in prod)
        print(f"[OrchestrationAgent] Query: {query}")
        print(f"[OrchestrationAgent] Routed to: {response.destination}")

        return response.destination
