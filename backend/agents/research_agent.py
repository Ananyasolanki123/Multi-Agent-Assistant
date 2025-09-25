import os
import fitz
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from docx import Document

# Load environment variables from .env file
load_dotenv()

# Define the Pydantic model for the output
class Keywords(BaseModel):
    """Keywords extracted from the document."""
    keywords: List[str] = Field(..., description="List of important keywords and terms.")

class ResearchAgent:
    """
    Agent to handle research-based queries on PDF and DOCX documents.
    It encapsulates the entire process from ingestion to query handling.
    """
    def __init__(self):
        """Initializes the LLM and empty data attributes."""
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        self.vector_db = None
        self.full_text = ""
        self.docs = []

    def ingest_document(self, file_path: str, file_type: str) -> str:
        """
        Ingests a PDF or DOCX file, splits it into chunks, and stores embeddings
        in a vector database for later retrieval.
        """
        try:
            full_text = ""
            if file_type == 'pdf':
                doc = fitz.open(file_path)
                for page in doc:
                    full_text += page.get_text()
                doc.close()
            elif file_type == 'docx':
                doc = Document(file_path)
                full_text = "\n".join([para.text for para in doc.paragraphs])
            else:
                return "Unsupported file type. Please upload a PDF or DOCX file."

            if not full_text:
                return "The document is empty or could not be read."

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            self.docs = text_splitter.create_documents([full_text])
            self.full_text = full_text

            embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_db = Chroma.from_documents(documents=self.docs, embedding=embeddings_model)
            
            return "Document ingested and ready for analysis."

        except Exception as e:
            print(f"Error during document ingestion: {e}")
            return f"Error ingesting the document: {e}"

    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Determines the specific action based on the research query
        and calls the appropriate method.
        """
        if not self.vector_db:
            return {
                "type": "text",
                "message": "No document has been ingested. Please upload a PDF or DOCX file first."
            }

        query_lower = query.lower()
        
        if "summarize" in query_lower:
            return self.summarize_paper()
        elif "keywords" in query_lower:
            # Join the list of keywords into a single string for a clean response
            keywords_list = self.extract_info("keywords")
            return {
                "type": "text",
                "message": ", ".join(keywords_list)
            }
        elif "abstract" in query_lower:
            return self.summarize_abstract()
        elif "question" in query_lower or "?" in query_lower:
            return self.answer_question(query)
        else:
            return {
                "type": "text",
                "message": "Research Agent is ready, but could not determine a specific task from your query."
            }

    def summarize_paper(self) -> dict:
        """Summarizes the paper using the map_reduce chain for long documents."""
        map_prompt_template = """
        The following is a part of a document:
        "{text}"
        Based on this part, write a concise summary.
        Summary:
        """
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        combine_prompt_template = """
        You are an expert summarizer. Take the following summaries from different sections
        of a document and combine them into a single, cohesive, final summary. The final summary
        should be simple enough to be clearly understandable by anyone.

        Summaries:
        "{text}"
        
        Final Summary:
        """
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt
        )
        summary = chain.invoke({"input_documents": self.docs})["output_text"]
        return {
            "type": "text",
            "message": summary
        }

    def summarize_abstract(self) -> dict:
        """Summarizes the abstract section of the document."""
        abstract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Provide a detailed summary of the abstract of the following text."),
            ("human", "Text: {text}")
        ])
        
        chain = abstract_prompt | self.llm
        # Use the first chunk of the document, which typically contains the abstract
        result = chain.invoke({"text": self.docs[0].page_content})
        return {
            "type": "text",
            "message": result.content
        }

    def extract_info(self, info_type: str) -> list:
        """Extracts keywords or other information using a structured output format."""
        extraction_prompt_template = """
        From the following text, extract {info_type}.
        Text: {text}
        """
        extraction_prompt = PromptTemplate(template=extraction_prompt_template, input_variables=["text", "info_type"])
        
        structured_llm = self.llm.with_structured_output(Keywords)
        extraction_chain = extraction_prompt | structured_llm
        
        # Use a subset of the documents to generate keywords to improve efficiency
        result = extraction_chain.invoke({"text": self.full_text[:4000], "info_type": info_type})
        return result.keywords

    def answer_question(self, question: str) -> dict:
        """Answers a question using a retrieval chain and the vector database."""
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(),
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"question": question, "chat_history": []})
        return {"type": "text", "message": result["answer"]}
