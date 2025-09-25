import os
import fitz
import io
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
from docx import Document

load_dotenv()

class Keywords(BaseModel):
    """List of important Keywords extracted from the document."""
    keywords: List[str] = Field(..., description="List of important keywords and terms.")

class ResearchQueryType(BaseModel):
    """
    Classifies the user's query intent. The LLM will use this to decide
    which function to call (e.g., summary, Q&A, etc.).
    """
    category: Literal["summary", "keywords", "abstract", "question", "unknown"] = Field(
        description="The type of query, based on the user's request."
    )

class ResearchAgent:
    """
    An intelligent agent for analyzing research papers.
    It can ingest documents, summarize them, extract keywords, and answer questions.
    """
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        self.vector_db = None
        if os.path.exists("backend/vector_store"):
            self.vector_db = Chroma(
                embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                persist_directory="backend/vector_store"
            )

        self.full_text = ""
        self.docs = []
    
    def ingest_document(self, file_path: str, file_type: str) -> Dict[str, str]:
        """
        Loads a PDF or DOCX file, splits it into manageable chunks, and
        stores the embeddings in a Chroma vector database.
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
                return {"type": "text", "message": "Unsupported file type. Please upload a PDF or DOCX file."}
            if not full_text:
                return {"type": "text", "message": "The document is empty or could not be read."}
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            self.docs = text_splitter.create_documents([full_text])
            self.full_text = full_text
            embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_db = Chroma.from_documents(
                documents=self.docs,
                embedding=embeddings_model,
                persist_directory="backend/vector_store"
            )
            self.vector_db.persist()
            return {"type": "text", "message": "Document ingested and ready for analysis."}
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            return {"type": "text", "message": f"Error ingesting the document: {e}"}

    def handle_query(self, query: str) -> Dict[str, str]:
        """
        Routes the user's query to the correct handler function based on its intent.
        It's essentially a smart router powered by an LLM.
        """
        if not self.vector_db:
            return {"type": "text", "message": "No document has been ingested. Please upload a PDF or DOCX file first."}
        parser = PydanticOutputParser(pydantic_object=ResearchQueryType)
        prompt_template = PromptTemplate(
            template="""
            Your sole purpose is to classify a user's query for a research agent.
            The user wants to analyze a research document, and your job is to determine their intent.
            Be lenient and flexible in your classification. For example, if the user asks for a "sumary" or "summry," classify it as 'summary'.

            Classify the user's query into one of the following categories:
            - 'summary': If the query asks to summarize the entire document. Keywords include: "summary", "summarize", "overview", "main points", "short version", "gist","conclusion".
            - 'abstract': If the query specifically asks for the abstract. Keywords include: "abstract", "paper abstract", "what is the abstract".
            - 'keywords': If the query asks for keywords or key terms. Keywords include: "keywords", "key terms", "important words".
            - 'question': If the query is a question to be answered from the document. This category should be a fallback if none of the others match.
            - 'unknown': If the query is completely unrelated to the document.
            
            {format_instructions}
            
            User query: {query}
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt_template | self.llm | parser
        try:
            query_type = chain.invoke({"query": query})
            if query_type.category == "summary":
                response = self.summarize_paper()
            elif query_type.category == "keywords":
                response = self.extract_info("keywords")
            elif query_type.category == "abstract":
                response = self.summarize_abstract()
            elif query_type.category == "question":
                response = self.answer_question(query)
            else:
                response = {"type": "text", "message": "Research Agent is ready, but could not determine a specific task from your query."}
            if not isinstance(response, dict) or 'message' not in response:
                response = {"type": "text", "message": "An error occurred while processing your request. Please try again."}
            return response      
        except Exception as e:
            print(f"Error classifying research query: {e}")
            return {"type": "text", "message": "An error occurred while processing your request. Please try again."}

    def summarize_paper(self) -> Dict[str, str]:
        if not self.full_text and self.vector_db:
            all_documents = self.vector_db.get(include=['metadatas', 'documents'])
            self.docs = [doc for doc in all_documents['documents']]
            self.full_text = "\n".join(self.docs)
        if not self.full_text:
            return {"type": "text", "message": "No document content available for summarization."}
        # Use of simpler, direct prompt for summarization
        summary_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at summarizing research papers. Your task is to provide a comprehensive, clear, and concise summary of the following document. Focus on the main arguments, key findings, and conclusions. The summary should be easy to understand for a non-expert."),
            ("human", "Document: {document_content}")
        ])
        chain = summary_prompt_template | self.llm
        result = chain.invoke({"document_content": self.full_text})
        return {"type": "text", "message": result.content}

    def summarize_abstract(self) -> Dict[str, str]:
        if not self.full_text and self.vector_db:
            all_documents = self.vector_db.get(include=['metadatas', 'documents'])
            self.docs = [doc for doc in all_documents['documents']]
            self.full_text = "\n".join(self.docs)
        if not self.full_text:
            return {"type": "text", "message": "No document content available for summarization."}
        abstract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Provide a detailed summary of the abstract of the following text."),
            ("human", "Text: {text}")
        ])
        chain = abstract_prompt | self.llm
        result = chain.invoke({"text": self.docs[0]})
        return {"type": "text", "message": result.content}

    def extract_info(self, info_type: str) -> Dict[str, str]:
        if not self.full_text and self.vector_db:
            all_documents = self.vector_db.get(include=['metadatas', 'documents'])
            self.docs = [doc for doc in all_documents['documents']]
            self.full_text = "\n".join(self.docs)   
        parser = PydanticOutputParser(pydantic_object=Keywords)
        extraction_prompt_template = """
        From the following text, extract {info_type}.
        {format_instructions}
        Text: {text}
        """
        extraction_prompt = PromptTemplate(
            template=extraction_prompt_template, 
            input_variables=["text", "info_type"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        extraction_chain = extraction_prompt | self.llm | parser
        result = extraction_chain.invoke({"text": self.full_text[:4000], "info_type": info_type})
        keywords_str = ", ".join(result.keywords)
        return {"type": "text", "message": keywords_str}

    def answer_question(self, question: str) -> Dict[str, str]:
        """Answers a question using a retrieval chain and the vector database."""
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(),
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"question": question, "chat_history": []})
        return {"type": "text", "message": result["answer"]}
