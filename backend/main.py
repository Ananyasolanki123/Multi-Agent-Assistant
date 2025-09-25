import os
import shutil
import uuid
import mimetypes
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Agent imports
from agents.orchestrator import OrchestrationAgent
from agents.research_agent import ResearchAgent
from agents.data_agent import DataIntelligenceAgent

# Load env variables
load_dotenv()

app = FastAPI()

# Add CORS middleware to allow requests from the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
orchestrator = OrchestrationAgent()
data_agent = DataIntelligenceAgent()
research_agent = ResearchAgent()

# Temporary storage
UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store the latest agent type loaded (data or research)
current_agent_type = None

class QueryPayload(BaseModel):
    query: str
    file_name: str  # ✅ new field


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and preprocess a file.
    CSV/XLSX -> Data Agent
    PDF/DOCX -> Research Agent
    """
    global current_agent_type

    # Save file temporarily
    file_ext = file.filename.rsplit(".", 1)[-1].lower()
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file_ext in ["csv", "xlsx"]:
            data_agent.load_data(file_path, file_ext, file.filename)
            current_agent_type = "data"
            return {"status": f"Data file '{file.filename}' processed successfully.", "file_name": file.filename}


        elif file_ext in ["pdf", "docx"]:
            status = research_agent.ingest_document(file_path, file_ext)
            current_agent_type = "research"
            return {"status": f"Research file '{file.filename}' processed successfully.", "file_name": file.filename}


        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    finally:
        # ✅ Cleanup: Delete the temporary file after ingestion, regardless of success or failure
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/analyze_query")
async def analyze_query(payload: QueryPayload):
    """
    Process a user query after a file has been uploaded.
    Routed via Orchestrator to the correct agent.
    """
    global current_agent_type

    if not current_agent_type:
        raise HTTPException(status_code=400, detail="No file uploaded. Please upload a file first.")

    file_name = payload.file_name  # ✅ get file_name from frontend
    query = payload.query
    destination = orchestrator.route_query(query)


    if destination == "data":
        response = data_agent.handle_query(query)
        return {"agent": "data", "response": response}

    elif destination == "research":
        response = research_agent.handle_query(query)
        return {"agent": "research", "response": response}
