import os
import shutil
import uuid
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
            return {"status": f"Data file '{file.filename}' processed successfully."}

        elif file_ext in ["pdf", "docx"]:
            status = research_agent.ingest_document(file_path, file_ext)
            current_agent_type = "research"
            return {"status": f"Research file '{file.filename}' processed successfully.", "details": status}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

  

@app.post("/analyze_query")
async def analyze_query(payload: QueryPayload):
    """
    Process a user query after a file has been uploaded.
    Routed via Orchestrator to the correct agent.
    """
    global current_agent_type

    if not current_agent_type:
        raise HTTPException(status_code=400, detail="No file uploaded. Please upload a file first.")

    destination = orchestrator.route_query(payload.query)

    if destination == "data":
        if current_agent_type != "data":
            raise HTTPException(status_code=400, detail="Uploaded file is not a data file. Upload CSV/XLSX.")
        response = data_agent.handle_query(payload.query)
        return {"agent": "data", "response": response}

    elif destination == "research":
        if current_agent_type != "research":
            raise HTTPException(status_code=400, detail="Uploaded file is not a research file. Upload PDF/DOCX.")
        response = research_agent.handle_query(payload.query)
        return {"agent": "research", "response": response}

    else:
        raise HTTPException(status_code=400, detail="Unable to classify query type.")
