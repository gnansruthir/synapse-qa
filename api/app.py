import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.rules import SymbolicValidator
from pipeline.reasoner import NeuroSymbolicReasoner
from pipeline.experiments import run_mlflow_experiments, get_benchmark_table

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Initialize pipeline
kg = KnowledgeGraph()
validator = SymbolicValidator(kg)
reasoner = NeuroSymbolicReasoner(kg, validator)

# Run MLflow runs on startup to populate local experiment repository
print("Initializing MLflow run telemetry...")
try:
    run_mlflow_experiments()
except Exception as e:
    print(f"MLflow initialization skipped/failed: {e}")

app = FastAPI(title="SynapseQA - Neuro-Symbolic QA Engine API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/query")
async def run_query(request: QueryRequest):
    """
    Accepts natural language question, processes via the Neuro-Symbolic 
    reasoner, runs SPARQL checks, and logs traces.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        result = reasoner.reason(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning process encountered an error: {str(e)}")

@app.get("/api/metrics")
def get_metrics():
    """Returns the benchmark table comparison results."""
    return {
        "success": True,
        "table": get_benchmark_table()
    }

@app.get("/api/health")
def health():
    return {"status": "active", "kg_triples": len(kg.get_all_triples())}
