import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app
from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.rules import SymbolicValidator
from pipeline.reasoner import NeuroSymbolicReasoner

client = TestClient(app)

@pytest.fixture
def qa_setup():
    kg = KnowledgeGraph()
    validator = SymbolicValidator(kg)
    reasoner = NeuroSymbolicReasoner(kg, validator)
    return kg, validator, reasoner

def test_knowledge_graph_triples(qa_setup):
    kg, _, _ = qa_setup
    
    # Check existings facts
    is_verified, correct_objs = kg.verify_triple("Alexander Graham Bell", "invented", "Telephone")
    assert is_verified is True
    assert "Telephone" in correct_objs
    
    # Check incorrect facts
    is_verified_wrong, _ = kg.verify_triple("Thomas Edison", "invented", "Telephone")
    assert is_verified_wrong is False

def test_symbolic_validator_parsing(qa_setup):
    _, validator, _ = qa_setup
    
    # Verify statement matching
    statement = "Thomas Edison invented the telephone in 1876."
    triple = validator.parse_statement(statement)
    
    assert triple is not None
    assert triple["subject"] == "Thomas Edison"
    assert triple["relation"] == "invented"
    assert triple["object"] == "Telephone"

def test_hallucination_catcher(qa_setup):
    _, validator, _ = qa_setup
    
    # Verify validation logic catches contradiction
    statement = "Thomas Edison invented the telephone."
    is_valid, error_msg, correction = validator.validate_answer(statement)
    
    assert is_valid is False
    assert "contradiction" in error_msg.lower()
    assert correction["expected"] == "Telephone"

def test_reasoner_pipeline(qa_setup):
    _, _, reasoner = qa_setup
    
    # Process target question
    result = reasoner.reason("Who invented the telephone?")
    
    assert result["success"] is True
    assert result["hallucination_caught"] is True
    assert "Alexander Graham Bell" in result["final_answer"]
    assert len(result["trace"]) > 0

def test_api_endpoints():
    # Health check
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "active"
    
    # Metrics
    metrics_response = client.get("/api/metrics")
    assert metrics_response.status_code == 200
    assert len(metrics_response.json()["table"]) == 3
    
    # Query endpoint
    query_response = client.post("/api/query", json={"question": "Who invented the telephone?"})
    assert query_response.status_code == 200
    assert query_response.json()["success"] is True
    assert "Alexander Graham Bell" in query_response.json()["final_answer"]
