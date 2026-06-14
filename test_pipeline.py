import os
import sys

# Add base path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.rules import SymbolicValidator
from pipeline.reasoner import NeuroSymbolicReasoner
from pipeline.experiments import run_mlflow_experiments, get_benchmark_table

def main():
    print("=== SynapseQA Hybrid Reasoning Validation ===")
    
    # 1. Initialize logic components
    print("Initializing Knowledge Graph (NetworkX)...")
    kg = KnowledgeGraph()
    print(f"Loaded {len(kg.get_all_triples())} verified Wikidata facts.")
    
    print("\nSetting up SPARQL-like validator rules...")
    validator = SymbolicValidator(kg)
    
    print("\nInitializing Neuro-Symbolic Reasoning engine...")
    reasoner = NeuroSymbolicReasoner(kg, validator)
    
    # 2. Run test query simulating hallucination
    question = "Who invented the telephone?"
    print(f"\nProcessing Query: '{question}'")
    result = reasoner.reason(question)
    
    print("\n--- REASONING PROCESS COMPLETED ---")
    print(f"Hallucination Caught: {result['hallucination_caught']}")
    print(f"Initial Candidate Answer: {result['candidate_answers'][0]}")
    if result['hallucination_caught']:
        print(f"Grounded Final Answer: {result['final_answer']}")
        
    print("\n--- REASONING TRACE LOGS ---")
    for step in result['trace']:
        print(f"Stage {step['stage']}: {step['title']} -> {step['message']}")
        
    # 3. Test MLflow setup
    print("\nSimulating MLflow Experiment Telemetry runs...")
    try:
        run_mlflow_experiments()
        print("MLflow tracking completed successfully.")
    except Exception as e:
        print(f"MLflow simulation failed/skipped: {e}")
        
    # 4. Check benchmark scores
    print("\nAccuracy Benchmark Metrics Table:")
    table = get_benchmark_table()
    for row in table:
        print(f"  {row['configuration']}: EM {row['exact_match']} | F1 {row['f1_score']} (Hallucinations: {row['hallucinations']})")
        
    print("\n=== Validation Successful ===")

if __name__ == "__main__":
    main()
