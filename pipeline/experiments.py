import os
import re
from collections import Counter

import mlflow

from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.reasoner import NeuroSymbolicReasoner
from pipeline.rules import SymbolicValidator

BENCHMARK_CACHE = None


def normalize_text(text):
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction, target):
    return normalize_text(prediction) == normalize_text(target)


def f1_score(prediction, target):
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_question_answer_pairs(kg):
    examples = []
    for triple in kg.get_all_triples():
        subject = triple["subject"]
        relation = triple["relation"]
        obj = triple["object"]

        if relation == "invented":
            question = f"Who invented the {obj}?"
            answer = f"{subject} invented the {obj}."
        elif relation == "developed by":
            question = f"Who developed {subject}?"
            answer = f"{subject} was developed by {obj}."
        elif relation == "located in":
            question = f"Where is the {subject} located?"
            answer = f"The {subject} is located in {obj}."
        elif relation == "capital of":
            question = f"What is the capital of {obj}?"
            answer = f"{subject} is the capital of {obj}."
        elif relation == "discovered":
            question = f"Who discovered {obj}?"
            answer = f"{subject} discovered {obj}."
        elif relation == "formulated":
            question = f"Who formulated {obj}?"
            answer = f"{subject} formulated {obj}."
        else:
            continue

        examples.append({
            "question": question,
            "answer": answer,
            "subject": subject,
            "relation": relation,
            "object": obj,
        })

    return examples


def build_retrieval_context(question, examples):
    question_lower = question.lower()
    for example in examples:
        if example["subject"].lower() in question_lower or example["object"].lower() in question_lower:
            return f"FACT: {example['subject']} {example['relation']} {example['object']}."
    return ""


def generate_benchmark_metrics():
    global BENCHMARK_CACHE
    if BENCHMARK_CACHE is not None:
        return BENCHMARK_CACHE

    kg = KnowledgeGraph()
    validator = SymbolicValidator(kg)
    # Keep benchmark results reproducible even when a Gemini key is configured.
    reasoner = NeuroSymbolicReasoner(kg, validator, use_live_model=False)
    examples = generate_question_answer_pairs(kg)

    results = {
        "LLM Alone": {"correct": 0, "f1": 0.0},
        "LLM + KG Retrieval": {"correct": 0, "f1": 0.0},
        "SynapseQA (LLM + KG + Symbolic)": {"correct": 0, "f1": 0.0},
    }
    symbolic_catches = 0
    symbolic_changes = 0

    for example in examples:
        question = example["question"]
        truth = example["answer"]

        retrieval_context = build_retrieval_context(question, examples)

        candidate_alone, _ = reasoner._query_llm_candidate(question)
        candidate_with_kg, _ = reasoner._query_llm_candidate(
            question,
            system_context=retrieval_context,
        )
        final_result = reasoner.reason(question)
        candidate_grounded = final_result.get("final_answer", "")
        if final_result.get("hallucination_caught"):
            symbolic_catches += 1
        if candidate_grounded != candidate_alone:
            symbolic_changes += 1

        for config, prediction in [
            ("LLM Alone", candidate_alone),
            ("LLM + KG Retrieval", candidate_with_kg),
            ("SynapseQA (LLM + KG + Symbolic)", candidate_grounded),
        ]:
            if exact_match(prediction, truth):
                results[config]["correct"] += 1
            results[config]["f1"] += f1_score(prediction, truth)

    total = len(examples)
    for config in results:
        results[config]["exact_match"] = results[config]["correct"] / total if total else 0.0
        results[config]["f1_score"] = results[config]["f1"] / total if total else 0.0

    BENCHMARK_CACHE = results
    results["_diagnostics"] = {
        "symbolic_catches": symbolic_catches,
        "symbolic_changes": symbolic_changes,
        "total_examples": total,
    }
    return results


def run_evaluation():
    """Computes benchmark metrics from the KG-backed question set and logs them with MLflow."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("SynapseQA Accuracy Evaluation")

    metrics = generate_benchmark_metrics()

    # Log each configuration as its own tracked run
    for config_name, config_values in metrics.items():
        if config_name.startswith("_"):
            continue
        run_name = config_name
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("architecture", "neuro_symbolic" if "Symbolic" in config_name else "neural_retrieval" if "KG" in config_name else "neural_alone")
            mlflow.log_param("kg_retrieval", "KG" in config_name)
            mlflow.log_param("symbolic_validation", "Symbolic" in config_name)
            mlflow.log_param("loop_retry_enabled", "Symbolic" in config_name)
            mlflow.log_metric("ExactMatch", config_values["exact_match"])
            mlflow.log_metric("F1_Score", config_values["f1_score"])
            print(f"Logged Configuration: {run_name}")

    return metrics


# Backward compatibility alias for existing import patterns
run_mlflow_experiments = run_evaluation


def get_benchmark_table():
    """Returns the current benchmark metrics as a frontend-ready table."""
    metrics = generate_benchmark_metrics()
    return [
        {
            "configuration": "LLM Alone",
            "exact_match": f"{metrics['LLM Alone']['exact_match'] * 100:.1f}%",
            "f1_score": f"{metrics['LLM Alone']['f1_score'] * 100:.1f}%",
            "hallucinations": "High"
        },
        {
            "configuration": "LLM + KG Retrieval",
            "exact_match": f"{metrics['LLM + KG Retrieval']['exact_match'] * 100:.1f}%",
            "f1_score": f"{metrics['LLM + KG Retrieval']['f1_score'] * 100:.1f}%",
            "hallucinations": "Moderate"
        },
        {
            "configuration": "LLM + KG + Symbolic (SynapseQA)",
            "exact_match": f"{metrics['SynapseQA (LLM + KG + Symbolic)']['exact_match'] * 100:.1f}%",
            "f1_score": f"{metrics['SynapseQA (LLM + KG + Symbolic)']['f1_score'] * 100:.1f}%",
            "hallucinations": "Contradictions Checked"
        },
    ]


if __name__ == "__main__":
    metrics = generate_benchmark_metrics()
    for row in get_benchmark_table():
        print(
            f"{row['configuration']}: "
            f"EM {row['exact_match']} | F1 {row['f1_score']} | "
            f"{row['hallucinations']}"
        )
    print(
        "Symbolic diagnostics: "
        f"{metrics['_diagnostics']['symbolic_catches']} catches, "
        f"{metrics['_diagnostics']['symbolic_changes']} changed outputs "
        f"out of {metrics['_diagnostics']['total_examples']} examples"
    )
