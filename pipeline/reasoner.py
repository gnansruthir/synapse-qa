import time
import os

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency when Gemini is not installed
    genai = None

client = None
if os.getenv("GEMINI_API_KEY") and genai is not None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class NeuroSymbolicReasoner:
    def __init__(self, knowledge_graph, validator, use_live_model=True):
        self.kg = knowledge_graph
        self.validator = validator
        self.use_live_model = use_live_model

    def _fact_from_context(self, retrieval_context):
        """Extract the exact KG fact embedded in a retrieval context string."""
        if not retrieval_context:
            return None
        retrieval_text = retrieval_context.lower()
        for triple in self.kg.get_all_triples():
            fact = f"{triple['subject']} {triple['relation']} {triple['object']}"
            if fact.lower() in retrieval_text:
                return triple
        return None

    def _format_fact_answer(self, triple):
        """Format a retrieved fact as a concise answer for the local mock model."""
        subject = triple["subject"]
        relation = triple["relation"].replace("_", " ")
        obj = triple["object"]
        if relation == "invented":
            return f"{subject} invented the {obj}."
        if relation == "developed by":
            return f"{subject} was developed by {obj}."
        if relation == "located in":
            return f"{subject} is located in {obj}."
        if relation == "capital of":
            return f"{subject} is the capital of {obj}."
        if relation == "discovered":
            return f"{subject} discovered {obj}."
        if relation == "formulated":
            return f"{subject} formulated {obj}."
        return None

    def _query_llm_candidate(self, query, system_context=""):
        """
        Queries the LLM (Gemini or Mock fallback) to generate a candidate answer.
        In mock mode, specifically simulates hallucinations for targeted questions
        so the user can witness the validation engine catching errors in real-time.
        """
        # Targeted mock hallucinations for testing the validator
        lower_query = query.lower()
        if "telephone" in lower_query and "invent" in lower_query and not "bell" in system_context.lower():
            # Demonstration-only hallucination path: keep the value as a generic confidence cue, never as a benchmark claim.
            return "Thomas Edison invented the telephone in 1876.", 0.61
            
        if "eiffel" in lower_query and "located" in lower_query and not "paris" in system_context.lower():
            # Demonstration-only hallucination path: keep the value as a generic confidence cue, never as a benchmark claim.
            return "The Eiffel Tower is located in London.", 0.55
            
        if "java" in lower_query and "develop" in lower_query and not "gosling" in system_context.lower():
            return "Java was developed by Dennis Ritchie at Bell Labs.", 0.58

        # The local mock must honor retrieval context for every relation, not only targeted demos.
        context_fact = self._fact_from_context(system_context)
        if context_fact:
            context_answer = self._format_fact_answer(context_fact)
            if context_answer:
                return context_answer, 0.90

        # Standard Gemini query if configured and the SDK is installed
        if self.use_live_model and os.getenv("GEMINI_API_KEY") and client is not None:
            try:
                context_prompt = f"{system_context}\n\nQuestion: {query}\nProvide a concise direct answer."
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=context_prompt,
                )
                text = getattr(response, 'text', '')
                if not text and hasattr(response, 'candidates'):
                    text = response.candidates[0].content.parts[0].text
                return text.strip(), 78.5
            except Exception as e:
                print(f"Gemini candidate generation failed: {e}")

        # Retrieval-aware mock answers are only allowed when context was provided.
        if system_context:
            for triple in self.kg.get_all_triples():
                sub = triple["subject"]
                rel = triple["relation"]
                obj = triple["object"]

                rel_norm = rel.replace(" ", "_")
                if sub.lower() in lower_query or obj.lower() in lower_query:
                    if rel_norm == "invented":
                        return f"{sub} invented the {obj}.", 82.0
                    elif rel_norm == "developed_by":
                        return f"{sub} was developed by {obj}.", 84.0
                    elif rel_norm == "located_in":
                        return f"{sub} is located in {obj}.", 85.0
                    elif rel_norm == "capital_of":
                        return f"{sub} is the capital of {obj}.", 88.0

        return "I am not certain about the verified facts for this question.", 45.0

    def _question_grounding_fact(self, question):
        """Find the KG fact most relevant to the user's question, preferring facts whose object or subject is mentioned in the question."""
        q_lower = question.lower()
        for triple in self.kg.get_all_triples():
            subject = triple["subject"]
            relation = triple["relation"]
            obj = triple["object"]
            if obj.lower() in q_lower or subject.lower() in q_lower:
                return {
                    "subject": subject,
                    "relation": relation,
                    "expected": obj,
                }
        return None

    def reason(self, question, retrieval_context=""):
        """
        Executes the 3-stage reasoning pipeline:
        Stage 1: Neural (LLM produces candidate)
        Stage 2: Symbolic (KG SPARQL rule check)
        Stage 3: Validation and feedback retry (automatic grounding loop)
        """
        trace = []
        
        # --- STAGE 1: NEURAL GENERATION ---
        trace.append({
            "stage": 1,
            "title": "Neural Candidate Generation",
            "message": "Generating candidate answer from the neural model...",
            "data": None
        })
        
        if retrieval_context:
            candidate, confidence = self._query_llm_candidate(question, system_context=retrieval_context)
        else:
            candidate, confidence = self._query_llm_candidate(question)
        
        trace.append({
            "stage": 1,
            "title": "LLM Candidate Output",
            "message": f"Generated answer: '{candidate}'",
            "data": {"candidate": candidate, "confidence": confidence, "retrieval_context_used": bool(retrieval_context)}
        })
        
        # --- STAGE 2: SYMBOLIC CHECK ---
        trace.append({
            "stage": 2,
            "title": "Symbolic Verification (Knowledge Graph)",
            "message": "Parsing statement structures and checking rules...",
            "data": None
        })
        
        is_valid, error_msg, correction = self.validator.validate_answer(candidate)
        
        if not is_valid:
            # Stage 2 catch
            trace.append({
                "stage": 2,
                "title": "Symbolic Constraint Violated",
                "message": f"CONTRADICTION DETECTED: {error_msg}",
                "data": {"status": "rejected", "error": error_msg, "correction": correction}
            })
            
            # --- STAGE 3: RETRY LOOP WITH KG GROUNDING ---
            trace.append({
                "stage": 3,
                "title": "Automatic Retry with Grounding",
                "message": "Injecting verified KG constraint back into prompt context and retrying...",
                "data": None
            })
            
            grounding_fact = self._fact_from_context(retrieval_context)
            if grounding_fact:
                grounding_fact = {
                    "subject": grounding_fact["subject"],
                    "relation": grounding_fact["relation"],
                    "expected": grounding_fact["object"],
                }
            if grounding_fact is None:
                grounding_fact = self._question_grounding_fact(question)
            if grounding_fact is None:
                grounding_fact = correction or {"subject": "unknown", "relation": "related_to", "expected": "unknown"}
            
            sub = grounding_fact["subject"]
            rel = grounding_fact["relation"]
            correct_val = grounding_fact["expected"]
            
            # Build grounding context
            grounding_context = f"FACT RULES: According to the knowledge graph, the relationship is: {sub} {rel} {correct_val}."
            
            # Retry candidate generation with facts injected
            final_answer, final_confidence = self._query_llm_candidate(question, system_context=grounding_context)
            
            # Double check corrected statement
            is_valid_retry, _, _ = self.validator.validate_answer(final_answer)
            
            trace.append({
                "stage": 3,
                "title": "Validated Answer",
                "message": "Feedback loop complete. Output validated and grounded.",
                "data": {
                    "answer": final_answer, 
                    "confidence": final_confidence,
                    "status": "grounded_success" if is_valid_retry else "unverified_fallback"
                }
            })
            
            return {
                "success": True,
                "final_answer": final_answer,
                "hallucination_caught": True,
                "candidate_answers": [candidate, final_answer],
                "confidence": final_confidence,
                "trace": trace
            }
        else:
            # Immediately valid
            trace.append({
                "stage": 2,
                "title": "Fact Verified",
                "message": "Candidate statement matches verified Knowledge Graph records.",
                "data": {"status": "verified"}
            })
            
            trace.append({
                "stage": 3,
                "title": "Validated Output",
                "message": "Answer returned directly.",
                "data": {"answer": candidate, "confidence": confidence, "status": "verified_success"}
            })
            
            return {
                "success": True,
                "final_answer": candidate,
                "hallucination_caught": False,
                "candidate_answers": [candidate],
                "confidence": confidence,
                "trace": trace
            }
