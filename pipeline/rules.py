import re
from rdflib import Literal

class SymbolicValidator:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        
        # Mapping common patterns to KG predicates
        self.intent_rules = [
            (r"(?i)invented\s+([a-zA-Z\s]+)", "invented"),
            (r"(?i)invented\s+by\s+([a-zA-Z\s]+)", "invented"),
            (r"(?i)developed\s+by\s+([a-zA-Z\s\+]+)", "developed_by"),
            (r"(?i)developed\s+([a-zA-Z\s\+]+)", "developed_by"),
            (r"(?i)located\s+in\s+([a-zA-Z\s]+)", "located_in"),
            (r"(?i)capital\s+of\s+([a-zA-Z\s]+)", "capital_of"),
            (r"(?i)discovered\s+([a-zA-Z\s]+)", "discovered"),
            (r"(?i)formulated\s+([a-zA-Z\s]+)", "formulated")
        ]

    def parse_statement(self, statement_text):
        """
        Attempts to extract a Subject, Relation, and Object triple from the text
        using pre-defined syntactic intent rules.
        """
        clean_text = re.sub(r'[^\w\s]', '', statement_text).strip()
        
        # Heuristics for common entity mappings in our database
        subjects = [
            "Alexander Graham Bell", "Thomas Edison", "Nikola Tesla", 
            "Albert Einstein", "Isaac Newton", "Marie Curie", "Wilhelm Röntgen",
            "Paris", "London", "Washington D.C.", "New Delhi", "Tokyo",
            "Eiffel Tower", "Big Ben", "Taj Mahal", "Agra", "Mount Fuji",
            "Python", "C Programming", "Java", "World Wide Web", "Dennis Ritchie",
            "Guido van Rossum", "James Gosling", "Tim Berners-Lee"
        ]
        
        found_subjects = []
        for s in subjects:
            if s.lower() in clean_text.lower():
                found_subjects.append(s)
                
        found_subjects.sort(key=len, reverse=True)
        
        if not found_subjects:
            return None
            
        subject = found_subjects[0]
        
        for pattern, relation in self.intent_rules:
            match = re.search(pattern, clean_text)
            if match:
                extracted_obj = match.group(1).strip()
                target_objects = [
                    "Telephone", "Lightbulb", "Phonograph", "Induction Motor",
                    "General Relativity", "Special Relativity", "Law of Universal Gravitation",
                    "Radium", "Polonium", "X-Rays",
                    "France", "United Kingdom", "United States", "India", "Japan",
                    "Paris", "London", "Agra"
                ]
                
                for obj in target_objects:
                    if obj.lower() in extracted_obj.lower() or extracted_obj.lower() in obj.lower():
                        return {
                            "subject": subject,
                            "relation": relation,
                            "object": obj
                        }
                        
        return None

    def _find_expected_object_for_subject(self, subject, relation):
        """Find the canonical object for a given subject/relation pair in the KG."""
        relation_key = relation.replace("_", " ").lower()
        for triple in self.kg.get_all_triples():
            if triple["subject"].lower() == subject.lower() and triple["relation"].replace("_", " ").lower() == relation_key:
                return triple["object"]
        return None

    def _find_subject_for_object(self, relation, obj):
        """Find the subject whose KG fact matches the given relation and object."""
        relation_key = relation.replace("_", " ").lower()
        target_obj = obj.lower()
        for triple in self.kg.get_all_triples():
            if triple["relation"].replace("_", " ").lower() == relation_key and triple["object"].lower() == target_obj:
                return triple["subject"]
        return None

    def validate_answer(self, candidate_answer_text):
        """
        Validates the LLM's candidate answer text against the KG using SPARQL.
        """
        triple = self.parse_statement(candidate_answer_text)
        if not triple:
            return False, "Candidate does not contain a parseable fact statement.", None
            
        sub = triple["subject"]
        rel = triple["relation"]
        obj = triple["object"]
        
        # 1. Ask the KG to verify this triple
        is_verified, correct_objs = self.kg.verify_triple(sub, rel, obj)
        
        if not is_verified:
            # Prefer the fact that actually matches the target object for this relation.
            correct_sub = self._find_subject_for_object(rel, obj)
            if correct_sub:
                error_msg = f"Logical contradiction caught in statement: '{sub} {rel} {obj}' violates constraints. SPARQL lookup states: '{correct_sub} {rel} {obj}'."
                return False, error_msg, {
                    "subject": correct_sub,
                    "relation": rel,
                    "expected": obj
                }

            subject_expected_obj = self._find_expected_object_for_subject(sub, rel)
            if subject_expected_obj:
                error_msg = f"Logical contradiction caught in statement: '{sub} {rel} {obj}' violates constraints. The KG states: '{sub} {rel} {subject_expected_obj}'."
                return False, error_msg, {
                    "subject": sub,
                    "relation": rel,
                    "expected": subject_expected_obj
                }

            # Fallback to standard correct objects for the subject
            correct_ans_str = ", ".join(correct_objs) if correct_objs else "unknown"
            error_msg = f"Logical contradiction caught in statement: '{sub} {rel} {obj}' contradicts KG rule. Expected: '{correct_ans_str}'"
            return False, error_msg, {
                "subject": sub,
                "relation": rel,
                "expected": correct_objs[0] if correct_objs else None
            }
            
        return True, None, None
