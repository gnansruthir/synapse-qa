import re

class SymbolicValidator:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        
        # Mapping common patterns to KG predicates
        self.intent_rules = [
            (r"(?i)invented\s+([a-zA-Z\s]+)", "invented"),
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
        # Let's clean input text
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
                
        # Sorted by length descending to match full names before single names
        found_subjects.sort(key=len, reverse=True)
        
        if not found_subjects:
            return None
            
        subject = found_subjects[0]
        
        # Match predicates and extract potential objects
        for pattern, relation in self.intent_rules:
            match = re.search(pattern, clean_text)
            if match:
                extracted_obj = match.group(1).strip()
                # Find matching target object in database
                target_objects = [
                    "Telephone", "Lightbulb", "Phonograph", "Induction Motor",
                    "General Relativity", "Special Relativity", "Law of Universal Gravitation",
                    "Radium", "Polonium", "X-Rays",
                    "France", "United Kingdom", "United States", "India", "Japan",
                    "Paris", "London", "Agra"
                ]
                
                # Check case insensitive
                for obj in target_objects:
                    if obj.lower() in extracted_obj.lower() or extracted_obj.lower() in obj.lower():
                        return {
                            "subject": subject,
                            "relation": relation,
                            "object": obj
                        }
                        
        return None

    def validate_answer(self, candidate_answer_text):
        """
        Validates the LLM's candidate answer text against the KG.
        Returns:
            is_valid: bool
            error_message: str (or None)
            correct_info: dict (or None)
        """
        triple = self.parse_statement(candidate_answer_text)
        if not triple:
            # If we cannot parse a structured fact, we consider it un-contradicted
            return True, None, None
            
        sub = triple["subject"]
        rel = triple["relation"]
        obj = triple["object"]
        
        is_verified, correct_objs = self.kg.verify_triple(sub, rel, obj)
        
        if not is_verified:
            # Let's search if there is a correct subject for this relation & object pair in the KG
            correct_subjects = []
            for u, v, data in self.kg.graph.edges(data=True):
                if data.get("relation") == rel and v.lower() == obj.lower():
                    correct_subjects.append(u)
                    
            if correct_subjects:
                correct_sub = correct_subjects[0]
                error_msg = f"Logical contradiction caught in statement: '{sub} {rel} {obj}' contradicts KG rule. Fact check states: '{correct_sub} {rel} {obj}'."
                return False, error_msg, {
                    "subject": correct_sub,
                    "relation": rel,
                    "expected": obj
                }
            else:
                # Fallback to standard correct objects for the subject
                correct_ans_str = ", ".join(correct_objs) if correct_objs else "unknown"
                error_msg = f"Logical contradiction caught in statement: '{sub} {rel} {obj}' contradicts KG rule. Expected: '{correct_ans_str}'"
                return False, error_msg, {
                    "subject": sub,
                    "relation": rel,
                    "expected": correct_objs[0] if correct_objs else None
                }
            
        return True, None, None
