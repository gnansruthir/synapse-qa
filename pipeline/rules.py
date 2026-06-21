import re
from rdflib import Literal

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

    def validate_answer(self, candidate_answer_text):
        """
        Validates the LLM's candidate answer text against the KG using SPARQL.
        """
        triple = self.parse_statement(candidate_answer_text)
        if not triple:
            return True, None, None
            
        sub = triple["subject"]
        rel = triple["relation"]
        obj = triple["object"]
        
        # 1. Ask the KG to verify this triple
        is_verified, correct_objs = self.kg.verify_triple(sub, rel, obj)
        
        if not is_verified:
            # 2. Run SPARQL query to find correct subject for this relationship and target object
            sparql_query = """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?subName WHERE {
                ?subNode ?pred ?objNode .
                ?subNode foaf:name ?subName .
            }
            """
            
            obj_node = self.kg.find_node_by_name(obj)
            pred_uri = self.kg.EX[self.kg._uri_friendly(rel)]
            
            correct_subjects = []
            if obj_node:
                qres = self.kg.graph.query(
                    sparql_query, 
                    initBindings={'pred': pred_uri, 'objNode': obj_node}
                )
                for row in qres:
                    correct_subjects.append(str(row.subName))
                    
            if correct_subjects:
                correct_sub = correct_subjects[0]
                error_msg = f"Logical contradiction caught in statement: '{sub} {rel} {obj}' violates constraints. SPARQL lookup states: '{correct_sub} {rel} {obj}'."
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
