import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        # Initialize a MultiDiGraph to support multiple relationships between entities
        self.graph = nx.MultiDiGraph()
        self._load_wikidata_subset()

    def _load_wikidata_subset(self):
        """Loads a subset of verified facts representing Wikidata relations."""
        # Triples format: (Subject, Predicate, Object)
        facts = [
            ("Alexander Graham Bell", "invented", "Telephone"),
            ("Thomas Edison", "invented", "Lightbulb"),
            ("Thomas Edison", "invented", "Phonograph"),
            ("Nikola Tesla", "invented", "Induction Motor"),
            ("Albert Einstein", "formulated", "General Relativity"),
            ("Albert Einstein", "formulated", "Special Relativity"),
            ("Isaac Newton", "formulated", "Law of Universal Gravitation"),
            ("Marie Curie", "discovered", "Radium"),
            ("Marie Curie", "discovered", "Polonium"),
            ("Wilhelm Röntgen", "discovered", "X-Rays"),
            
            ("Paris", "capital_of", "France"),
            ("London", "capital_of", "United Kingdom"),
            ("Washington D.C.", "capital_of", "United States"),
            ("New Delhi", "capital_of", "India"),
            ("Tokyo", "capital_of", "Japan"),
            
            ("Eiffel Tower", "located_in", "Paris"),
            ("Big Ben", "located_in", "London"),
            ("Taj Mahal", "located_in", "Agra"),
            ("Agra", "located_in", "India"),
            ("Mount Fuji", "located_in", "Japan"),
            
            ("Python", "developed_by", "Guido van Rossum"),
            ("C Programming", "developed_by", "Dennis Ritchie"),
            ("Java", "developed_by", "James Gosling"),
            ("World Wide Web", "developed_by", "Tim Berners-Lee")
        ]
        
        for subject, predicate, obj in facts:
            self.graph.add_edge(subject, obj, relation=predicate)

    def lookup_relation(self, subject, relation):
        """Looks up the target object given a subject and relation predicate."""
        if not self.graph.has_node(subject):
            # Try a case-insensitive lookup
            matching_nodes = [n for n in self.graph.nodes if n.lower() == subject.lower()]
            if not matching_nodes:
                return []
            subject = matching_nodes[0]
            
        results = []
        for u, v, key, data in self.graph.out_edges(subject, keys=True, data=True):
            if data.get("relation") == relation:
                results.append(v)
        return results

    def verify_triple(self, subject, relation, obj):
        """
        Verifies if a triple (Subject, Relation, Object) exists in the KG.
        Returns (is_verified, correct_objects)
        """
        # Case-insensitive subject match
        matching_subjects = [n for n in self.graph.nodes if n.lower() == subject.lower()]
        if not matching_subjects:
            return False, []
        
        sub = matching_subjects[0]
        correct_objects = []
        is_verified = False
        
        # Check edges
        for u, v, key, data in self.graph.out_edges(sub, keys=True, data=True):
            if data.get("relation") == relation:
                correct_objects.append(v)
                if v.lower() == obj.lower():
                    is_verified = True
                    
        return is_verified, correct_objects

    def get_all_triples(self):
        """Returns list of all triples in the graph."""
        triples = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            triples.append({
                "subject": u,
                "relation": data.get("relation"),
                "object": v
            })
        return triples
