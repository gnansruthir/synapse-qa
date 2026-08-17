import re
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, FOAF

class KnowledgeGraph:
    def __init__(self):
        self.graph = Graph()
        # Custom Namespace for demo RDF fact base
        self.EX = Namespace("http://example.org/demo-facts/")
        self.graph.bind("ex", self.EX)
        self._load_demo_facts()

    def _uri_friendly(self, val):
        """Converts strings to URI-compatible camelcase tokens."""
        # Remove special characters and spaces
        val_clean = re.sub(r'[^\w\s\.-]', '', val).strip()
        # Camelcase it
        words = val_clean.split()
        if not words:
            return "Unknown"
        return "_".join(words)

    def _load_demo_facts(self):
        """Loads demo fact triples into an RDF Graph for testing and evaluation."""
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
            sub_uri = self.EX[self._uri_friendly(subject)]
            pred_uri = self.EX[self._uri_friendly(predicate)]
            obj_uri = self.EX[self._uri_friendly(obj)]
            
            # Store primary URI nodes
            self.graph.add((sub_uri, pred_uri, obj_uri))
            # Store string literal representations for mapping
            self.graph.add((sub_uri, FOAF.name, Literal(subject)))
            self.graph.add((obj_uri, FOAF.name, Literal(obj)))

    def find_node_by_name(self, name):
        """Finds matching RDF URI node by checking literal FOAF names case-insensitively."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?node WHERE {
            ?node foaf:name ?name .
            FILTER(LCASE(STR(?name)) = LCASE(?target_name))
        }
        LIMIT 1
        """
        qres = self.graph.query(query, initBindings={'target_name': Literal(name)})
        for row in qres:
            return row.node
        return None

    def get_node_label(self, node_uri):
        """Retrieves literal name representation of a node."""
        name = self.graph.value(subject=node_uri, predicate=FOAF.name)
        if name:
            return str(name)
        # Fallback to URI leaf
        return str(node_uri).split("/")[-1].replace("_", " ")

    def verify_triple(self, subject, relation, obj):
        """
        Verifies if a triple exists using a SPARQL ASK query.
        Returns (is_verified, list_of_correct_objects)
        """
        sub_node = self.find_node_by_name(subject)
        obj_node = self.find_node_by_name(obj)
        pred_uri = self.EX[self._uri_friendly(relation)]
        
        if not sub_node:
            return False, []

        # 1. Ask SPARQL query to verify fact existence
        ask_query = """
        ASK {
            ?sub ?pred ?obj .
        }
        """
        is_verified = bool(self.graph.query(
            ask_query, 
            initBindings={'sub': sub_node, 'pred': pred_uri, 'obj': obj_node or URIRef("http://none")}
        ))
        
        # 2. Query SPARQL to find correct answers/alternatives for this relationship
        select_query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?objName WHERE {
            ?sub ?pred ?objNode .
            ?objNode foaf:name ?objName .
        }
        """
        correct_objects = []
        qres = self.graph.query(select_query, initBindings={'sub': sub_node, 'pred': pred_uri})
        for row in qres:
            correct_objects.append(str(row.objName))
            
        return is_verified, correct_objects

    def get_all_triples(self):
        """Returns list of all triples in the graph via SPARQL SELECT."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?subName ?pred ?objName WHERE {
            ?subNode ?pred ?objNode .
            ?subNode foaf:name ?subName .
            ?objNode foaf:name ?objName .
        }
        """
        qres = self.graph.query(query)
        triples = []
        for row in qres:
            # Extract relation label from URI
            relation = str(row.pred).split("/")[-1].replace("_", " ")
            triples.append({
                "subject": str(row.subName),
                "relation": relation,
                "object": str(row.objName)
            })
        return triples
