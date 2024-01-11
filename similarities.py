import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import babelnet as bn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from sentence_transformers import SentenceTransformer
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer


class ArchitecturalElement:
    def __init__(self, element_id, name, desc=" "):
        self.element_id = element_id
        self.name = name
        self.attributes = []
        self.related_elements = []
        self.description = desc

    def add_related_element(self, element_name, relationship_type):
        self.related_elements.append({"element_name": element_name, "relationship_type": relationship_type})

    def add_attribute(self, attribute):
        self.attributes.append(attribute)

    def get_id(self):
        return self.element_id

    def get_name(self):
        return self.name

    def get_atribbutes(self):
        return self.attributes

    def get_relationships(self):
        return self.related_elements

    def get_description(self):
        return self.description


class ArchitecturalGraph:
    def __init__(self, name):
        self.graph = nx.DiGraph(name=name)

    def add_node(self, element):
        self.graph.add_node(element.name, description=element.description)

    def add_edge(self, source_element_name, target_element_name, relationship_type):
        self.graph.add_edge(source_element_name, target_element_name, relationship=relationship_type)

    def create_graph(self, architectural_data, threshold=0.9):
        for element_data in architectural_data:
            element = ArchitecturalElement(
                element_id=element_data["element_id"], name=element_data["name"], desc=element_data["description"]
            )
            self.add_node(element)

            for related_element_data in element_data["related_elements"]:
                self.add_edge(element.name, related_element_data["element_name"], related_element_data["relationship_type"])

            # # Automatic Enrichment based on Semantic Similarity
            # self.automatic_enrichment(element, threshold)

        return self.graph

    # def automatic_enrichment(self, element, threshold): 
    #     for term, similarity_score in self.get_similar_terms(element.name, threshold):
    #         # Check if the similar term is already a node in the graph
    #         if term not in self.graph.nodes:
    #             self.add_node(ArchitecturalElement(element_id=None, name=term, desc=" "))

    #         # Add edge with a weight based on the similarity score
    #         self.add_edge(element.name, term, relationship_type="BabelNetSimilarity", weight=similarity_score)

    # def get_similar_terms(self, term, threshold):
    #     similar_terms = bn.get_synset(term, from_langs=[Language.EN])
    #     return similar_terms

    def draw_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=1500, font_size=8, edge_color='gray')
        plt.show()

def calculate_structural_similarity(graph_1, graph_2):
    structural_similarity = float(nx.is_isomorphic(graph_1, graph_2))
    return structural_similarity

def calculate_textual_similarity(graph_1, graph_2):
    descriptions1 = [graph_1.nodes[node]["description"] for node in graph_1.nodes if "description" in graph_1.nodes[node]]
    descriptions2 = [graph_2.nodes[node]["description"] for node in graph_2.nodes if "description" in graph_2.nodes[node]]
    
    # Use a pre-trained BERT model for embedding sentences
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    # Obtain embeddings for descriptions
    embeddings1 = model.encode(descriptions1, convert_to_tensor=True)
    embeddings2 = model.encode(descriptions2, convert_to_tensor=True)
    # Compute cosine similarity between embeddings
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    avg_similarity = similarity_matrix.mean()
    return avg_similarity

def calculate_contextual_similarity(term1, term2):
    concepts1 = get_concepts(term1)
    concepts2 = get_concepts(term2)

    similarity_score = jaccard_set(concepts1, concepts2)
    return similarity_score

def assess_all_similarities(graphs):
    similarities = {}
    for graph_1, graph_2 in combinations(graphs, 2):
        if graph_1.name not in similarities:
            similarities[graph_1.name] = {}
        similarities[graph_1.name][graph_2.name] = {
            "structural_similarity": calculate_structural_similarity(graph_1, graph_2),
            "textual_similarity": calculate_textual_similarity(graph_1, graph_2),
            "contextual_similarity": calculate_contextual_similarity(graph_1.name, graph_2.name),
        }
    return similarities

# ----- AUX -----
def jaccard_set(list1, list2):
    #Define Jaccard Similarity function for two sets
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union != 0:
        return float(intersection) / union
    else:
        return float(0)
    
def get_concepts(term):
    #List of concepts related to the term
    synsets = []
    for synset in bn.get_synsets(term, from_langs=[Language.EN]):
        lemma_s = synset.main_sense(Language.EN).full_lemma
        synsets.append(lemma_s)
        by = bn.get_synset(synset.id)
        for edge in by.outgoing_edges(BabelPointer.ANY_HYPERNYM):
            by_t = bn.get_synset(edge.id_target)
            lemma_n = by_t.main_sense(Language.EN).full_lemma
            if lemma_n not in synsets: 
                synsets.append(lemma_n)

    return synsets

# Example:
def main():
    # Data for creating the graphs (data objects)
    architectural_data1 = [
        "Patient",
        {
            "element_id": 1,
            "name": "Patient",
            "description": "Individual seeking health services",
            "related_elements": [
                {"element_name": "Doctor", "relationship_type": "Consults"},
                {"element_name": "Medical Record", "relationship_type": "HasRecord"},
            ],
        },
    ]

    architectural_data2 = [
        "Doctor",
        {
            "element_id": 2,
            "name": "Doctor",
            "description": "Healthcare provider",
            "related_elements": [
                {"element_name": "Patient", "relationship_type": "Treats"},
                {"element_name": "Medical Record", "relationship_type": "Consults"},
            ],
        },
    ]

    architectural_data3 = [
        "Medical Record",
        {
            "element_id": 3,
            "name": "Medical Record",
            "description": "Health information storage",
            "related_elements": [
                {"element_name": "Patient", "relationship_type": "Contains"},
            ],
        },
    ]

    # Creation of the graphs
    architectural_graph1 = ArchitecturalGraph(architectural_data1[0])
    graph1 = architectural_graph1.create_graph([architectural_data1[1]])

    architectural_graph2 = ArchitecturalGraph(architectural_data2[0])
    graph2 = architectural_graph2.create_graph([architectural_data2[1]])

    architectural_graph3 = ArchitecturalGraph(architectural_data3[0])
    graph3 = architectural_graph3.create_graph([architectural_data3[1]])

    # Draw the graphs
    architectural_graph1.draw_graph()
    architectural_graph2.draw_graph()
    architectural_graph3.draw_graph()

    all_graphs = [graph1, graph2, graph3]

    all_similarities = assess_all_similarities(all_graphs)

    for graph1, similarities in all_similarities.items():
        for graph2, values in similarities.items():
            print(f"\nSimilarities between {graph1} and {graph2}")
            print(f"Structural similarity: {values['structural_similarity']}")
            print(f"Textual similarity: {values['textual_similarity']}")
            print(f"Contextual similarity: {values['contextual_similarity']}")

if __name__ == "__main__":
    main()
