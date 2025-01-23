"""
https://arxiv.org/pdf/2404.16130v1
https://github.com/microsoft/graphrag

Graph, community, entity
"""

import logging

from ..models.retrieval_auto import GraphRetrieval
from ..tools.refiner import LLMRefiner

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    def __init__(self):
        import networkx as nx

        self.graph = nx.Graph()


class GraphRAG:
    def __init__(self, documents, document_processor, embedding_model, knowledge_graph, llm_generator):
        self.documents = documents
        self.document_processor = document_processor
        self.embedding_model = embedding_model
        self.knowledge_graph = knowledge_graph
        self.llm_generator = llm_generator

    def query(self, prompt: str):
        return
