import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

from transformers import AutoModel

from ..tools.file_parser import FileParser

logger = logging.getLogger(__name__)


class ModelCenter(object):
    """Model inference"""

    def __init__(self):
        pass

    def chat(self, query: str, chat_history):
        return


class KnowledgeCenter(object):
    """Knowledge parse, store"""

    def __init__(self):
        self.parser = FileParser()

    def init_vector_db(self):
        pass

    def add_document(self, file_path: str):
        doc = self.parser.read(file_path)
        print(doc)


class Session(object):
    def __init__(self, query: str, history: list):
        self.query = query
        self.history = history


class SimpleRAG(object):
    def __init__(self):
        pass

    def load_knowledge(self):
        pass

    def add_knowledge(self, file_path: Union[str]):
        pass

    def chat(self, question: str):
        pass
