import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

from transformers import AutoModel

from ..tools.file_parser import FileParser

logger = logging.getLogger(__name__)


class ChatCenter(object):
    """Model inference"""

    def __init__(self):
        pass

    def chat(self, query: str, chat_history):
        return


class KnowledgeCenter(object):
    """Knowledge parse, store"""

    def __init__(self, knowledge_path, loader, spliter, embedder):
        self.knowledge_path = knowledge_path
        self.loader = loader
        self.splitter = spliter
        self.embedder = embedder
        self.file_parser = FileParser()

    def init_vector_db(self, file_path: str):
        for doc in os.listdir(file_path):
            logger.info(f'Init knowledge center to {self.knowledge_path}, load file: {doc}')
            document = self.loader(doc)
            texts = self.splitter.split_documents(document)
            self.embedder.build_index(texts, path=self.knowledge_path)

    def add_document(self, file_path: str):
        doc = self.loader.read(file_path)
        print(doc)

    def _preprocess(self, files: List[str]):
        pool = Pool(processes=16)

        for idx, file in enumerate(files):
            if file._type in ['pdf', 'word', 'excel', 'ppt', 'html']:
                md5 = self.file_parser.md5(file.origin)
                print(md5)
                pool.apply_async(self._read_and_save, file)
        pass

    def _read_and_save(self, file):
        content, error = self.file_parser.read(file.origin)
        with open(file.copypath, 'w') as f:
            f.write(content)


class Session(object):
    def __init__(self, query: str, history: list):
        self.query = query
        self.history = history


class SimpleRAG(object):
    def __init__(self):
        self.knowledge_center = KnowledgeCenter()
        self.chat_center = ChatCenter

        self.retrieval = None

    def load_knowledge(self):
        pass

    def add_knowledge(self, file_path: Union[str]):
        pass

    def chat(self, question: str):
        pass
