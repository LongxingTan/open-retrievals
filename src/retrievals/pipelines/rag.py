"""RAG pipeline"""

import argparse
import logging
import re
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

from transformers import AutoModel

from ..tools.file_parser import FileParser
from ..tools.generator import BaseLLM
from ..tools.prompts import RAG_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', required=False)
    return parser.parse_args()


class RAGConfig(object):
    def __init__(self):
        pass

    @classmethod
    def from_dict(
        cls,
    ):
        return RAGConfig()


class SimpleRAG(object):
    def __init__(self, embedder, retriever, generator, reranker=None):
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def generate(self, query: str, context: Optional[str] = None):
        prompt_kwargs = {'context': context, 'query': query}
        response = self.generator(prompt_kwargs)
        return response

    @classmethod
    def from_dict(cls):
        """Create an instance from previously serialized data"""
        obj = cls.__new__(cls)
        return obj


def index_process():
    return


def retrieval_process(query, top_k: int = 3):
    return


def rerank_process():
    return


def chat_process(llm, prompt, history):
    response, history = llm.chat(prompt, history)
    return response, history


def rag_process(prompt, corpus_path, top_k: int = 3):
    global llm, history

    context = retrieval_process(prompt, top_k=top_k)

    context = rerank_process(prompt, context)

    prompt_with_context = RAG_PROMPT.format(question=prompt, context="\n".join(context))

    response, history = chat_process(prompt_with_context, history)
    return


class Session(object):
    def __init__(self, query: str, history: list):
        self.query = query
        self.history = history


def extract_citations(bullet):
    # matches digits or commas
    matches = re.findall(r"\[([\d, ]+)\]", bullet)
    ref_ids = []
    for match in matches:
        ref_ids += [int(m.strip()) for m in match.split(",") if len(m.strip()) > 0]
    return ref_ids


def main():
    args = parse_args()
    print(args)
    return


if __name__ == '__main__':
    main()
