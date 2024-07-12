"""RAG pipeline"""

import argparse
import logging
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

from transformers import AutoModel

from ..tools.file_parser import FileParser
from ..tools.generator import BaseLLM
from ..tools.prompts import Prompt

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

llm = BaseLLM()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', required=False)
    return parser.parse_args()


class RAGConfig(object):
    def __init__(self):
        pass


def rag_process(prompt, history, top_k: int = 3):
    context = retrieval_process(prompt, top_k=top_k)

    prompt_with_context = Prompt.RAG_PROMPT.format(question=prompt, context="\n".join(context))

    response, history = chat_process(prompt_with_context, history)
    return


def retrieval_process(query, top_k: int = 3):
    return


def chat_process(prompt, history):
    response, history = llm.chat(prompt, history)
    return response, history


class Session(object):
    def __init__(self, query: str, history: list):
        self.query = query
        self.history = history


def main():
    args = parse_args()
    print(args)
    return


if __name__ == '__main__':
    main()
