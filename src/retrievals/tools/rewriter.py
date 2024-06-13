from abc import ABC, abstractmethod

import numpy as np


class BaseRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str) -> str:
        """rewrite the query"""
        raise NotImplementedError


class HyDE(BaseRewriter):
    """
    https://github.com/texttron/hyde/blob/main/src/hyde/hyde.py
    """

    def __init__(self, promptor, generator, encoder, searcher):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher

    def rewrite(self, query: str):
        return
