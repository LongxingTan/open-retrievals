import logging
from abc import ABC, abstractmethod

from .generator import BaseLLM

logger = logging.getLogger(__name__)


class BaseRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str) -> str:
        """Rewrite the query"""


class HyDE(BaseRewriter):
    """
    https://github.com/texttron/hyde/blob/main/src/hyde/hyde.py
    """

    def __init__(self, llm: BaseLLM, prompt: str):
        self.llm = llm
        self.prompt = prompt

    def rewrite(self, query: str):
        return self.llm.generate(self.prompt)
