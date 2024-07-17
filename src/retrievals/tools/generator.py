"""LLM generator"""

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar


class BaseLLM(ABC):
    """Base class for LLM chat"""

    @abstractmethod
    def generate(self, prompt: str, max_length: int) -> str:
        """Generate LLM response"""

    @abstractmethod
    async def agenerate(self, prompt: str, max_length: int):
        """Generate LLM response async"""


class BaseLLMCallback:
    """Base class for LLM callback"""

    def __init__(self):
        self.response = []
