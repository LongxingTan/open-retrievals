"""LLM generator"""

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar


class BaseLLM(ABC):
    """Base class for LLM chat"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate LLM response"""

    @abstractmethod
    async def agenerate(self, prompt: str):
        """Generate LLM response async"""


class BaseLLMCallback:
    """Base class for LLM callback"""

    def __init__(self):
        self.response = []
