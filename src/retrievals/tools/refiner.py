import json
import logging
from abc import ABC, abstractmethod
from typing import Union

from .generator import BaseLLM

logger = logging.getLogger(__name__)


class BaseRefiner(ABC):
    @abstractmethod
    def refine(self, context):
        """Refine the context"""


class LLMRefiner(BaseRefiner):
    def __init__(self, llm: BaseLLM, prompt: str):
        self.llm = llm
        self.prompt = prompt

    def refine(self, context):
        return self.llm.generate(self.prompt)
