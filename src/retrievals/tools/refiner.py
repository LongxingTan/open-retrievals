import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

from .generator import BaseLLM
from .prompts import SUMMARIZE_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_TOKENS = 4000
DEFAULT_MAX_SUMMARY_LENGTH = 500


class BaseRefiner(ABC):
    @abstractmethod
    def refine(self, context):
        """Refine the context"""


class LLMRefiner(BaseRefiner):
    def __init__(
        self,
        llm: BaseLLM,
        prompt: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        max_summary_length: Optional[int] = None,
    ):
        self.llm = llm
        self.prompt = prompt or SUMMARIZE_PROMPT
        self.max_input_tokens = max_input_tokens or DEFAULT_MAX_INPUT_TOKENS
        self.max_summary_length = max_summary_length or DEFAULT_MAX_SUMMARY_LENGTH

    def refine(self, context):
        return self.llm.generate(self.prompt)
