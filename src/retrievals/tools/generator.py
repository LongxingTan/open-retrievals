"""LLM generator"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Protocol, TypeVar


class BaseLLM(ABC):
    """Base class for LLM chat"""

    @abstractmethod
    def generate(self, prompt: str, max_length: int) -> str:
        """Generate LLM response"""
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, max_length: int):
        """Generate LLM response async"""
        pass


class BaseLLMCallback:
    """Base class for LLM callback"""

    def __init__(self):
        self.response: List[str] = []

    def __str__(self):
        """String representation for easy inspection of stored responses"""
        return f"BaseLLMCallback(responses={self.response})"


class OpenAILLM(BaseLLM):
    """Concrete implementation of BaseLLM using OpenAI API"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        import openai

        openai.api_key = api_key
        self.model = model

    def generate(self, prompt: str, max_length: int) -> str:
        """Generate LLM response using OpenAI API (synchronous)"""
        import openai

        try:
            response = openai.Completion.create(model=self.model, prompt=prompt, max_tokens=max_length)
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""

    async def agenerate(self, prompt: str, max_length: int) -> str:
        """Generate LLM response using OpenAI API (asynchronous)"""
        # OpenAI API does not provide direct async support, but we can use `openai` in an async manner.
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, max_length)


class HfLocalLLM(BaseLLM):
    """Concrete implementation of BaseLLM using a local LLM"""

    def __init__(self, model_name_or_path: str, device: Optional[str] = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.device = torch.device(device)
        self.model.to(self.device)

    def generate(self, prompt: str, max_length: int) -> str:
        """Generate LLM response using a local model (synchronous)"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length + len(inputs['input_ids'][0]),  # Add input length
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

    async def agenerate(self, prompt: str, max_length: int) -> str:
        """Generate LLM response using a local model (asynchronous)"""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, max_length)
