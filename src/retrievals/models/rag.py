import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import pytoml
import requests
from transformers import AutoModel

logger = logging.getLogger(__name__)


class RAG(object):
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        index_root: Optional[str] = None,
    ):
        instance = cls()
        instance.model = AutoModel()
        return instance

    @classmethod
    def from_index(cls, index_path: Union[str, Path], n_gpu: int = -1, verbose: int = 1):
        instance = cls()
        index_path = Path(index_path)
        instance.model = AutoModel()

        return instance

    def add_to_index(self):
        return

    def encode(self):
        return

    def index(self):
        return

    def search(self):
        return


class ChatGenerator(object):
    def __init__(self, config_path: str):
        self.config_path = config_path

    def _load_config(self):
        with open(self.config_path, encoding='utf8') as f:
            config = pytoml.load(f)
            return config['llm']
        pass

    def generate(self, prompt, history=None, remote=False):
        pass
