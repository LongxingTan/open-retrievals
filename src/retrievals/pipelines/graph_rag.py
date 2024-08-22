"""
https://arxiv.org/pdf/2404.16130v1
https://github.com/microsoft/graphrag

Graph, community, entity
"""

import logging

from ..models.retrieval_auto import GraphRetrieval
from ..tools.refiner import LLMRefiner

logger = logging.getLogger(__name__)
