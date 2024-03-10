from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CBEventType, EventPayload
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.utils import infer_torch_device

from src.retrievals import RerankModel


class LlamaIndexReranker(BaseNodePostprocessor):
    model: str = Field(ddescription="Sentence transformer model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    _model: Any = PrivateAttr()

    def __init__(self, model: str, top_n: int = 5, device: Optional[str] = None, **kwargs):
        self.model = RerankModel()
        device = infer_torch_device() if device is None else device
        super().__init__(model=model, top_n=top_n, device=device)

    @classmethod
    def class_name(cls):
        return 'RetrievalsRank'

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ):
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query = query_bundle.query_str
        passages = []
        valid_nodes = []
        invalid_nodes = []
        for node in nodes:
            passage = node.node.get_content(metadata_mode=MetadataMode.EMBED)
            if isinstance(passage, str) and len(passage) > 0:
                passages.append(passage.replace('\n', ' '))
                valid_nodes.append(node)
            else:
                invalid_nodes.append(node)

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            rerank_result = self._model.rerank(query, passages)
            new_nodes = []
            for score, nid in zip(rerank_result['rerank_scores'], rerank_result['rerank_ids']):
                node = valid_nodes[nid]
                node.score = score
                new_nodes.append(node)
            for node in invalid_nodes:
                node.score = 0
                new_nodes.append(node)

            assert len(new_nodes) == len(nodes)

            new_nodes = new_nodes[: self.top_n]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
