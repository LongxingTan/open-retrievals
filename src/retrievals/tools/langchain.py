from typing import Any, Dict, Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra, root_validator
from pydantic.v1 import PrivateAttr

from ..models.embedding_auto import AutoModelForEmbedding
from ..models.rerank import RerankModel


class LangchainEmbedding(AutoModelForEmbedding, Embeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LangchainReranker(BaseDocumentCompressor):
    model: Any
    kwargs: dict = {}
    k: int = 5

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Optional[Callbacks] = None, **kwargs
    ) -> Sequence[Document]:
        """
        Compress documents using `BCEmbedding RerankerModel API`.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:
            return []
        doc_list = list(documents)

        passages = []
        valid_doc_list = []
        invalid_doc_list = []
        for d in doc_list:
            passage = d.page_content
            if isinstance(passage, str) and len(passage) > 0:
                passages.append(passage.replace('\n', ' '))
                valid_doc_list.append(d)
            else:
                invalid_doc_list.append(d)

        rerank_result = self.model.rerank(query, passages)
        final_results = []
        for score, doc_id in zip(rerank_result['rerank_scores'], rerank_result['rerank_ids']):
            doc = valid_doc_list[doc_id]
            doc.metadata["relevance_score"] = score
            final_results.append(doc)
        for doc in invalid_doc_list:
            doc.metadata["relevance_score"] = 0
            final_results.append(doc)

        final_results = final_results[: self.top_n]
        return final_results
