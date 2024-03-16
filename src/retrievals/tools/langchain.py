from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra, root_validator

from src.retrievals.models.embedding_auto import AutoModelForEmbedding
from src.retrievals.models.rerank import RerankModel


class LangchainEmbedding(AutoModelForEmbedding, Embeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LangchainReranker(BaseDocumentCompressor):
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __init__(
        self,
        model_name_or_path: str,
        top_n: int = 3,
        device: str = "cuda",
        max_length: int = 1024,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
    ):
        self._model = RerankModel(model_name_or_path=model_name_or_path, max_length=max_length, device=device)
        super().__init__(
            model=model_name_or_path,
            top_n=top_n,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
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
        if len(documents) == 0:  # to avoid empty api call
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

        rerank_result = self._model.rerank(query, passages)
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


class RagFeature(object):
    def __init__(self, config_path: str = 'config.ini'):
        pass
