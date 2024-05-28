import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain.llms.base import LLM
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.embedding_auto import AutoModelForEmbedding
from ..models.rerank import RerankModel

logger = logging.getLogger(__name__)


class LangchainEmbedding(AutoModelForEmbedding, Embeddings):
    """
    Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            model_name_or_path = "BAAI/bge-large-en"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = LangchainEmbeddings(
                model_name_or_path=model_name_or_path,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any
    model_name_or_path: Optional[str] = None
    cache_folder: Optional[str] = None
    model_kwargs: Dict[str, Any] = dict()
    encode_kwargs: Dict[str, Any] = dict()

    def __init__(self, **kwargs: Any):
        Embeddings.__init__(self)
        AutoModelForEmbedding.__init__(self, **kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model."""
        embeddings = self.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model."""
        return self.embed_documents([text])[0]


class LangchainRetrieval(BaseRetriever):
    model: Any
    kwargs: dict = dict()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa
    ) -> List[Document]:
        """Get documents relevant to a query."""
        docs = self.model.search(query, **self.kwargs)
        return [Document(page_content=doc["content"], metadata=doc.get("document_metadata", {})) for doc in docs]


class LangchainReranker(BaseDocumentCompressor):
    model: Any
    kwargs: dict = dict()
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


class LangchainLLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    max_token: int = 10000
    temperature: float = 0.1
    top_p: float = 0.9
    history = []

    def __init__(self, model_name_or_path: str, trust_remote_code: bool = True):
        super(LangchainLLM, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        ).cuda()
        self.model = self.model.eval()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        tensor_inputs = self.tokenizer.batch_encode_plus([prompt], padding='longest', return_tensors='pt')
        output = self.model.generate(
            **tensor_inputs,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        response = self.tokenizer.batch_decode(
            output.cpu()[:, tensor_inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )[0]
        return response

    @property
    def _llm_type(self):
        return "Open-retrievals-llm"
