from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from retrievals import AutoModelForEmbedding, RerankModel
from retrievals.tools import LangchainReranker, RagFeature

embed_model = AutoModelForEmbedding(model_name_or_path='')
rerank_model = LangchainReranker(model_name_or_path='', top_n=5, device='cuda')


documents = PyPDFLoader("llama.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

retriever = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
    search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 10}
)

compression_retriever = ContextualCompressionRetriever(base_compressor=rerank_model, base_retriever=retriever)
response = compression_retriever.get_relevant_documents("What is Llama 2?")
