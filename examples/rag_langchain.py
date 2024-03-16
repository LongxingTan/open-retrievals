import torch
import transformers
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from retrievals.tools.langchain import LangchainEmbedding, LangchainReranker, RagFeature


class CFG:
    retrieval_model = 'BAAI/bge-large-zh'
    rerank_model = ''
    llm_model = 'Qwen/Qwen-7B-Chat'


embed_model = LangchainEmbedding(model_name_or_path=CFG.retrieval_model)
rerank_model = LangchainReranker(model_name_or_path=CFG.rerank_model, top_n=5, device='cuda')


documents = PyPDFLoader("llama.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

retriever = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
    search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 10}
)

# compression_retriever = ContextualCompressionRetriever(base_compressor=rerank_model, base_retriever=retriever)
# response = compression_retriever.get_relevant_documents("What is Llama 2?")


tokenizer = AutoTokenizer.from_pretrained(CFG.llm_model, trust_remote_code=True)
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
model = AutoModelForCausalLM.from_pretrained(
    CFG.llm_model, device_map='auto', load_in_4bit=True, max_memory=max_memory, trust_remote_code=True, fp16=True
)
model = model.eval()
model.generation_config = GenerationConfig.from_pretrained(CFG.llm_model, trust_remote_code=True)

query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

llm = HuggingFacePipeline(pipeline=query_pipeline)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)

qa.run('你看了这篇文章后有何感性?')
