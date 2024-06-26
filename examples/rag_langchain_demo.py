import json
import os
import re
import tempfile

import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from retrievals.tools.langchain import LangchainEmbedding, LangchainLLM

st.set_page_config(page_title="RAG with Open-retrievals")

with st.sidebar:
    st.write("**RAG with Open-retrievals**")


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = LangchainEmbedding(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(splits, embeddings)
    retrieval_args = {"search_type": "similarity", "score_threshold": 0.15, "k": 30}
    retriever = vectordb.as_retriever(**retrieval_args)
    return retriever


uploaded_files = st.sidebar.file_uploader(label="Upload PDF files", type=["pdf"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()
retriever = configure_retriever(uploaded_files)


llm = LangchainLLM(model_name_or_path="Qwen/Qwen1.5-1.8B-Chat", temperature=0.5, max_tokens=2048, top_k=10)
msgs = StreamlitChatMessageHistory()

RESPONSE_TEMPLATE = """[INST]
<>
You are a helpful AI assistant. Use the following pieces of context to answer the user's question.<>
Anything between the following `context` html blocks is retrieved from a knowledge base.

    {context}

REMEMBER:
- If you don't know the answer, just say that you don't know, don't try to make up an answer.
- Let's take a deep breath and think step-by-step.

Question: {question}[/INST]
Helpful Answer:
"""

PROMPT = PromptTemplate.from_template(RESPONSE_TEMPLATE)
PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=retriever,
    chain_type_kwargs={
        "verbose": True,
        "prompt": PROMPT,
    },
)

if len(msgs.messages) == 0 or st.sidebar.button("New Chat"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        response = qa_chain({"query": user_query})
        answer = response["result"]
        st.write(answer)

about = st.sidebar.expander("About")
about.write("Powered by [open-retrievals](https://github.com/LongxingTan/open-retrievals)")
