RAG
=========

.. _rag:

Build an RAG Application
---------------------------

RAG could help solve the false information, out-of-date information, and data security for LLM by searching the external data.
The basic RAG process is document indexing, query embedding, retrieval, optional rerank, and LLM generate.

* Output reference for explainability
* LLM Hallucination


Integrated with Langchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

    from retrievals.tools.langchain import LangchainEmbedding, LangchainReranker, LangchainLLM
    from retrievals import AutoModelForRanking
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.vectorstores import Chroma as Vectorstore
    from langchain.prompts.prompt import PromptTemplate
    from langchain.chains import RetrievalQA

    persist_directory = './database/faiss.index'
    embed_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model_name_or_path = "BAAI/bge-reranker-base"
    llm_model_name_or_path = "microsoft/Phi-3-mini-128k-instruct"

    embeddings = LangchainEmbedding(model_name_or_path=embed_model_name_or_path)
    vectordb = Vectorstore(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    retrieval_args = {"search_type" :"similarity", "score_threshold": 0.15, "k": 10}
    retriever = vectordb.as_retriever(**retrieval_args)

    ranker = AutoModelForRanking.from_pretrained(rerank_model_name_or_path)
    reranker = LangchainReranker(model=ranker, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )

    llm = LangchainLLM(model_name_or_path=llm_model_name_or_path)

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

    PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type='stuff',
        retriever=compression_retriever,
        chain_type_kwargs={
            "verbose": True,
            "prompt": PROMPT,
        }
    )

    user_query = 'Introduce this'
    response = qa_chain({"query": user_query})
    print(response)


Integrated with Llamaindex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Custom RAG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Enhance RAG Performance
---------------------------

* Multi-vector retrieval, sparse + dense
* Rerank
* Long contexts LLM
* Query rewrite, or multi-queries
* Hierarchy retrieval
* Multi-chunks
* Pretrain and finetune of embeddings and rerank weights
* Meta data of documents



pdf parse
--------------

There are some tools help parse the pdf file.

* PyPDF2
    - Good for English
    - Without bbox
* pdfplumber
    - Good for English and Chinese
    - Good for table parse
    - With bbox
* pdfminer
* Camelot
* pymupdf
* papermage
* llama_index parse
    - support table and figure


But if the file is a scanned pdf, we need to use the OCR.

* fitz
    - transfer pdf to image
* https://github.com/mittagessen/kraken
* ppocr


Layout
~~~~~~~~~~~~~~~~~

* https://github.com/LynnHaDo/Document-Layout-Analysis
* Layout-parser
* llama_index parse (support table and figure)
* ppsturcture
* unstructured
