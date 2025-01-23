# 🚀 Open-Retrievals Examples
- [t2-ranking数据集](./t2_ranking/README.md)
- [scifact数据集](./scifact/README.md)
- [wikipedia-nq数据集](./wikipedia-nq/README.md)


```shell
export HF_ENDPOINT=https://hf-mirror.com
```

## 🔍 1. [向量模型](./0_embedding)

**数据格式**

- 文本对：用于批次内负样本微调
```
{'query': TEXT_TYPE, 'positive': List[TEXT_TYPE]}
...
```

- 文本三元组：硬负样本（或混合批次内负样本）微调
```
{'query': TEXT_TYPE, 'positive': List[TEXT_TYPE], 'negative': List[TEXT_TYPE]}
...
```

- 带分数的文本对：
```
{(query, positive, label), (query, negative, label), ...}
```

**微调**

- [向量模型pairwise微调](./0_embedding/train_pairwise.py)
- [decoder大模型向量模型pairwise微调](./0_embedding/train_llm.py)
  - 设置 `query_instruction`
    - "给定一个查询和一个相关文档，检索与查询相关的文档\n查询: {}"
  - 使用适当的 `pooling_method`
    - `last`
  - 由于模型尺寸较大，可能需要减少批次大小
  - 如果想要使用 lora 减少训练内存，设置 `use_lora` 为 `True`


## 📊 2. [重排](./2_reranking)

- [重排序-交叉编码器](./2_reranking/train_cross_encoder.py)
- [重排序-ColBERT](3_colbert/train_colbert.py)
- [重排序-LLM微调](./2_reranking/train_llm.py)
  - `AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm=True)`
  - 提示: "给定一个带有相关正文的查询，通过提供'是'或'否'的预测来确定文档是否与查询相关。"


## 📚 3. [RAG](./4_rag)
- [使用Langchain的RAG](4_rag/rag_langchain_demo.py)



## 4. 常见问题

1. 训练过程中的 grad_norm 始终为零？
- 考虑更改 fp16 或 bf16
- 训练时，在`TrainingArguments`中设置`bf16`或`fp16`；推理时，在`AutoModelForEmbedding`或`LLMRanker`中设置`use_fp16=True`

2. 推理过程中微调后的嵌入性能比原始模型差？
- 检查 pooling_method 是否正确
- 检查LLM大模型使用的提示词或instruction是否与训练时一致

3. 如何微调 `BAAI/bge-m3` ColBERT 模型？
- open-retrievals 支持直接微调 `BAAI/bge-m3 colbert`，只需在微调时不设置 `use_fp16=True`，并将学习率设置得更小
