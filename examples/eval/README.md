# 评测


```python
from datasets import load_dataset

dataset = load_dataset("C-MTEB/T2Reranking", split="dev")
ds = dataset.train_test_split(test_size=0.1, seed=42)

ds_train = (
    ds["train"]
    .filter(lambda x: len(x["positive"]) > 0 and len(x["negative"]) > 0)
)

ds_train.to_json("t2_ranking.jsonl", force_ascii=False)
```



```python
from typing import List, Union, Dict
import numpy as np
from retrievals import AutoModelForEmbedding


class AutoModelForEmbeddingEval(AutoModelForEmbedding):
    def __init__(self, **kwargs):
        super(AutoModelForEmbeddingEval, self).__init__()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """For MTEB eval
        This function will be used for retrieval task
        if there is an instruction for queries, we will add it to the query text
        """
        if self.query_instruction is not None:
            input_texts = ['{}{}'.format(self.query_instruction, q) for q in queries]
        else:
            input_texts = queries
        return self.encode_from_text(input_texts, batch_size=4)

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        """For MTEB eval
        This function will be used for retrieval task
        encode corpus for retrieval task
        """
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode_from_text(input_texts, batch_size=4)
```
