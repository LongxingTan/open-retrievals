# Evaluation

**Prerequisites**
```shell
pip install datasets mteb[beir]
pip install open-retrievals[eval]
```


```python
from typing import List, Union, Dict
import numpy as np
from retrievals import AutoModelForEmbedding


class AutoModelForEmbeddingEval(AutoModelForEmbedding):
    def __init__(self, **kwargs):
        super(AutoModelForEmbeddingEval, self).__init__(**kwargs)

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


## Reference

- https://github.com/beir-cellar/beir
- https://github.com/AmenRa/ranx
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
