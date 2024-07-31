"""eval for embedding model"""

import argparse
import functools
import random
from typing import Dict, List

import numpy as np
import torch

# from C_MTEB.tasks import *
from mteb import MTEB, DRESModel

from retrievals import AutoModelForEmbedding

TASKS_WITH_PROMPTS = [
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CovidRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "VideoRetrieval",
]

parser = argparse.ArgumentParser(description='evaluation for CMTEB')
parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='which model to use')
parser.add_argument('--output_dir', default='zh_results/', type=str, help='output directory')
parser.add_argument('--max_len', default=512, type=int, help='max length')

args = parser.parse_args()


class RetrievalModel(DRESModel):
    def __init__(self, encoder, query_instruction='', document_instruction='', **kwargs):
        self.encoder = encoder
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """For MTEB eval
        This function will be used for retrieval task
        if there is an instruction for queries, we will add it to the query text
        """
        input_texts = [self.query_instruction + q for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        """For MTEB eval
        This function will be used for retrieval task
        encode corpus for retrieval task
        """
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus

        input_texts = [self.document_instruction + t for t in input_texts]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        return self.encoder.encode(
            sentences=input_texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True
        )


if __name__ == '__main__':
    encoder = AutoModelForEmbedding.from_pretrained(args.model_name)
    encoder.encode = functools.partial(encoder.encode, normalize_embeddings=True)

    task_names = [t.description["name"] for t in MTEB(task_langs=['zh', 'zh-CN']).tasks]
    random.shuffle(task_names)

    for task in task_names:
        evaluation = MTEB(tasks=[task], task_langs=['zh', 'zh-CN'])
        if task in TASKS_WITH_PROMPTS:
            evaluation.run(RetrievalModel(encoder), output_folder=args.output_dir, overwrite_results=False)
        else:
            evaluation.run(encoder, output_folder=args.output_dir, overwrite_results=False)
