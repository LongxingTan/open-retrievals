from unittest import TestCase

import faiss
import numpy as np
import torch

from src.retrievals.models.retrieval_auto import AutoRetrieval


class AutoRetrievalTest(TestCase):
    def test_match(self):
        pass

    def test_similarity_search_cosine(self):
        num_queries = 20
        num_k = 10

        doc_emb = torch.tensor(np.random.randn(1000, 100))
        q_emb = torch.tensor(np.random.randn(num_queries, 100))
        matcher = AutoRetrieval(method="cosine")
        dists, indices = matcher.similarity_search(
            q_emb, doc_emb, top_k=num_k, query_chunk_size=5, corpus_chunk_size=17
        )

        assert len(dists) == num_queries
        assert len(dists[0]) == num_k

        # # Sanity Check of the results
        # cos_scores = matcher.similarity_search(q_emb, doc_emb)
        # cos_scores_values, cos_scores_idx = cos_scores.topk(num_k)
        # cos_scores_values = cos_scores_values.cpu().tolist()
        # cos_scores_idx = cos_scores_idx.cpu().tolist()
        #
        # for qid in range(num_queries):
        #     for hit_num in range(num_k):
        #         assert dists[qid][hit_num]["corpus_id"] == cos_scores_idx[qid][hit_num]
        #         assert np.abs(dists[qid][hit_num]["score"] - cos_scores_values[qid][hit_num]) < 0.001
