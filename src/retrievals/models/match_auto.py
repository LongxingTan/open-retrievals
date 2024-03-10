import logging
from typing import Union

import faiss
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AutoModelForMatch(nn.Module):
    def __init__(self, method="knn") -> None:
        super().__init__()
        self.method = method

    def similarity_search(
        self, query_embed, passage_embed=None, top_k=1, query_chunk_size=5, corpus_chunk_size=17, **kwargs
    ):
        """ """
        if self.method == "knn":
            neighbors_model = NearestNeighbors(n_neighbors=top_k, metric="cosine", n_jobs=-1)
            neighbors_model.fit(passage_embed)
            dists, indices = neighbors_model.kneighbors(query_embed)
            return dists, indices

        elif self.method == "cosine":
            dists, indices = cosine_similarity_search(query_embed, passage_embed, top_k=top_k)
            return dists, indices

        else:
            raise ValueError

    def faiss_search(
        self,
        query_embed,
        index_path: str = "/faiss.index",
        top_k=1,
        batch_size: int = 256,
        max_length: int = 512,
    ):
        faiss_index = faiss.read_index(index_path)
        dists, indices = faiss_search(
            query_embeddings=query_embed,
            faiss_index=faiss_index,
            top_k=top_k,
            batch_size=batch_size,
        )
        return dists, indices

    def get_rerank_df(self):
        pass


def cosine_similarity_search(
    query_embed,
    passage_embed,
    top_k: int = 1,
    batch_size: int = 512,
    penalty: bool = True,
    temperature: float = 0,
):
    chunk = batch_size
    embeddings_chunks = query_embed.split(chunk)

    vals = []
    inds = []
    for idx in range(len(embeddings_chunks)):
        cos_sim_chunk = torch.mm(embeddings_chunks[idx], passage_embed.transpose(0, 1))
        cos_sim_chunk = torch.nan_to_num(cos_sim_chunk, nan=0.0)
        # if penalty:
        #     pen = ((contents["old_source_count"].values==0) & (contents["old_nonsource_count"].values==1))
        #     cos_sim_chunk = cos_sim_chunk.clamp(0,1) ** torch.Tensor((1 + pen*0.5).reshape(1,-1)).to(DEVICE)
        #     cos_sim_chunk = cos_sim_chunk.clamp(0,1) ** torch.Tensor((1-(contents["old_count"].values==0)*0.1).
        #     reshape(1,-1)).to(DEVICE)
        if temperature:
            cos_sim_chunk = cos_sim_chunk / temperature
        top_k = min(top_k, cos_sim_chunk.size(1))
        vals_chunk, inds_chunk = torch.topk(cos_sim_chunk, k=top_k, dim=1)
        vals.append(vals_chunk[:, :].detach().cpu())
        inds.append(inds_chunk[:, :].detach().cpu())
    vals = torch.cat(vals).detach().cpu().numpy()
    inds = torch.cat(inds).detach().cpu().numpy()
    return inds, vals


def faiss_search(
    query_embeddings,
    faiss_index: faiss.Index,
    top_k: int = 100,
    batch_size: int = 256,
    max_length: int = 512,
):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    # query_embeddings = model.encode_queries(
    #     queries["query"], batch_size=batch_size, max_length=max_length
    # )
    query_size = len(query_embeddings)

    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i:j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=top_k)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices


class FaissIndex:
    def __init__(self, device) -> None:
        if isinstance(device, torch.device):
            if device.index is None:
                device = "cpu"
            else:
                device = device.index
        self.device = device

    def build(self, encoded_corpus, index_factory, metric):
        if metric == "l2":
            metric = faiss.METRIC_L2
        elif metric in ["ip", "cos"]:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise NotImplementedError(f"Metric {metric} not implemented!")

        index = faiss.index_factory(encoded_corpus.shape[1], index_factory, metric)

        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            # logger.info("using fp16 on GPU...")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)

        logging.info("training index...")
        index.train(encoded_corpus)
        logging.info("adding embeddings...")
        index.add(encoded_corpus)
        self.index = index
        return index

    def load(self, index_path):
        logging.info(f"loading index from {index_path}...")
        index = faiss.read_index(index_path)
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)
        self.index = index
        return index

    def save(self, index_path):
        logging.info(f"saving index at {index_path}...")
        if isinstance(self.index, faiss.GpuIndex):
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, index_path)

    def search(self, query, hits):
        return self.index.search(query, k=hits)
