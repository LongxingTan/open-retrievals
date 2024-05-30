import logging
import time
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.autonotebook import trange

logger = logging.getLogger(__name__)


class AutoModelForRetrieval(object):
    def __init__(
        self, embedding_model: Optional[nn.Module] = None, method: Literal['cosine', 'knn'] = "cosine"
    ) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.method = method

    def similarity_search(
        self,
        query_embed: Union[torch.Tensor, np.ndarray],
        document_embed: Optional[torch.Tensor] = None,
        index_path: Optional[str] = None,
        top_k: int = 3,
        batch_size: int = -1,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(query_embed.shape) == 1:
            if isinstance(query_embed, np.ndarray):
                query_embed = query_embed.reshape(1, -1)
            else:
                query_embed = query_embed.view(1, -1)

        self.top_k = top_k
        if document_embed is None and index_path is None:
            logging.warning('Please provide either document_embed for knn/tensor search or index_path for faiss search')
            return

        if index_path is not None:
            import faiss

            start_time = time.time()
            faiss_index = faiss.read_index(index_path)
            logger.info(f'Loading faiss index successfully, elapsed time: {time.time()-start_time:.2}s')

            if batch_size < 1:
                dists, indices = faiss_index.search(query_embed.astype(np.float32), k=top_k)
            else:
                dists, indices = faiss_search(
                    query_embed=query_embed,
                    faiss_index=faiss_index,
                    top_k=top_k,
                    batch_size=batch_size,
                )
            return dists, indices

        elif self.method == "knn":
            neighbors_model = NearestNeighbors(n_neighbors=top_k, metric="cosine", n_jobs=-1)
            neighbors_model.fit(document_embed)
            dists, indices = neighbors_model.kneighbors(query_embed)

        elif self.method == "cosine":
            dists, indices = cosine_similarity_search(
                query_embed, document_embed, top_k=top_k, batch_size=batch_size, convert_to_numpy=convert_to_numpy
            )

        else:
            raise ValueError(
                f"Only 'cosine' and 'knn' method are supported by similarity_search, while get {self.method}"
            )

        return dists, indices

    def search(self):
        return

    def similarity(self, queries: Union[str, List[str]], keys: Union[str, List[str], np.ndarray]):
        return

    def get_relevant_documents(self, query: str):
        return

    def get_pandas_candidate(
        self,
        query_ids: Union[pd.Series, np.ndarray],
        document_ids: Union[pd.Series, np.ndarray],
        dists: np.ndarray,
        indices: np.ndarray,
    ) -> pd.DataFrame:
        if isinstance(query_ids, pd.Series):
            query_ids = query_ids.values
        if isinstance(document_ids, pd.Series):
            document_ids = document_ids.values

        retrieval = {
            'query_id': np.repeat(query_ids, self.top_k),
            'predict_id': document_ids[indices.ravel()],
            'score': dists.ravel(),
        }
        return pd.DataFrame(retrieval)

    def get_rerank_df(
        self,
        input_df: pd.DataFrame,
        query_key: str = 'query_id',
        document_key: str = 'document_id',
        predict_key: str = 'predict_id',
    ) -> pd.DataFrame:
        """
        1: the candidate is in ground truth pool
        2: the candidate is related or not with query
        """
        logger.info('Generate rerank samples based on the retrieval prediction with: query_id, document_ids, pred_ids')

        samples = []
        for _, row in tqdm(input_df.iterrows(), total=len(input_df)):
            query_id = row[query_key]
            document_ids = row[document_key]
            if pd.isna(row[predict_key]):
                print(f' Data error, no {predict_key} in input_df')
                continue
            else:
                predict_ids_list = row[predict_key].split()

            if pd.isna(document_ids):
                print(f' Data Error, the ground truth {document_key} is none')
                for id in predict_ids_list:
                    samples.append({query_key: query_id, document_key: id, 'label': 0})
            else:
                documents = document_ids.split()
                for id in predict_ids_list:
                    if id not in documents:
                        samples.append({query_key: query_id, document_key: id, 'label': 0})
                    else:
                        samples.append({query_key: query_id, document_key: id, 'label': 1})

        return pd.DataFrame(samples)


class EnsembleRetriever(object):
    def __init__(self, retrievers, weights=None):
        pass


def cosine_similarity_search(
    query_embed: torch.Tensor,
    document_embed: torch.Tensor,
    top_k: int = 1,
    batch_size: int = 128,
    penalty: bool = True,
    temperature: float = 0,
    convert_to_numpy: bool = True,
    show_progress_bar: bool = None,
):
    if len(query_embed.size()) == 1:
        query_embed = query_embed.view(1, -1)
    assert query_embed.size()[1] == document_embed.size()[1], (
        f"The embed Shape of query_embed and document_embed should be same, "
        f"while received query {query_embed.size()} and document {document_embed.size()}"
    )
    chunk = batch_size if batch_size > 0 else len(query_embed)
    embeddings_chunks = query_embed.split(chunk)

    dists = []
    indices = []
    for idx in trange(0, len(embeddings_chunks), desc="Batches", disable=not show_progress_bar):
        cos_sim_chunk = torch.matmul(embeddings_chunks[idx], document_embed.transpose(0, 1))
        cos_sim_chunk = torch.nan_to_num(cos_sim_chunk, nan=0.0)
        # if penalty:
        #     pen = ((contents["old_source_count"].values==0) & (contents["old_nonsource_count"].values==1))
        #     cos_sim_chunk = cos_sim_chunk.clamp(0,1) ** torch.Tensor((1 + pen*0.5).reshape(1,-1)).to(DEVICE)
        #     cos_sim_chunk = cos_sim_chunk.clamp(0,1) ** torch.Tensor((1-(contents["old_count"].values==0)*0.1).
        #     reshape(1,-1)).to(DEVICE)
        if temperature:
            cos_sim_chunk = cos_sim_chunk / temperature
        top_k = min(top_k, cos_sim_chunk.size(1))
        dists_chunk, indices_chunk = torch.topk(cos_sim_chunk, k=top_k, dim=1)
        dists.append(dists_chunk[:, :].detach().cpu())
        indices.append(indices_chunk[:, :].detach().cpu())
    dists = torch.cat(dists)
    indices = torch.cat(indices)
    if convert_to_numpy:
        dists = dists.numpy()
        indices = indices.numpy()
    return dists, indices


def faiss_search(
    query_embed: torch.Tensor,
    faiss_index,
    top_k: int = 100,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    # query_embeddings = model.encode_queries(
    #     queries["query"], batch_size=batch_size, max_length=max_length
    # )
    query_size = len(query_embed)
    assert query_size > 0, 'Please make sure the query_embeddings is not empty'

    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embed[i:j]
        score, index = faiss_index.search(query_embedding.astype(np.float32), k=top_k)
        all_scores.append(score)
        all_indices.append(index)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
