from functools import partial

import faiss
import pandas as pd
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from retrievals import AutoModelForEmbedding, AutoModelForRetrieval


class CFG:
    input_dir = "/root/retrievals/inputs/"
    output_dir = "/root/retrievals/outputs/"
    wikipedia_path = "/root/retrievals/wikipiedia_stem_articles/"

    seed = 42

    model_name_or_path = "BAAI/bge-small-en-v1.5"
    NUM_TITLES = 5

    batch_size = 256
    max_len = 512


def preprocess_corpus(batch, tokenizer, max_len, query_instruction):
    batch["text"] = [query_instruction + x for x in batch["text"]]
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=max_len,
    )
    return tokens.to("cuda")


def build_index():
    dataset = load_from_disk(CFG.wikipedia_path)
    # dataset = Dataset.from_file(CFG.wikipedia_path + "data-00002-of-00004.arrow")

    print(dataset[0])
    print(len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)

    dataset.set_transform(
        partial(
            preprocess_corpus,
            tokenizer=tokenizer,
            max_len=CFG.max_len,
            query_instruction="Represent this sentence for searching relevant passages: ",
        )
    )
    dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False)

    model = AutoModelForEmbedding.from_pretrained(CFG.model_name_or_path, use_fp16=True, max_length=CFG.max_len)

    print("Start to build index")
    model.build_index(
        dataloader,
        index_path=CFG.output_dir + "faiss_index.index",
        use_gpu=True,
        show_progress_bar=True,
    )


def recall():
    df = pd.read_csv(CFG.input_dir + "test.csv", index_col="id")
    inputs = df.apply(
        lambda row: " ".join([row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]]),
        axis=1,
    ).values

    model = AutoModelForEmbedding.from_pretrained(CFG.model_name_or_path, use_fp16=True, max_length=CFG.max_len)
    prompt_embeddings = model.encode(inputs, show_progress_bar=False)

    matcher = AutoModelForRetrieval()
    dists, indices = matcher.similarity_search(
        prompt_embeddings,
        index_path=CFG.output_dir + "faiss_index.index",
        batch_size=512,
        top_k=20,
    )
    print(indices)

    dataset = load_from_disk(CFG.wikipedia_path)
    for i in range(len(df)):
        df.loc[i, "context"] = "-" + "\n-".join([dataset[int(j)]["text"] for j in indices[i]])

    print(df)
    print(df.columns)

    df.to_csv(CFG.output_dir + "context_df.csv", index=False)
