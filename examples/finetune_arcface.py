import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    set_seed,
)
from utils_data import (
    metric,
    prepare_lecr_data,
    prepare_lecr_data_pointwise2,
    seed_everything,
)

from retrievals import AutoModelForEmbedding, AutoModelForMatch, CustomTrainer
from retrievals.losses import ArcFaceAdaptiveMarginLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFG:
    exp_id = "retrieval_arcface_001"
    env = "autodl"
    input_dir = "/root/autodl-tmp/learning-equality-curriculum-recommendations/"
    output_dir = "/root/autodl-tmp/lecr/"
    top_n = 50  # 1000
    num_workers = 8
    MODEL_NAME = "bert-base-multilingual-uncased"
    # MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    text_column = "description"
    label_column = "label"

    max_len = 64
    batch_size = 64
    seed = 0
    n_folds = 3
    train_fold = [0]
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    apex = True

    epochs = 20
    learning_rate = 0.0004
    weight_decay = 0.0
    warmup_epochs = 9
    differential_learning_rate_layers = "model"
    differential_learning_rate = 4.0e-05
    arcface_margin = 0.15
    arcface_scale = 15.0

    wandb = False  # https://wandb.ai/authorize
    debug = False

    # evaluate
    topk = 20


def build_data(CFG):
    df, _, _ = prepare_lecr_data_pointwise2(CFG)
    train_df = df.loc[df["fold"] != 0]
    print(train_df.shape)
    print(train_df.head())

    train_df[CFG.label_column] = LabelEncoder().fit_transform(train_df[CFG.label_column])

    tmp = train_df[CFG.label_column].value_counts().sort_index().values
    tt = 1 / np.log1p(tmp)
    tt = (tt - tt.min()) / (tt.max() - tt.min())

    CFG.dataset_margin = tt * CFG.arcface_margin + 0.05
    CFG.num_classes = train_df[CFG.label_column].nunique(dropna=False)
    return train_df


class RetrievalTrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, aug=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.aug = aug
        # self.num_classes = len(le_topic.classes_)
        self.num_classes = self.df[CFG.label_column].nunique(dropna=False)
        CFG.num_classes = self.num_classes

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoding = self.tokenizer(
            str(row.description),
            return_offsets_mapping=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        encoding = {key: torch.as_tensor(val) for key, val in encoding.items()}

        if self.aug:
            ix = torch.rand(size=(self.max_len,)) < 0.15
            encoding["input_ids"][ix] = self.mask_token

        target = row.label

        # target = torch.zeros(self.num_classes, dtype=torch.float32)
        # # 每一个匹配的，根据数量设置为 1/count
        # for l in row.label:
        #     target[l] = 1 / len(row.label)

        # sw: 样本权重
        return encoding, target  # , torch.FloatTensor([row.sw])


def get_optimizer(model, lr, weight_decay=0.0):
    # params = model.parameters()

    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = CFG.differential_learning_rate_layers

    optimizer_parameters = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if (not any(layer in name for layer in differential_layers))
                and (not any(nd in name for nd in no_decay))
            ],
            "lr": CFG.learning_rate,
            "weight_decay": CFG.weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if (not any(layer in name for layer in differential_layers)) and (any(nd in name for nd in no_decay))
            ],
            "lr": CFG.learning_rate,
            "weight_decay": 0,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if (any(layer in name for layer in differential_layers)) and (not any(nd in name for nd in no_decay))
            ],
            "lr": CFG.differential_learning_rate,
            "weight_decay": CFG.weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if (any(layer in name for layer in differential_layers)) and (any(nd in name for nd in no_decay))
            ],
            "lr": CFG.differential_learning_rate,
            "weight_decay": 0,
        },
    ]

    print("Params", len(list(model.named_parameters())))

    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    return optimizer


def get_scheduler(optimizer, cfg, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_epochs * (total_steps // cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    )
    return scheduler


def get_scheduler999(optimizer, epoch):
    if epoch >= 2:
        lr = 2e-5
    else:
        lr = 1e-6

    lr *= 4

    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["lr"] = 100 * lr
    return lr


def cross_entropy(pred, soft_targets, sw=1):
    if not sw:
        sw = 1
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1) * sw)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, reduction="mean"):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        if reduction == "mean":
            return loss.mean()
        else:
            return loss


def train():
    topic_df = build_data(CFG)
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, add_prefix_space=True)
    CFG.tokenzier = tokenizer

    train_dataset = RetrievalTrainDataset(topic_df, tokenizer, CFG.max_len, aug=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    loss_fn = ArcFaceAdaptiveMarginLoss(
        criterion=cross_entropy,
        in_features=768,
        out_features=CFG.num_classes,
        scale=CFG.arcface_scale,
        margin=CFG.arcface_margin,
    )
    model = AutoModelForEmbedding(CFG.MODEL_NAME, pooling_method="cls", loss_fn=loss_fn)

    optimizer = get_optimizer(model, lr=CFG.learning_rate)
    scheduler = get_scheduler(optimizer=optimizer, cfg=CFG, total_steps=len(train_dataset))
    trainer = CustomTrainer(model, device="cuda", apex=CFG.apex)
    trainer.train(
        train_loader=train_loader,
        criterion=None,
        optimizer=optimizer,
        epochs=CFG.epochs,
        scheduler=scheduler,
        dynamic_margin=True,
    )
    torch.save(model.state_dict(), CFG.output_dir + f"model_{CFG.exp_id}.pth")


class RetrievalTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512, aug=False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.aug = aug

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        row = self.texts[index]
        encoding = self.tokenizer(
            row,
            return_offsets_mapping=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        encoding = {key: torch.as_tensor(val) for key, val in encoding.items()}

        if self.aug:
            ix = torch.rand(size=(self.max_len,)) < 0.15
            encoding["input_ids"][ix] = self.mask_token

        return encoding


def evaluate():
    df, topics, contents = prepare_lecr_data_pointwise2(CFG)
    topics = topics.loc[topics["fold"] == 0]
    print(topics.shape, contents.shape)
    print(topics)
    print(contents)

    query = topics[CFG.text_column].tolist()
    content = contents[CFG.text_column].tolist()
    tokenizer = AutoTokenizer.from_pretrained(CFG.output_dir)
    CFG.tokenizer = tokenizer

    query_dataset = RetrievalTestDataset(query, tokenizer, CFG.max_len, aug=False)
    content_dataset = RetrievalTestDataset(content, tokenizer, CFG.max_len, aug=False)
    query_loader = DataLoader(
        query_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
    )
    content_loader = DataLoader(
        content_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    print(CFG.MODEL_NAME)
    model = AutoModelForEmbedding(model_name_or_path=CFG.MODEL_NAME, pooling_method="mean", pretrained=False)
    state = torch.load(CFG.output_dir + f"model_{CFG.exp_id}.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state, strict=False)

    query_embedding = model.encode(query_loader)
    content_embedding = model.encode(content_loader)
    print(query_embedding.shape)
    print(content_embedding.shape)

    matcher = AutoModelForMatch(method="knn")
    dists, indices = matcher.similarity_search(query_embedding, content_embedding, topk=CFG.topk)

    res_df = pd.DataFrame(
        {
            "topic_id": np.repeat(topics["id"].values, CFG.topk),
            "content_id": contents["id"].values[indices.ravel()],
            "vec_dist": dists.ravel(),
        }
    )
    print(res_df)
    res_df["rank"] = res_df.groupby("topic_id")["vec_dist"].rank(method="dense")
    pred_df = res_df[(res_df["rank"] <= 2) | (res_df["vec_dist"] < 0.3)]

    pred_df = pred_df.groupby("topic_id")["content_id"].apply(lambda x: " ".join(list(x)))
    pred_df = pred_df.reset_index().rename(columns={"content_id": "pred"})
    pred_df = pred_df.merge(pd.read_csv(CFG.input_dir + "correlations.csv"), on="topic_id")
    score = metric(pred_df)
    print(score)
    return score


if __name__ == "__main__":
    set_seed(CFG.seed)
    train()
    # evaluate()
