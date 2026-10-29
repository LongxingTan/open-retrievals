"""Load training rows and translate user columns once, at the data boundary."""

import random
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset


def _texts(value, field):
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(text, str) and text for text in values):
        raise ValueError(f"{field} must contain nonempty strings")
    return values


class RetrievalDataset(Dataset):
    """Canonical query/positive/negative rows; evaluation sampling is deterministic."""

    def __init__(self, rows, config, training=True):
        self.rows = rows
        self.config = config
        self.training = training
        required = {config.query_col, config.positive_col}
        if not required.issubset(rows.column_names):
            raise ValueError(f"Dataset is missing columns: {sorted(required - set(rows.column_names))}")

        errors = []
        for index, row in enumerate(rows):
            try:
                query = row[config.query_col]
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a nonempty string")
                if not _texts(row[config.positive_col], "positive"):
                    raise ValueError("Each query needs at least one positive")
                _texts(row.get(config.negative_col, []), "negative")
            except ValueError as error:
                errors.append(f"row {index}: {error}")
        if errors:
            raise ValueError("Invalid dataset rows: " + "; ".join(errors))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        query = row[self.config.query_col]
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a nonempty string")
        positives = _texts(row[self.config.positive_col], "positive")
        if not positives:
            raise ValueError("Each query needs at least one positive")
        result = {"query": query, "positive": random.choice(positives) if self.training else positives[0]}
        negatives = _texts(row.get(self.config.negative_col, []), "negative")
        count = self.config.negative_samples
        if count and negatives:
            count = min(count, len(negatives))
            result["negative"] = random.sample(negatives, count) if self.training else negatives[:count]
        return result


def load_training_dataset(config, task, training=True):
    """Load JSON/JSONL, CSV, Parquet, or a Hugging Face dataset."""
    source = config.train_file if training else config.valid_file
    if source:
        suffix = Path(source).suffix.lower()
        formats = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}
        if suffix not in formats:
            raise ValueError(f"Unsupported dataset extension: {suffix}")
        rows = load_dataset(formats[suffix], data_files=source, split="train")
    elif training and config.dataset_name:
        rows = load_dataset(config.dataset_name, config.dataset_config, split=config.dataset_split)
    elif not training:
        return None
    else:
        raise ValueError("Set data.train_file or data.dataset_name")
    if not len(rows):
        raise ValueError("Dataset is empty")
    if not training and config.valid_max_samples is not None:
        rows = rows.select(range(min(len(rows), config.valid_max_samples)))
    if task == "cross_encoder" and config.label_col in rows.column_names:
        required = {config.query_col, config.doc_col, config.label_col}
        if not required.issubset(rows.column_names):
            raise ValueError(f"Reranking requires labeled pairs with columns {sorted(required)}")

        def normalize(row, index):
            for field in (config.query_col, config.doc_col):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"row {index}: {field} must be a nonempty string")
            try:
                label = float(row[config.label_col])
            except (TypeError, ValueError) as error:
                raise ValueError(f"row {index}: label must be numeric") from error
            if not 0 <= label <= 1:
                raise ValueError(f"row {index}: classification labels must be between zero and one")
            return {"query": row[config.query_col], "document": row[config.doc_col], "labels": label}

        return rows.map(normalize, with_indices=True, remove_columns=rows.column_names)
    return RetrievalDataset(rows, config, training=training)
