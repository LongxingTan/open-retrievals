import time
from unittest import TestCase

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from src.retrievals.trainer.custom_trainer import CustomTrainer, timeSince


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}, 0


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x, labels=None):
        x = x['input_ids'].float()
        if labels is not None:
            x = self.fc(x)
            labels = labels.float()
            return {'loss': nn.CrossEntropyLoss()(x, labels.long())}
        return self.fc(x)


class CustomTrainerTest(TestCase):
    def setUp(self) -> None:
        self.model = MockModel()
        self.trainer = CustomTrainer(
            self.model,
        )

        input_ids = torch.randint(0, 10, (10, 5))
        attention_mask = torch.ones_like(input_ids)

        dataset = CustomDataset(input_ids, attention_mask)
        self.data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        self.optimizer = SGD(self.model.parameters(), lr=0.01)

    def test_train(self):
        self.trainer.train(self.data_loader, self.optimizer, epochs=1)


class timeSinceTest(TestCase):
    def test_time_since(self):
        timeSince(time.time(), 0.1)
