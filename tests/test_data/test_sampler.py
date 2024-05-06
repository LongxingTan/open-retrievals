import random
from unittest import TestCase

import torch

from src.retrievals.data.sampler import GroupSortedBatchSampler


class GroupSortedBatchSamplerTest(TestCase):
    def setUp(self) -> None:
        self.n_samples = 16

    def test_sampler(self):
        indices = list(range(self.n_samples))
        random.shuffle(indices)

        sampler = torch.utils.data.SequentialSampler(indices)
        # for x in sampler:
        #     print(x)
        #
        # for x in torch.utils.data.BatchSampler(sampler, batch_size=4, drop_last=False):
        #     print(x)

        batch_sampler = GroupSortedBatchSampler(
            sampler,
            group_ids=[0, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 2, 3, 3, 3],
            batch_size=4,
            shuffle=False,
            seed=43,
        )

        for x in batch_sampler:
            print(x)
