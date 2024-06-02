import logging
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler

logger = logging.getLogger(__name__)


class GroupedBatchSampler(BatchSampler):
    """
    It enforces that the batch only contain elements from the same group.
    shuffle should be set to False for custom sampler
    """

    def __init__(
        self, sampler: Sampler, group_ids: List[int], batch_size: int, shuffle: bool = True, seed: Optional[int] = None
    ):
        """
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a set of integers in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
        """
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]  # yield a copy of the list
                del group_buffer[:]

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def _create_batches(self):
        indices = [idx for idx in self.sampler]
        self.rng.shuffle(indices)

        for idx in indices:
            group_id = self.group_ids[idx]

            group_buffer = self.buffer_per_group[group_id]
            visited = self.visited_per_group[group_id]

            added = False
            for i, visited_set in enumerate(visited):
                added = True
                group_buffer[i].append(idx)
                if len(group_buffer[i]) == self.batch_size:
                    self.batches.append(group_buffer[i])
                    group_buffer[i] = None
                    visited[i] = None
                break

            self.buffer_per_group[group_id] = [x for x in group_buffer if x is not None]
            self.visited_per_group[group_id] = [x for x in visited if x is not None]
            if not added:
                self.buffer_per_group[group_id].append([idx])
                self.visited_per_group[group_id].append(set())

        if self.shuffle:
            self.rng.shuffle(self.batches)

    def _clear(self):
        self.batches = []
        self.buffer_per_group = {k: [] for k in self.groups}
        self.visited_per_group = {k: [] for k in self.groups}


class GroupSortedBatchSampler(BatchSampler):
    def __init__(
        self, sampler, group_ids, batch_size, shuffle: bool = True, drop_last: bool = False, seed: Optional[int] = None
    ):
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        groups = np.unique(self.group_ids).tolist()

        self.batches = []
        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        if not self.batches:
            self._create_batches()
        for batch in self.batches:
            yield batch
        self._clear()

    def __len__(self):
        if not self.batches:
            self._create_batches()
        return len(self.batches)

    def _create_batches(self):
        indices = []
        group_id_map = dict()
        group_idx = 0
        group_indices = []

        for idx in self.sampler:
            indices.append(idx)

        # if self.shuffle:
        #     self.rng.shuffle(indices)

        for idx in indices:
            group_id = self.group_ids[idx]
            if group_id not in group_id_map:
                group_id_map[group_id] = group_idx
                group_idx += 1
            group_indices.append(group_id_map[group_id])

        l = list(zip(group_indices, indices))
        l.sort()

        l = np.asarray([i[-1] for i in l])
        batches = split_batches(l, self.batch_size, drop_last=self.drop_last)
        for batch in batches:
            self.batches.append(list(batch))

    def _clear(self):
        self.batches = []
        # self.buffer_per_group = {k: [] for k in self.groups}


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
        expand_new_speaker=False,
        new_speaker_ratio=None,
        speaker_uniform_batch=False,
        drop_last=False,
    ):
        pass

    def _create_buckets(self):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        return


def split_batches(inputs, batch_size: int, drop_last: bool = False):
    l = np.split(inputs, np.arange(batch_size, len(inputs), batch_size))
    if drop_last:
        if len(l[-1]) != batch_size:
            l = l[:-1]
    return l


class SyncedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super(SyncedSampler, self).__init__(dataset, num_replicas, rank, shuffle, seed)
        self.num_samples = len(self.dataset)
        self.total_size = len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # DO NOT SUB SAMPLE!
        assert len(indices) == self.total_size
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch: int):
        super(SyncedSampler, self).set_epoch(epoch)
        logger.info(f'Setting Data Sampler Epoch to {epoch}')
