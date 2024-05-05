import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, Sampler


class GroupedBatchSampler(BatchSampler):
    """
    It enforces that the batch only contain elements from the same group.
    shuffle should be set to False for custom sampler
    refer: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py
    """

    def __init__(self, sampler, group_ids, batch_size):
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}

    def __iter__(self):
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]  # yield a copy of the list
                del group_buffer[:]

    def __len__(self):
        return


class GroupSortedBatchSampler(BatchSampler):
    def __init__(self):
        pass

    def __int__(self):
        pass

    def __len__(self):
        return


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
