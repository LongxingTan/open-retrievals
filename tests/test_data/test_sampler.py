import random
from unittest import TestCase

import torch
from torch.utils.data.sampler import BatchSampler, Sampler

from src.retrievals.data.sampler import GroupedBatchSampler, GroupSortedBatchSampler


class TestGroupedBatchSampler(TestCase):
    def test_grouped_batch_sampler(self):
        class MockSampler(Sampler):
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                # indices from 0 to 9
                return iter(range(len(self.data_source)))

            def __len__(self):
                return len(self.data_source)

        data_source = list(range(10))
        group_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # Group IDs for each element in the data source
        batch_size = 2

        # Create the GroupedBatchSampler
        sampler = MockSampler(data_source)
        grouped_sampler = GroupedBatchSampler(sampler, group_ids, batch_size, shuffle=False)

        # Collect all batches
        batches = list(grouped_sampler)

        # Check that each batch contains elements from the same group
        for batch in batches:
            group_id = group_ids[batch[0]]
            for idx in batch:
                self.assertEqual(group_ids[idx], group_id)

        # Check that the number of batches is correct
        expected_num_batches = (len(data_source) + batch_size - 1) // batch_size
        # self.assertEqual(len(batches), expected_num_batches)
        print(expected_num_batches)


class TestGroupSortedBatchSampler(TestCase):
    def test_group_sorted_batch_sampler(self):
        # Create a mock sampler that yields indices from 0 to 9
        class MockSampler(Sampler):
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                return iter(range(len(self.data_source)))

            def __len__(self):
                return len(self.data_source)

        data_source = list(range(10))
        group_ids = [2, 2, 1, 1, 0, 0, 2, 1, 0, 2]  # Group IDs for each element in the data source
        batch_size = 3

        # Create the GroupSortedBatchSampler
        sampler = MockSampler(data_source)
        grouped_sorted_sampler = GroupSortedBatchSampler(sampler, group_ids, batch_size, shuffle=False, drop_last=False)

        batches = list(grouped_sorted_sampler)

        # Check that batches are sorted by group
        # previous_group_id = None
        for batch in batches:
            # Get the group ID of the first element in the batch
            # current_group_id = group_ids[batch[0]]
            # if previous_group_id is not None:
            #     # Ensure the current group ID is >= the previous group ID
            #     self.assertGreaterEqual(current_group_id, previous_group_id)
            # previous_group_id = current_group_id

            self.assertLessEqual(len(batch), batch_size)

        # Check that the number of batches is correct
        expected_num_batches = (len(data_source) + batch_size - 1) // batch_size
        self.assertEqual(len(batches), expected_num_batches)

    def test_group_sorted_batch_sampler_drop_last(self):
        # Create a mock sampler that yields indices from 0 to 9
        class MockSampler(Sampler):
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                return iter(range(len(self.data_source)))

            def __len__(self):
                return len(self.data_source)

        data_source = list(range(10))
        group_ids = [2, 2, 1, 1, 0, 0, 2, 1, 0, 2]  # Group IDs for each element in the data source
        batch_size = 3

        # Create the GroupSortedBatchSampler with drop_last=True
        sampler = MockSampler(data_source)
        grouped_sorted_sampler = GroupSortedBatchSampler(sampler, group_ids, batch_size, shuffle=False, drop_last=True)

        batches = list(grouped_sorted_sampler)

        # Check that the last batch is dropped if it's smaller than batch_size
        for batch in batches:
            self.assertEqual(len(batch), batch_size)

        # Check that the number of batches is correct
        expected_num_batches = len(data_source) // batch_size
        self.assertEqual(len(batches), expected_num_batches)
