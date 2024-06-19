import json
import tempfile
import unittest
from unittest import TestCase

from src.retrievals.data.dataset import RerankDataset, RetrievalDataset


class RetrievalDatasetTest(TestCase):
    def setUp(self):
        # Sample data to be written to the temp file
        self.data = {
            "query": "A man pulls two women down a city street in a rickshaw.",
            "pos": ["A man is in a city."],
            "neg": [
                "A man is a pilot of an airplane.",
                "It is boring and mundane.",
                "The morning sunlight was shining brightly and it was warm.",
                "Two people jumped off the dock.",
                "People watching a spaceship launch.",
                "Mother Teresa is an easy choice.",
                "It's worth being able to go at a pace you prefer.",
            ],
        }

    def test_retrieval_dataset(self):
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as temp_file:
            temp_file_name = temp_file.name
            json.dump(self.data, temp_file)
            temp_file.flush()

        dataset = RetrievalDataset(temp_file_name, positive_key='pos', negative_key='neg')
        print(dataset[0])

        dataset = RetrievalDataset(
            temp_file_name,
            positive_key='pos',
            negative_key='neg',
            query_instruction='query: ',
            document_instruction='document: ',
        )
        print(dataset[0])


class RerankDatasetTest(TestCase):
    def test_rerank_dataset(self):
        pass
