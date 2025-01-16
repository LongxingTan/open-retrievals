import unittest
from unittest.mock import MagicMock

from src.retrievals.models.utils import DocumentSplitter


class TestDocumentSplitter(unittest.TestCase):
    def test_create_documents(self):

        tokenizer = MagicMock()
        tokenizer.encode_plus.side_effect = [
            {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]},  # Query encoding
            {'input_ids': [4, 5, 6, 7], 'attention_mask': [1, 1, 1, 1]},  # Document encoding
        ]
        tokenizer.sep_token_id = 99

        splitter = DocumentSplitter(chunk_size=10, chunk_overlap=2)

        query = "test query"
        documents = ["test document"]

        expected_merge_inputs = [
            {'input_ids': [1, 2, 3, 99, 4, 5, 6, 7, 99], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
        ]
        expected_merge_inputs_pids = [0]

        res_merge_inputs, res_merge_inputs_pids = splitter.create_documents(query, documents, tokenizer)

        self.assertEqual(res_merge_inputs, expected_merge_inputs)
        self.assertEqual(res_merge_inputs_pids, expected_merge_inputs_pids)
