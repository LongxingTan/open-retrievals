import json
import os
import tempfile
from unittest import TestCase

from transformers import BertTokenizer

from src.retrievals.data.dataset import (
    EncodeDataset,
    RerankTrainDataset,
    RetrievalTrainDataset,
)


class RetrievalTrainDatasetTest(TestCase):
    def setUp(self):
        # Sample data to be written to the temp file
        data = {
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

        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as temp_file:
            self.temp_file_name = temp_file.name
            json.dump(data, temp_file)
            temp_file.flush()

        tmpdirname = tempfile.mkdtemp()
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        vocab_file = os.path.join(tmpdirname, "vocab.txt")
        with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
        self.tokenizer = BertTokenizer(vocab_file)

    def test_retrieval_dataset(self):

        dataset = RetrievalTrainDataset(
            self.temp_file_name,
            positive_key='pos',
            negative_key='neg',
            query_instruction='query: ',
            document_instruction='document: ',
        )
        print(dataset[0])

        dataset = RerankTrainDataset(self.temp_file_name, positive_key='pos', negative_key='neg')
        print(dataset[0])

        dataset = EncodeDataset(self.temp_file_name, id_key=None, text_key='query', tokenizer=self.tokenizer)
        print(dataset[0])
