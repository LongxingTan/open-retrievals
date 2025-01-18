import shutil
import tempfile
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from src.retrievals.data.collator import RerankCollator
from src.retrievals.losses import ColbertLoss
from src.retrievals.models.rerank import AutoModelForRanking, ColBERT, LLMRanker
from src.retrievals.trainer.trainer import RerankTrainer

from .test_modeling_common import ModelTesterMixin


class PseudoRerankTrainDataset(Dataset):
    def __init__(self):
        self.examples = [
            {'query': 'how are you, fans', 'document': 'I am fine', 'labels': 1},
            {'query': 'hallo?', 'document': 'what is your problem', 'labels': 1},
            {'query': 'how are you doing', 'document': 'be survive', 'labels': 0},
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


class AutoModelForRankingTest(TestCase, ModelTesterMixin):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()
        model_name_or_path = 'BAAI/bge-reranker-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForRanking.from_pretrained(model_name_or_path, device='cpu', temperature=0.05)
        self.data_collator = RerankCollator(self.tokenizer, max_length=32)

        self.text = '张华考上了北京大学'
        self.text_list = ['李萍进了中等技术学校', '我在百货公司当售货员', '我们都有光明的前途']
        self.text_pairs = [[self.text, i] for i in self.text_list]

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_preprocess(self):
        batch = self.model.preprocess_pair(self.text_pairs, query_max_length=9, document_max_length=9)
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)

    def test_compute_score(self):
        scores = self.model.compute_score(sentence_pairs=self.text_pairs, data_collator=self.data_collator)
        document_ranked = self.model.rerank(query=self.text, documents=self.text_list, data_collator=self.data_collator)

        self.assertEqual(len(scores), len(self.text_pairs))
        self.assertIn('rerank_document', document_ranked)
        self.assertIn('rerank_scores', document_ranked)

    def test_trainer(self):
        train_dataset = PseudoRerankTrainDataset()
        training_args = TrainingArguments(
            output_dir=self.output_dir, remove_unused_columns=False, per_device_train_batch_size=2, num_train_epochs=1
        )

        self.trainer = RerankTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
        )
        self.trainer.train()
        self.trainer.save_model(self.output_dir)


class TestColBERT(TestCase):
    def setUp(self):
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.linear_layer = nn.Linear(self.model.config.hidden_size, 1024)
        self.loss_fn = ColbertLoss()
        self.colbert = ColBERT(
            model=self.model,
            tokenizer=self.tokenizer,
            linear_layer=self.linear_layer,
            loss_fn=self.loss_fn,
            device="cpu",
        )

    def test_forward(self):
        query_input_ids = torch.tensor([[101, 2054, 2003, 1996, 1045, 2572, 102]])
        query_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        pos_input_ids = torch.tensor([[101, 2054, 2003, 1996, 1045, 2572, 102]])
        pos_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])

        self.colbert.train()  # Set the model to training mode
        output = self.colbert(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            neg_input_ids=pos_input_ids,
            neg_attention_mask=pos_attention_mask,
        )
        self.assertIn('loss', output)
        self.assertIsInstance(output['loss'], torch.Tensor)

    def test_encode(self):
        sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = self.colbert.encode(sentences, batch_size=2, convert_to_numpy=False)
        self.assertEqual(len(embeddings), 2)

    def test_compute_score(self):
        sentence_pairs = [("This is a query.", "This is a document.")]
        scores = self.colbert.compute_score(sentence_pairs, batch_size=1)
        self.assertIsInstance(scores, float)

    def test_preprocess_pair(self):
        batch_sentence_pair = [["Hello, world!", "Hello there!"]]
        preprocessed = self.colbert.preprocess_pair(
            batch_sentence_pair=batch_sentence_pair, query_max_length=128, document_max_length=128
        )
        self.assertIn("query_input_ids", preprocessed)
        self.assertIn("doc_input_ids", preprocessed)
        self.assertIsInstance(preprocessed["query_input_ids"], torch.Tensor)
        self.assertIsInstance(preprocessed["doc_input_ids"], torch.Tensor)


class TestLLMRanker(TestCase):
    def setUp(self):
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.task_prompt = (
            "Given a query A and a passage B, determine whether the passage contains "
            "an answer to the query by providing a prediction of either 'Yes' or 'No'."
        )

        self.llm_ranker = LLMRanker(
            model=self.model,
            tokenizer=self.tokenizer,
            task_prompt=self.task_prompt,
            target_token='Yes',
            sep_token='\n',
            query_instruction='A: {}',
            document_instruction='B: {}',
            device="cpu",
        )

    def test_initialization(self):
        self.assertIsNotNone(self.llm_ranker.model)
        self.assertIsNotNone(self.llm_ranker.tokenizer)
        self.assertEqual(self.llm_ranker.task_prompt, self.task_prompt)
        self.assertEqual(
            self.llm_ranker.target_token_loc, self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        )
        self.assertEqual(self.llm_ranker.sep_token, '\n')

    def test_preprocess_pair(self):
        batch_sentence_pair = [("This is a query.", "This is a document.")]
        processed = self.llm_ranker.preprocess_pair(batch_sentence_pair, max_length=10)
        self.assertIn('input_ids', processed)
        self.assertIn('attention_mask', processed)
        self.assertIsInstance(processed['input_ids'], torch.Tensor)
        self.assertIsInstance(processed['attention_mask'], torch.Tensor)

    def test_compute_score(self):
        sentence_pairs = [("This is a query.", "This is a document.")]
        scores = self.llm_ranker.compute_score(sentence_pairs, batch_size=1)

        self.assertIsInstance(scores, float)

    def test_forward(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        output = self.llm_ranker.forward(input_ids, attention_mask)
        self.assertIn('logits', output)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_from_pretrained(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.model
        llm_ranker = LLMRanker.from_pretrained(
            model_name_or_path=self.model_name,
            task_prompt=self.task_prompt,
            query_instruction='A: {}',
            document_instruction='B: {}',
            device="cpu",
        )
        self.assertIsNotNone(llm_ranker.model)
        self.assertIsNotNone(llm_ranker.tokenizer)
        self.assertEqual(llm_ranker.task_prompt, self.task_prompt)
