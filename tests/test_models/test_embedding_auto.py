import tempfile
from unittest import TestCase

import numpy as np
import torch
from transformers import AutoConfig

from src.retrievals.models.embedding_auto import (
    AutoModelForEmbedding,
    ListwiseModel,
    PairwiseModel,
)

from .test_modeling_common import (
    ModelTesterMixin,
    device,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class AutoModelForEmbeddingTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        embedding_size=16,
        hidden_size=36,
        num_hidden_layers=2,
        num_hidden_groups=2,
        num_attention_heads=6,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return AutoConfig()

    # def create_and_check_model(self, config, pixel_values, labels):
    #     model = AutoModelForEmbedding()
    #     model.to(device)
    #     model.eval()
    #     result = model(pixel_values)
    #     self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))


class AutoModelForEmbeddingTest(TestCase, ModelTesterMixin):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()
        self.model_tester = AutoModelForEmbeddingTester(self)
        # self.config_tester = ConfigTester()
        model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="cls")

    def test_config(self):
        pass

    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        pass

    # def test_model(self):
    #     config, inputs_dict = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_model(*config_and_inputs)
    #     config.return_dict = True
    #
    # def test_model_from_pretrained(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_encode_from_text(self):
        emb = self.model.encode([("Hello Word, a test sentence", "Second input for model")])
        assert emb.shape == (1, 384)
        # assert abs(np.sum(emb) - 9.503508) < 0.001

        emb = self.model.encode(
            [
                ("Hello Word, a test sentence", "Second input for model"),
                ("My second tuple", "With two inputs"),
                ("Final tuple", "final test Oh"),
            ]
        )
        assert emb.shape == (3, 384)
        # assert abs(np.sum(emb) - 32.14627) < 0.001

    def test_forward_from_text(self):
        pass


class PairwiseModelTest(TestCase, ModelTesterMixin):
    def setUp(self) -> None:
        pass

    def test_pairwise_model(self):
        pass


class ListwiseModelTest(TestCase):
    def setUp(self) -> None:
        model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = ListwiseModel.from_pretrained(model_name_or_path)

    def test_unsorted_segment_mean(self):
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        segment_ids = torch.tensor([0, 0, 1, 1])
        num_segments = 2

        list_pool = self.model._unsorted_segment_mean(input_tensor, segment_ids, num_segments)
        print(list_pool)
        # self.assertEqual()
