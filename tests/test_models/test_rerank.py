import shutil
import tempfile
from unittest import TestCase

from transformers import AutoConfig, AutoTokenizer

from src.retrievals.data.collator import RerankCollator
from src.retrievals.models.rerank import RerankModel

from .test_modeling_common import (
    ModelTesterMixin,
    device,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class RerankModelTest(TestCase, ModelTesterMixin):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()
        # self.config_tester = ConfigTester()
        model_name_or_path = 'BAAI/bge-reranker-base'
        self.data_collator = RerankCollator(AutoTokenizer.from_pretrained(model_name_or_path))
        self.model = RerankModel(model_name_or_path, pooling_method="mean")

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_compute_score(self):
        text = '张华考上了北京大学'
        text_pair = ['李萍进了中等技术学校', '我在百货公司当售货员', '我们都有光明的前途']
        text = [text] * len(text_pair)
        scores = self.model.compute_score(text, text_pair=text_pair, data_collator=self.data_collator)
        document_ranked = self.model.rerank(text, text_pair, data_collator=self.data_collator)

        print(scores)
        print(document_ranked)
