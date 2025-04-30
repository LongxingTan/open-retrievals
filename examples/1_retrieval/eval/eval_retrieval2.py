import logging
import shutil

from C_MTEB.tasks import AbsTaskReranking, ChineseRerankingEvaluator
from datasets import load_dataset
from mteb import MTEB

from retrievals import (
    AutoModelForEmbedding,
    AutoModelForRanking,
    ColBERT,
    LLMRanker,
    RetrievalCollator,
    RetrievalTrainer,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

model_name_or_path = 'BAAI/bge-reranker-v2-gemma'

model = LLMRanker.from_pretrained(
    model_name_or_path,
    causal_lm=True,
    use_fp16=True,
    lora_path=None,
)
dataset = load_dataset("C-MTEB/T2Reranking", split="dev")
ds = dataset.train_test_split(test_size=0.1, seed=42)


class CustomReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "CustomReranking",
            "hf_hub_name": "C-MTEB/T2Reranking",
            "description": ("T2Ranking: A large-scale Chinese Benchmark for Passage Ranking"),
            "reference": "https://arxiv.org/abs/2304.03679",
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "map",
        }

    def evaluate(self, model, split="test", **kwargs):
        # if not self.data_loaded:
        # self.load_data()
        self.dataset = ds
        data_split = self.dataset[split]
        print("Test data size: ", len(data_split))

        evaluator = ChineseRerankingEvaluator(data_split, **kwargs)
        scores = evaluator(model)

        return dict(scores)


def run_evaluate(model):
    evaluation = MTEB(tasks=[CustomReranking()], task_langs=["zh", "zh-CN"])
    evaluation.run(model, output_folder=f"zh_results/{model_name_or_path.split('/')[-1]}")


if __name__ == "__main__":
    shutil.rmtree(f"zh_results/{model_name_or_path.split('/')[-1]}", ignore_errors=True)
    run_evaluate(model)
