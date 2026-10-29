"""Offline contract tests: real tiny transformers, local data, no downloaded weights."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from transformers import BertConfig, BertModel, BertTokenizerFast, GPT2Config, GPT2LMHeadModel, GPT2Model

from src.retrievals.config import ConfigManager
from src.retrievals.losses import ColbertLoss
from src.retrievals.models import AutoModelForEmbedding
from src.retrievals.models.rerank import LLMRanker
from src.retrievals.train import build_trainer, load_model, resolve_config, save_model


class TrainingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.model_path = cls.root / "tiny-bert"
        cls.model_path.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "cat", "dog", "pet", "car", "road"]
        (cls.model_path / "vocab.txt").write_text("\n".join(vocab))
        BertTokenizerFast(vocab_file=str(cls.model_path / "vocab.txt")).save_pretrained(cls.model_path)
        BertModel(
            BertConfig(
                vocab_size=len(vocab),
                hidden_size=12,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=32,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
            )
        ).save_pretrained(cls.model_path)
        cls.retrieval_file = cls.root / "retrieval.jsonl"
        cls.retrieval_file.write_text(
            "\n".join(
                json.dumps(row)
                for row in [
                    {"question": "cat", "answer": ["pet"], "negatives": ["car"]},
                    {"question": "car", "answer": ["road"], "negatives": ["dog"]},
                ]
            )
        )
        cls.ranking_file = cls.root / "ranking.jsonl"
        cls.ranking_file.write_text(
            "\n".join(
                json.dumps(row)
                for row in [
                    {"question": "cat", "text": "pet", "relevance": 1},
                    {"question": "car", "text": "dog", "relevance": 0},
                ]
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def config(self, task="dense_retrieval"):
        ranking = task == "cross_encoder"
        source = str(self.ranking_file if ranking else self.retrieval_file)
        return resolve_config(
            {
                "task_type": task,
                "output_dir": str(self.root / task),
                "model": {"model_name_or_path": str(self.model_path), "pooling_method": "mean", "colbert_dim": 8},
                "training": {
                    "use_cpu": True,
                    "max_steps": 1,
                    "batch_size": 2,
                    "eval_batch_size": 2,
                    "eval_strategy": "epoch",
                    "save_strategy": "steps",
                    "save_steps": 1,
                    "learning_rate": 0.01,
                    "logging_steps": 1,
                    "warmup_ratio": 0.0,
                },
                "data": {
                    "train_file": source,
                    "valid_file": source,
                    "query_col": "question",
                    "positive_col": "answer",
                    "negative_col": "negatives",
                    "doc_col": "text",
                    "label_col": "relevance",
                    "negative_samples": 1,
                    "query_max_length": 8,
                    "doc_max_length": 8,
                    "max_length": 12,
                },
                "loss": {"loss_type": "bce" if ranking else "colbert" if task == "colbert" else "infonce"},
            }
        )

    def test_models_learn_toy_pairs(self):
        for task in ("dense_retrieval", "colbert", "cross_encoder"):
            config = self.config(task)
            config.training.max_steps = 50
            config.training.eval_strategy = "no"
            config.training.save_strategy = "no"
            config.training.logging_steps = 50
            trainer = build_trainer(config)
            trainer.train()
            trainer.model.eval()
            if task == "cross_encoder":
                batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
                with torch.no_grad():
                    _, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)
                scores = outputs["logits"].reshape(-1)
                # With one positive and one negative, this is exactly AUC = 1.
                self.assertGreater(scores[0].item(), scores[1].item())
            else:
                from src.retrievals.trainer.retrieval_metrics import RetrievalEvaluator

                corpus = self.root / f"learn-{task}-corpus.jsonl"
                queries = self.root / f"learn-{task}-queries.jsonl"
                corpus.write_text("\n".join(json.dumps({"id": text, "text": text}) for text in ("pet", "road")))
                queries.write_text(
                    "\n".join(
                        json.dumps({"query": query, "relevant_ids": [doc]})
                        for query, doc in (("cat", "pet"), ("car", "road"))
                    )
                )
                config.data.valid_query_file = str(queries)
                config.data.valid_corpus_file = str(corpus)
                metrics = RetrievalEvaluator(config.data, task)(trainer.model, 2)
                self.assertEqual(metrics["eval_recall_at_1"], 1.0, task)

    def test_corpus_evaluation(self):
        corpus = self.root / "corpus.jsonl"
        queries = self.root / "queries.jsonl"
        corpus.write_text(json.dumps({"id": "cat", "text": "cat"}) + "\n" + json.dumps({"id": "car", "text": "car"}))
        queries.write_text(json.dumps({"query": "cat", "relevant_ids": ["cat"]}))
        for task in ("dense_retrieval", "colbert"):
            config = self.config(task)
            config.data.valid_file = None
            config.data.valid_query_file = str(queries)
            config.data.valid_corpus_file = str(corpus)
            trainer = build_trainer(config)
            metrics = trainer.evaluate()
            self.assertEqual(metrics["eval_recall_at_1"], 1.0)
            self.assertEqual(metrics["eval_ndcg_at_10"], 1.0)
            self.assertEqual(trainer.args.metric_for_best_model, "eval_ndcg_at_10")

    def test_grouped_reranking(self):
        config = self.config("cross_encoder")
        config.data.train_file = str(self.retrieval_file)
        config.data.valid_file = str(self.retrieval_file)
        trainer = build_trainer(config)
        batch = next(iter(trainer.get_train_dataloader()))
        self.assertEqual(batch["group_sizes"].tolist(), [2, 2])
        loss = trainer.compute_loss(trainer.model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_exported_lengths(self):
        config = self.config()
        trainer = build_trainer(config)
        path = save_model(trainer, config)
        reloaded = AutoModelForEmbedding.from_pretrained(str(path), device="cpu")
        self.assertEqual(reloaded.max_length, config.data.doc_max_length)
        self.assertEqual(reloaded.query_max_length, config.data.query_max_length)

    def test_tasks_train_evaluate_export_reload(self):
        for task in ("dense_retrieval", "cross_encoder", "colbert"):
            with self.subTest(task=task):
                config = self.config(task)
                trainer = build_trainer(config)
                before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
                result = trainer.train()
                self.assertEqual(result.global_step, 1)
                self.assertTrue(
                    any(not torch.equal(before[name], value) for name, value in trainer.model.named_parameters())
                )
                metrics = trainer.evaluate()
                self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_loss"])))
                exported = save_model(trainer, config)
                reloaded_config = resolve_config(exported / "config.yaml")
                reloaded_config.data.train_file = "/does/not/exist.jsonl"
                reloaded = build_trainer(reloaded_config, training=False)
                self.assertAlmostEqual(reloaded.evaluate()["eval_loss"], metrics["eval_loss"], places=5)
                for key, value in trainer.model.state_dict().items():
                    torch.testing.assert_close(reloaded.model.state_dict()[key], value)

    def test_dense_checkpoint_resumes(self):
        config = self.config()
        config.output_dir = str(self.root / "resume")
        first = build_trainer(config)
        first.train()
        checkpoint = Path(config.output_dir) / "checkpoint-1"
        self.assertTrue((checkpoint / "optimizer.pt").is_file())
        config.training.max_steps = 2
        second = build_trainer(config)
        result = second.train(resume_from_checkpoint=str(checkpoint))
        self.assertEqual(result.global_step, 2)

    def test_embedding_export_preserves_pooling_and_normalization(self):
        config = self.config()
        model = load_model(config)
        directory = self.root / "embedding-export"
        model.save_pretrained(str(directory))
        reloaded = AutoModelForEmbedding.from_pretrained(str(directory), device="cpu")
        self.assertEqual(reloaded.pooling_method, "mean")
        self.assertTrue(reloaded.normalize_embeddings)
        model.eval()
        reloaded.eval()
        inputs = model.tokenizer(["cat pet"], return_tensors="pt")
        torch.testing.assert_close(model(inputs), reloaded(inputs))

    def test_canonical_override_wins_over_yaml(self):
        path = self.root / "aliases.yaml"
        config = OmegaConf.to_container(self.config())
        OmegaConf.save(OmegaConf.create(config), path)
        loaded = ConfigManager.load_config(str(path), ["model.pooling_method=cls"])
        self.assertEqual(loaded.model.pooling_method, "cls")

    def test_invalid_backend_fails_before_loading(self):
        config = OmegaConf.to_container(self.config())
        config["training"]["framework"] = "verl"
        with self.assertRaises(ConfigKeyError):
            build_trainer(config)

    def test_colbert_padding_cannot_win_maxsim(self):
        loss = ColbertLoss(use_inbatch_negative=True)
        query = torch.tensor([[[1.0, 0], [100.0, 0]]])
        documents = torch.tensor([[[-1.0, 0], [0.0, 0]]])
        score = loss.similarity(query, documents, torch.tensor([[1, 0]]), torch.tensor([[1, 0]]))
        torch.testing.assert_close(score, torch.tensor([[-1.0]]))

    def test_positive_only_dense_batch(self):
        config = self.config()
        config.data.negative_samples = 0
        trainer = build_trainer(config)
        batch = next(iter(trainer.get_train_dataloader()))
        self.assertNotIn("negative", batch)
        loss = trainer.compute_loss(trainer.model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_colbert_accepts_mixed_negative_batches_in_either_order(self):
        trainer = build_trainer(self.config("colbert"))
        rows = [
            {"query": "cat", "positive": "pet"},
            {"query": "car", "positive": "road", "negative": ["dog"]},
        ]
        for batch in (rows, rows[::-1]):
            inputs = trainer.data_collator(batch)
            self.assertEqual(inputs["neg_input_ids"].shape[0], 1)
            self.assertTrue(torch.isfinite(trainer.compute_loss(trainer.model, inputs)))

    def test_export_preserves_relative_dictionary_paths(self):
        config = self.config()
        config.data.train_file = os.path.relpath(config.data.train_file)
        config.data.valid_file = os.path.relpath(config.data.valid_file)
        config.output_dir = os.path.relpath(self.root / "relative-output")
        trainer = build_trainer(config)
        exported = save_model(trainer, config)
        loaded = resolve_config(exported / "config.yaml")
        self.assertEqual(Path(loaded.data.valid_file), self.retrieval_file.resolve())
        self.assertEqual(Path(loaded.output_dir), (self.root / "relative-output").resolve())
        self.assertTrue(torch.isfinite(torch.tensor(build_trainer(loaded, training=False).evaluate()["eval_loss"])))

    def test_decoder_training_and_export(self):
        path = self.root / "tiny-decoder"
        GPT2Model(
            GPT2Config(
                vocab_size=10, n_embd=12, n_layer=1, n_head=2, n_positions=32, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0
            )
        ).save_pretrained(path)
        config = self.config()
        config.model.model_name_or_path = str(path)
        config.model.tokenizer_name = str(self.model_path)
        config.model.model_type = "llm"
        config.model.pooling_method = "last"
        config.output_dir = str(self.root / "decoder-output")
        trainer = build_trainer(config)
        trainer.train()
        export = save_model(trainer, config)
        reloaded = build_trainer(export / "config.yaml", training=False)
        self.assertAlmostEqual(trainer.evaluate()["eval_loss"], reloaded.evaluate()["eval_loss"], places=5)

    def test_llm_ranker_loads_custom_target_token(self):
        path = self.root / "tiny-generative-ranker"
        GPT2LMHeadModel(GPT2Config(vocab_size=10, n_embd=12, n_layer=1, n_head=2, n_positions=32)).save_pretrained(path)
        tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        tokenizer.save_pretrained(path)
        ranker = LLMRanker.from_pretrained(str(path), target_token="cat", device="cpu")
        self.assertEqual(ranker.target_token_loc, tokenizer.convert_tokens_to_ids("cat"))

    @unittest.skipUnless(importlib.util.find_spec("peft"), "Install the lora extra to test adapters")
    def test_lora_export_reload(self):
        config = self.config()
        config.model.use_lora = True
        config.model.lora_target_modules = "query,value"
        config.model.lora_dropout = 0.0
        config.output_dir = str(self.root / "lora-output")
        trainer = build_trainer(config)
        trainer.train()
        export = save_model(trainer, config)
        self.assertTrue((export / "adapter_config.json").is_file())
        reloaded = build_trainer(export / "config.yaml", training=False)
        self.assertAlmostEqual(trainer.evaluate()["eval_loss"], reloaded.evaluate()["eval_loss"], places=5)

    @unittest.skipUnless(importlib.util.find_spec("peft"), "Install the lora extra to test adapters")
    def test_lora_gradient_checkpointing_updates_adapters(self):
        for task in ("dense_retrieval", "cross_encoder", "colbert"):
            with self.subTest(task=task):
                config = self.config(task)
                config.model.use_lora = True
                config.model.lora_target_modules = "query,value"
                config.model.lora_dropout = 0.0
                config.training.gradient_checkpointing = True
                trainer = build_trainer(config)
                parameters = dict(trainer.model.named_parameters())
                adapters = {name: value.detach().clone() for name, value in parameters.items() if "lora_B" in name}
                frozen = {name: value.detach().clone() for name, value in parameters.items() if not value.requires_grad}
                self.assertTrue(adapters)
                self.assertTrue(frozen)
                result = trainer.train()
                self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
                self.assertTrue(any(not torch.equal(value, parameters[name]) for name, value in adapters.items()))
                for name, value in frozen.items():
                    self.assertTrue(torch.equal(value, parameters[name]), name)

    def test_cli_train_then_eval_without_train_file(self):
        from src.retrievals.cli import main

        config = self.config()
        config.output_dir = str(self.root / "cli-output")
        path = self.root / "cli.yaml"
        OmegaConf.save(config, path)
        with patch("sys.argv", ["rag-cli", "train", str(path), "seed=123"]):
            main()
        export = Path(config.output_dir) / "model"
        self.assertEqual(resolve_config(export / "config.yaml").seed, 123)
        with patch(
            "sys.argv",
            [
                "rag-cli",
                "evaluate",
                str(export / "config.yaml"),
                "data.train_file=/missing.jsonl",
            ],
        ):
            main()

    def test_alias_conflict_is_rejected(self):
        config = OmegaConf.to_container(self.config())
        config["model"]["pooling_strategy"] = "cls"
        with self.assertRaises(ConfigKeyError):
            resolve_config(config)

    def test_packaged_templates_and_relative_paths(self):
        from src.retrievals.cli import main

        for task in ("dense", "cross_encoder", "colbert"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "train.yaml"
                main(["init", task, "--output", str(path)])
                config = resolve_config(path)
                self.assertEqual(Path(config.data.train_file).resolve(), path.with_suffix(".jsonl").resolve())
                self.assertTrue(Path(config.data.train_file).is_file())
                original = path.read_text()
                with self.assertRaises(SystemExit):
                    main(["init", task, "--output", str(path)])
                self.assertEqual(path.read_text(), original)

    def test_init_rejects_data_path_and_quotes_filenames(self):
        from src.retrievals.cli import main

        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "train.jsonl"
            with self.assertRaises(SystemExit):
                main(["init", "dense", "--output", str(invalid)])
            self.assertFalse(invalid.exists())
            path = Path(directory) / "train # sample.yaml"
            main(["init", "dense", "--output", str(path)])
            config = resolve_config(path)
            self.assertTrue(Path(config.data.train_file).is_file())

    def test_examples_validate(self):
        root = Path(__file__).resolve().parents[2] / "examples" / "configs"
        for name in ("dense_retrieval", "cross_encoder", "colbert", "smoke_test"):
            resolve_config(root / f"{name}.yaml")
