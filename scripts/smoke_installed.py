"""Run against an installed wheel, from outside the repository, without downloads."""

import subprocess
import tempfile
from pathlib import Path

from transformers import BertConfig, BertModel, BertTokenizerFast


def run(*args):
    subprocess.run(args, check=True)


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    model = root / 'tiny-bert'
    model.mkdir()
    (model / 'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'cat', 'pet']))
    BertTokenizerFast(vocab_file=str(model / 'vocab.txt')).save_pretrained(model)
    BertModel(
        BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    ).save_pretrained(model)
    for task in ('dense', 'cross_encoder', 'colbert'):
        config = root / task / 'train.yaml'
        run('rag-cli', 'init', task, '--output', str(config))
        output = root / task / 'output'
        overrides = [
            f'model.model_name_or_path={model}',
            f'output_dir={output}',
            'training.use_cpu=true',
            'training.max_steps=1',
            'data.query_max_length=8',
            'data.doc_max_length=8',
            'data.max_length=16',
        ]
        run('rag-cli', 'validate', str(config), *overrides)
        run('rag-cli', 'train', str(config), *overrides)
        run('rag-cli', 'evaluate', str(output / 'model/config.yaml'))
