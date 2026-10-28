"""YAML training command line interface."""

import argparse
import json
from importlib.resources import files
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train retrieval models from YAML")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a configuration and sample data")
    init.add_argument("task", choices=("dense", "cross_encoder", "colbert"))
    init.add_argument("--output", type=Path, default=Path("train.yaml"))
    for command in ("validate", "train", "evaluate"):
        child = commands.add_parser(command)
        child.add_argument("config", type=Path)
        child.add_argument("overrides", nargs="*", metavar="key=value")
    export = commands.add_parser("export", help="Export saved dense embeddings for serving")
    export.add_argument("--model", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--backend", choices=("vllm", "triton"), required=True)
    export.add_argument("--model-name", default="embedding")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            from .export import export_model

            print(export_model(args.model, args.output, args.backend, args.model_name))
            return
        if args.command == "init":
            template = files("retrievals.templates")
            if args.output.suffix.lower() not in (".yaml", ".yml"):
                raise ValueError("--output must end in .yaml or .yml")
            data_path = args.output.with_suffix(".jsonl")
            if args.output.exists() or data_path.exists():
                raise ValueError("Configuration or sample data already exists; choose another --output")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            content = OmegaConf.create(template.joinpath(args.task + ".yaml").read_text())
            content.data.train_file = data_path.name
            content.data.valid_file = data_path.name
            data = "ranking.jsonl" if args.task == "cross_encoder" else "retrieval.jsonl"
            data_path.write_text(template.joinpath(data).read_text())
            OmegaConf.save(content, args.output)
            print(f"Created {args.output} and {data_path}")
            return
        from .config import ConfigManager

        if any("=" not in value for value in args.overrides):
            raise ValueError("Overrides must use key=value syntax")
        config = ConfigManager.load_config(str(args.config), args.overrides)
        if args.command == "validate":
            print("Configuration is valid")
        elif args.command == "train":
            from .train import train

            train(config)
        else:
            from .train import evaluate

            print(json.dumps(evaluate(config), indent=2))
    except (ValueError, OSError, OmegaConfBaseException) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
