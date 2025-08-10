import os
import subprocess
import sys
from copy import deepcopy
from functools import partial

from .logger import logging
from .utils import get_device_count, is_env_enabled

USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   rag-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   rag-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   rag-cli eval -h: evaluate models                        |\n"
    + "|   rag-cli export -h: merge LoRA adapters and export model |\n"
    + "|   rag-cli train -h: train models                          |\n"
    + "|   rag-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   rag-cli webui: launch LlamaBoard                        |\n"
    + "|   rag-cli version: show version info                      |\n"
    + "-" * 70
)


def main():
    from . import launcher
    from .pipelines import *
    from .version import short_version as VERSION

    logger = logging.get_logger(__name__)
    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to Open Retrievals, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/LongxingTan/open-retrievals |\n"
        + "-" * 58
    )

    COMMAND_MAP = {
        "api": run_api,
        "chat": run_chat,
        "env": print_env,
        "eval": run_eval,
        "export": export_model,
        "train": run_train,
        "webchat": run_web_demo,
        "webui": run_web_ui,
        "version": partial(print, WELCOME),
        "help": partial(print, USAGE),
    }
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if command == "train" and (
        is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not is_env_enabled("USE_RAY"))
    ):
        # launch distributed training
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
        if int(nnodes) > 1:
            logger.info_rank0(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

        # elastic launch support
        max_restarts = os.getenv("MAX_RESTARTS", "0")
        rdzv_id = os.getenv("RDZV_ID")
        min_nnodes = os.getenv("MIN_NNODES")
        max_nnodes = os.getenv("MAX_NNODES")

        env = deepcopy(os.environ)
        if is_env_enabled("OPTIM_TORCH", "1"):
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        if rdzv_id is not None:
            # launch elastic job with fault tolerant support when possible
            # see also https://docs.pytorch.org/docs/stable/elastic/train_script.html
            rdzv_nnodes = nnodes
            # elastic number of nodes if MIN_NNODES and MAX_NNODES are set
            if min_nnodes is not None and max_nnodes is not None:
                rdzv_nnodes = f"{min_nnodes}:{max_nnodes}"

            process = subprocess.run(
                (
                    "torchrun --nnodes {rdzv_nnodes} --nproc-per-node {nproc_per_node} "
                    "--rdzv-id {rdzv_id} --rdzv-backend c10d --rdzv-endpoint {master_addr}:{master_port} "
                    "--max-restarts {max_restarts} {file_name} {args}"
                )
                .format(
                    rdzv_nnodes=rdzv_nnodes,
                    nproc_per_node=nproc_per_node,
                    rdzv_id=rdzv_id,
                    master_addr=master_addr,
                    master_port=master_port,
                    max_restarts=max_restarts,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
