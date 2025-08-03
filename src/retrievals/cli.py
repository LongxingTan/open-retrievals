import os
import subprocess
import sys
from copy import deepcopy
from functools import partial


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
    from . import *
    