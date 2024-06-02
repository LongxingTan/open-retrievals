import numpy as np


class HyDE(object):
    """
    https://github.com/texttron/hyde/blob/main/src/hyde/hyde.py
    """

    def __init__(self, promptor, generator, encoder, searcher):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher
