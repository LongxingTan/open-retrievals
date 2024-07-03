import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OcrCFG:
    pass


class OCRecognizor(object):
    def __init__(self, use_ocr='ppocr'):
        if use_ocr == 'ppocr':
            self.ocr_model = PPRecognizor()

    def recognize(self, file):
        return

    def apply_preprocess(self, data, preprocessors):
        """preprocess set"""
        for preprocessor in preprocessors:
            data = preprocessor.process(data)
        return data

    def apply_postprocess(self, data, postprocessors):
        """post-process set"""
        for postprocessor in postprocessors:
            data = postprocessor.process(data)
        return data

    def _structure(self, img):
        """structure analysis"""
        return

    def _ocr(self, img):
        """ocr function"""
        return


class PreProcessor(ABC):
    """base preprocess"""

    def __init__(self) -> None:
        pass

    def process(self, example):
        raise NotImplementedError("Subclasses should implement this method")


class PostProcessor(ABC):
    """base post-process"""

    def __init__(self) -> None:
        pass

    def process(self, example):
        raise NotImplementedError("Subclasses should implement this method")


class PPRecognizor:
    def __init__(self):
        from paddleocr import PaddleOCR, PPStructure

        self.ocr_model = PaddleOCR(use_angle_cls=False, lang="ch")
