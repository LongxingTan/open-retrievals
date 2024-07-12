import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OcrCFG:
    ocr_model = 'ppocr'


class OCRecognizer(object):
    """OCR class for recognize the scanned document"""

    def __init__(self, ocr_model='ppocr'):
        if ocr_model == 'ppocr':
            self.ocr_model = PPRecognizer()

    def recognize(self, file):
        logging.info('START OCR')
        start_time = time.time()

        elapsed_time = time.time() - start_time
        logging.info(f'FINISH OCR, Elapsed time: {elapsed_time:.3f}s')
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
    """Base class for OCR post-processing"""

    def __init__(self) -> None:
        pass

    def process(self, example):
        raise NotImplementedError("Subclasses should implement this method")


class PPRecognizer:
    """A class to encapsulate PaddleOCR functionality for text recognition in images."""

    def __init__(self, use_angle_cls=False, lang="ch"):
        from paddleocr import PaddleOCR, PPStructure

        self.ocr_model = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def recognize(self, img):
        try:
            result = self.ocr_model.ocr(img, cls=False)
            return result
        except Exception as e:
            print(f"An error occurred during OCR: {e}")
            return None
