"""
refer and rights reserved by:
https://github.com/InternLM/HuixiangDou
"""

import hashlib
import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class FileParser:
    def __init__(self):
        self.image_suffix = ['.jpg', '.jpeg', '.png', '.bmp']
        self.md_suffix = ['.md']
        self.text_suffix = ['.txt', '.text']
        self.excel_suffix = ['.xlsx', '.xls', '.csv']
        self.pdf_suffix = ['.pdf']
        self.ppt_suffix = ['.pptx']
        self.html_suffix = ['.html', '.htm', '.shtml', '.xhtml']
        self.word_suffix = ['.docx', '.doc']
        self.normal_suffix = (
            self.md_suffix
            + self.text_suffix
            + self.excel_suffix
            + self.pdf_suffix
            + self.word_suffix
            + self.ppt_suffix
            + self.html_suffix
        )

    def read(self, filepath: str, to_document: bool = False):
        file_type = self.get_type(filepath)
        text = ''

        try:
            if file_type == 'md' or file_type == 'text':
                with open(filepath) as f:
                    text = f.read()

            elif file_type == 'pdf':
                text += self.read_pdf(filepath)

            elif file_type == 'excel':
                text += self.read_excel(filepath)

            elif file_type == 'word' or file_type == 'ppt':
                import textract

                text = textract.process(filepath).decode('utf8')
                if file_type == 'ppt':
                    text = text.replace('\n', ' ')

            elif file_type == 'html':
                from bs4 import BeautifulSoup

                with open(filepath) as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text += soup.text

        except Exception as e:
            logger.error((filepath, str(e)))
            return '', e

        text = text.replace('\n\n\n', '\n')
        text = text.replace('\n\n', '\n')
        text = text.replace('  ', ' ')
        return text, None

    def get_type(self, filepath: str):
        for suffix in self.pdf_suffix:
            if filepath.endswith(suffix):
                return 'pdf'

        for suffix in self.md_suffix:
            if filepath.endswith(suffix):
                return 'md'

        for suffix in self.ppt_suffix:
            if filepath.endswith(suffix):
                return 'ppt'

        for suffix in self.image_suffix:
            if filepath.endswith(suffix):
                return 'image'

        for suffix in self.text_suffix:
            if filepath.endswith(suffix):
                return 'text'

        for suffix in self.word_suffix:
            if filepath.endswith(suffix):
                return 'word'

        for suffix in self.excel_suffix:
            if filepath.endswith(suffix):
                return 'excel'

        for suffix in self.html_suffix:
            if filepath.endswith(suffix):
                return 'html'
        return None

    def md5(self, filepath: str):
        hash_object = hashlib.sha256()
        with open(filepath, 'rb') as file:
            chunk_size = 8192
            while chunk := file.read(chunk_size):
                hash_object.update(chunk)

        return hash_object.hexdigest()[0:8]

    def read_pdf(self, filename):
        with open(filename, 'rb') as file:
            text = self.extract_text_from_pdf(file)
        return text

    @staticmethod
    def read_txt(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()

    def read_url(self, url):
        response = requests.get(url)
        if url.endswith('.pdf'):
            return self.extract_text_from_pdf(io.BytesIO(response.content))
        elif url.endswith('.txt'):
            return response.text
        else:
            return "Unsupported file format"

    def read_excel(self, filepath: str):
        table = None
        if filepath.endswith('.csv'):
            table = pd.read_csv(filepath)
        else:
            table = pd.read_excel(filepath)
        if table is None:
            return ''
        json_text = table.dropna(axis=1).to_json(force_ascii=False)
        return json_text

    def read_epub(self, filepath: str):
        import ebooklib
        from bs4 import BeautifulSoup

        book = ebooklib.epub.read_epub(filepath)

        text_content = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                text_content.append(item.get_content().decode("utf-8"))

        wt = "\n".join(text_content)
        sp = BeautifulSoup(wt, "html.parser")
        text = sp.get_text()
        return text

    @staticmethod
    def extract_text_from_pdf(file_stream):
        import pypdf

        text = ""
        reader = pypdf.PdfReader(file_stream)
        for page in reader.pages:
            text += page.extract_text()
        return text
