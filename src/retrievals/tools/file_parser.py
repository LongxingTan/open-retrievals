"""
file parser for RAG
https://github.com/InternLM/HuixiangDou
"""

import hashlib
import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def process_pdf_plumber(file_path):
    import pdfplumber

    with pdfplumber.open(file_path) as pdf:
        text = ''
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                text_wo_header_footer = '\n'.join(lines[1:-1])
                text += text_wo_header_footer + '\n\n'

            if page.extract_tables():
                for table in page.extract_tables():
                    markdown_table = ''
                    for i, row in enumerate(table):
                        row = [cell for cell in row if cell]
                        processed_row = [str(cell).strip() if cell is not None else '' for cell in row]
                        markdown_row = '| ' + '| '.join(processed_row) + '|\n'
                        markdown_table += markdown_row
                        if i == 0:
                            sep = [':----' if cell.isdigit() else '----' for cell in row]
                            markdown_table += '| ' + '| '.join(sep) + '|\n'
                    text += markdown_table + '\n'
    return text


def process_pdf(file_path):
    import pypdf

    text = ""
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def process_pdf2(file_path):
    import PyPDF2

    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text().strip()
    return text


def process_url(url):
    response = requests.get(url)
    if url.endswith('.pdf'):
        return process_pdf(io.BytesIO(response.content))
    elif url.endswith('.txt'):
        return response.text
    else:
        return "Unsupported file format"


def process_txt(file_path, text_splitter):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    chunks = text_splitter.create_documents([text])
    documents = []
    for chunk in chunks:
        chunk.metadata = {'source': file_path, 'read': file_path}
        documents.append(chunk)
    return documents


def process_word(file_path):
    from docx import Document

    doc = Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text_ = paragraph.text.strip()
        if text_:
            text += text_
            text += '\n'
    return text


def process_excel(file_path: str):
    if file_path.endswith('.csv'):
        table = pd.read_csv(file_path)
    else:
        table = pd.read_excel(file_path)
    if table is None:
        return ''
    json_text = table.dropna(axis=1).to_json(force_ascii=False)
    return json_text


def process_markdown(file_path: str, head_splitter, md_splitter, text_splitter):
    text = ''
    with open(file_path, encoding='utf8') as f:
        text = f.read()

    docs = head_splitter.split_text(text)
    final = []
    for doc in docs:
        header = ''
        if len(doc.metadata) > 0:
            if 'Header 1' in doc.metadata:
                header += doc.metadata['Header 1']
            if 'Header 2' in doc.metadata:
                header += ' '
                header += doc.metadata['Header 2']
            if 'Header 3' in doc.metadata:
                header += ' '
                header += doc.metadata['Header 3']

        if len(doc.page_content) >= 1024:
            subdocs = md_splitter.create_documents([doc.page_content])
            for subdoc in subdocs:
                if len(subdoc.page_content) >= 10:
                    final.append('{} {}'.format(header, subdoc.page_content.lower()))
        elif len(doc.page_content) >= 10:
            final.append('{} {}'.format(header, doc.page_content.lower()))  # noqa E501

        return final


def process_epub(file_path: str):
    import ebooklib
    from bs4 import BeautifulSoup

    book = ebooklib.epub.read_epub(file_path)

    text_content = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text_content.append(item.get_content().decode("utf-8"))

    wt = "\n".join(text_content)
    sp = BeautifulSoup(wt, "html.parser")
    text = sp.get_text()
    return text


class FileParser:
    """File parser class RAG upload file"""

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

    def read(self, file_path: str, to_document: bool = False):
        file_type = self.get_type(file_path)
        text = ''

        try:
            if file_type == 'md' or file_type == 'text':
                with open(file_path) as f:
                    text = f.read()

            elif file_type == 'pdf':
                text += process_pdf(file_path)

            elif file_type == 'excel':
                text += process_excel(file_path)

            elif file_type == 'word' or file_type == 'ppt':
                import textract

                text = textract.process(file_path).decode('utf8')
                if file_type == 'ppt':
                    text = text.replace('\n', ' ')

            elif file_type == 'html':
                from bs4 import BeautifulSoup

                with open(file_path) as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text += soup.text

        except Exception as e:
            logger.error((file_path, str(e)))
            return '', e

        text = text.replace('\n\n\n', '\n')
        text = text.replace('\n\n', '\n')
        text = text.replace('  ', ' ')
        return text, None

    def get_type(self, file_path: str):
        for suffix in self.pdf_suffix:
            if file_path.endswith(suffix):
                return 'pdf'

        for suffix in self.md_suffix:
            if file_path.endswith(suffix):
                return 'md'

        for suffix in self.ppt_suffix:
            if file_path.endswith(suffix):
                return 'ppt'

        for suffix in self.image_suffix:
            if file_path.endswith(suffix):
                return 'image'

        for suffix in self.text_suffix:
            if file_path.endswith(suffix):
                return 'text'

        for suffix in self.word_suffix:
            if file_path.endswith(suffix):
                return 'word'

        for suffix in self.excel_suffix:
            if file_path.endswith(suffix):
                return 'excel'

        for suffix in self.html_suffix:
            if file_path.endswith(suffix):
                return 'html'
        return None

    def md5(self, file_path: str):
        hash_object = hashlib.sha256()
        with open(file_path, 'rb') as file:
            chunk_size = 8192
            while chunk := file.read(chunk_size):
                hash_object.update(chunk)

        return hash_object.hexdigest()[0:8]
