from ..tools.file_parser import FileParser


class ModelCenter(object):
    def __init__(self):
        pass

    def chat(self, query: str, chat_history):
        return


class KnowledgeCenter(object):
    def __init__(self):
        self.parser = FileParser()

    def init_vector_db(self):
        pass

    def add_document(self, file_path: str):
        doc = self.parser.read(file_path)
        print(doc)
