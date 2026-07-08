import logging
from pathlib import Path
from langchain_community.document_loaders import (PyPDFLoader, Docx2txtLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader, TextLoader)
logger = logging.getLogger(__name__)
LOADER_MAP = {".pdf":PyPDFLoader,".docx":Docx2txtLoader,".html":UnstructuredHTMLLoader,".htm":UnstructuredHTMLLoader,".md":UnstructuredMarkdownLoader,".txt":TextLoader}
class DocumentLoader:
    def __init__(self, supported_formats):
        self.supported = {f".{f.lstrip('.')}" for f in supported_formats}
    def load(self, source):
        path = Path(source); suffix = path.suffix.lower()
        if suffix not in self.supported: raise ValueError(f"Unsupported: {suffix}")
        docs = LOADER_MAP[suffix](str(path)).load()
        for doc in docs: doc.metadata.update({"source":str(path),"file_name":path.name})
        return docs
    def load_directory(self, directory):
        all_docs = []
        for path in sorted(Path(directory).rglob("*")):
            if path.suffix.lower() in self.supported and path.is_file():
                try: all_docs.extend(self.load(path))
                except Exception as e: logger.warning(f"Skipping {path.name}: {e}")
        return all_docs
