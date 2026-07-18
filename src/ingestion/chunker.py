from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64


class Chunker:
    def __init__(self, config: ChunkConfig):
        self._s = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents):
        chunks = self._s.split_documents(documents)
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
        return chunks
