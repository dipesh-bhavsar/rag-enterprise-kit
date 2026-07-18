import pytest
from langchain_core.documents import Document
from src.ingestion.chunker import Chunker, ChunkConfig


@pytest.fixture
def ch():
    return Chunker(ChunkConfig(chunk_size=100, chunk_overlap=10))


@pytest.fixture
def docs():
    return [
        Document(page_content=" ".join(["word"] * 300), metadata={"source": "t.txt"})
    ]


def test_produces_multiple(ch, docs):
    assert len(ch.split(docs)) > 1


def test_index_added(ch, docs):
    for c in ch.split(docs):
        assert "chunk_index" in c.metadata


def test_empty(ch):
    assert ch.split([]) == []
