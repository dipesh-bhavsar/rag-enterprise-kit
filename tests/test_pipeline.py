import os
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

# Set mock environment keys for test imports
os.environ["OPENAI_API_KEY"] = "mock-openai-key"
os.environ["COHERE_API_KEY"] = "mock-cohere-key"

from src.api.app import app


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@patch("src.api.app.ChatOpenAI")
@patch("src.api.app.ChromaStore")
@patch("src.api.app.CohereReranker")
def test_end_to_end_pipeline(mock_reranker_cls, mock_store_cls, mock_llm_cls, temp_dir):
    # Setup mock stores and response behaviors
    mock_store = MagicMock()
    stored_docs = []

    def mock_add_documents(documents):
        stored_docs.extend(documents)
        return len(documents)

    mock_store.add_documents.side_effect = mock_add_documents
    type(mock_store).documents = property(lambda self: stored_docs)

    doc1 = Document(page_content="Mock doc 1 content", metadata={"source": "test.txt"})
    doc2 = Document(page_content="Mock doc 2 content", metadata={"source": "test.txt"})
    mock_store.search.return_value = [doc1, doc2]
    mock_store_cls.return_value = mock_store

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, docs: docs
    mock_reranker_cls.return_value = mock_reranker

    mock_llm = MagicMock()
    mock_llm_response = MagicMock()
    mock_llm_response.content = "This is a mock answer based on context."
    mock_llm.invoke.return_value = mock_llm_response
    mock_llm_cls.return_value = mock_llm

    with TestClient(app) as client:
        # Create a temporary file to ingest
        test_file = temp_dir / "test.txt"
        test_file.write_text(
            "Mock doc 1 content\n\nMock doc 2 content", encoding="utf-8"
        )

        # 1. Ingest test
        response = client.post("/api/v1/ingest", json={"source": str(test_file)})
        assert response.status_code == 200
        data = response.json()
        assert "Successfully ingested" in data["message"]
        assert data["chunks_count"] == 1

        # 2. Query test
        response = client.post("/api/v1/query", json={"query": "test query"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a mock answer based on context."
        assert data["query"] == "test query"
        assert len(data["contexts"]) == 3
