# rag-enterprise-kit

> Production-ready RAG pipeline for enterprise.

## Stack
Python 3.11 · LangChain · ChromaDB · FastAPI

## Features
- PDF/DOCX/HTML/MD ingestion
- Hybrid search + RRF fusion
- Cohere reranking · RAGAS eval

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn src.api.app:app --reload
```
