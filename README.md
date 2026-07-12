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


## Evaluation

```bash
python -m src.evaluation.evaluator --dataset data/eval.json
```

| Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|
| 0.91 | 0.88 | 0.84 |
