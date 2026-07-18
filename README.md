# 🚀 RAG Enterprise Starter Kit

An production-ready, extensible Retrieval-Augmented Generation (RAG) starter kit designed for high-performance semantic search and contextual question answering. 

This repository showcases advanced RAG patterns—including **Hybrid Search**, **Reciprocal Rank Fusion (RRF)**, **Cohere Reranking**, and **RAGAS Evaluations**—serving as a solid foundation for deploying production-grade AI search agents.

---

## 📐 Architecture Overview

The system is split into two pipelines: an asynchronous document ingestion pipeline and a low-latency query retrieval pipeline.

```
[Ingestion Pipeline]
Documents (.pdf, .docx, .md, .txt) ──> DocumentLoader ──> Chunker ──> Vector Store (FAISS/Pickle)

[Query Pipeline]
                          ┌──> Dense Search (FAISS) ───────┐
Query ──> HybridRetriever ┼                                ├──> RRF Fusion ──> Cohere Reranker ──> GPT-4o-Mini LLM ──> Answer
                          └──> Sparse Search (BM25 Okapi) ─┘
```

---

## 🌟 Key Features

- 📑 **Multi-Format Ingestion:** Built-in support for loading and parsing PDF, DOCX, HTML, Markdown, and TXT files.
- ✂️ **Structural Chunking:** Chunks text using recursive boundary separators and tags each fragment with indexing metadata.
- 🔍 **Hybrid Retrieval:** Fuses semantic similarity searches (OpenAI `text-embedding-3-small`) with keyword-based searches (BM25) using **Reciprocal Rank Fusion (RRF)**.
- ⚡ **Cohere Reranking:** Applies Cohere `rerank-english-v3.0` models to optimize document relevance and minimize LLM token distraction.
- 🤖 **Contextual Generation:** Employs OpenAI `gpt-4o-mini` with structured system instructions to produce hallucination-free, grounded answers.
- 📊 **RAGAS Evaluations:** Assess and benchmark the quality of the pipeline using standard metrics: **Faithfulness**, **Answer Relevancy**, and **Context Precision**.

---

## 💼 Real-World Usecases

This starter kit can be tailored for:
1. **Enterprise Knowledge Management:** Connect to internal wikis, policy files, and training manuals to provide employee self-service QA.
2. **Customer Support Automation:** Feed the pipeline product docs and FAQs to resolve customer questions accurately.
3. **Legal & Compliance Auditing:** Ingest contracts, regulations, and reports to quickly find provisions and cross-reference compliance rules.

---

## 🛠️ Stack & Technologies

* **Backend Framework:** FastAPI & Uvicorn (async HTTP service)
* **Orchestration:** LangChain Ecosystem (`langchain-core`, `langchain-community`, `langchain-openai`)
* **Vector Index:** FAISS (Facebook AI Similarity Search)
* **Reranking API:** Cohere Rerank API
* **LLM Engine:** OpenAI Chat Completions API
* **Evaluations:** Ragas & Pandas

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11 to 3.13
- OpenAI API Key (and optionally Cohere API Key)

### 2. Installation
Clone the repository and install the dependencies in your virtual environment:

```bash
# In Git Bash or Command Prompt:
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```env
OPENAI_API_KEY=your-openai-api-key
COHERE_API_KEY=your-cohere-api-key
```

### 4. Run the Server
Start the FastAPI application:

```bash
# Using Git Bash:
python -m uvicorn src.api.app:app --reload

# Using Windows Command Prompt / PowerShell:
.venv\Scripts\python -m uvicorn src.api.app:app --reload
```

The server will start at **`http://localhost:8000`**.

---

## 🔌 API Documentation & Usage

FastAPI generates automatic interactive docs. Once the server is running, visit **[http://localhost:8000/docs](http://localhost:8000/docs)** to test the endpoints in your browser.

### 📥 1. Document Ingestion (`POST /api/v1/ingest`)
Send a path to a file or directory of documents to ingest.

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "Content-Type: application/json" \
     -d '{"source": "docs/architecture.md"}'
```

### 💬 2. Contextual Question Answering (`POST /api/v1/query`)
Ask questions based on your ingested knowledge base.

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain the query pipeline structure."}'
```

---

## 📈 Running Evaluations

RAGAS metrics help you continuously measure and improve the reliability of the system.

To evaluate predictions from a JSON dataset:
```bash
python -m src.evaluation.evaluator --dataset path/to/dataset.json --output results.csv
```

### Target Benchmarks:
| Faithfulness | Answer Relevancy | Context Precision |
| :---: | :---: | :---: |
| **0.91** | **0.88** | **0.84** |

---

## 🛠️ How to Extend This Starter Kit

This starter kit is designed to be highly modular. Here is how you can expand it for production workloads:

* **Production Vector Database:** Replace the local FAISS-backed store in `chroma_store.py` with full integrations (e.g., Pgvector, Pinecone, Qdrant, or Chroma).
* **Multi-Modal Data Ingestion:** Add Unstructured loaders, OCR (via Tesseract or layout models), or Whisper transcripts to support images, scans, and audio files.
* **Semantic Cache:** Add a Redis-backed semantic caching layer to bypass LLM calls for repeating/similar user queries.
* **Conversational History (Memory):** Extend the `/api/v1/query` endpoint with DynamoDB or PostgreSQL Chat Message History to handle multi-turn dialogue.
* **Guardrails:** Add a layer like NeMo Guardrails or Llama Guard to filter toxic user requests or ensure the LLM's response stays on topic.
