import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from src.config.settings import get_settings
from src.vectorstore.chroma_store import ChromaStore
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import Chunker, ChunkConfig
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CohereReranker
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

log = structlog.get_logger()

vector_store = None
loader = None
chunker = None
retriever = None
reranker = None
llm = None


@asynccontextmanager
async def lifespan(app):
    global vector_store, loader, chunker, retriever, reranker, llm
    log.info("startup")

    settings = get_settings()
    config = settings.load_yaml()

    vector_store = ChromaStore(
        collection_name=config["vector_store"]["collection_name"],
        persist_directory=config["vector_store"]["persist_directory"],
        embedding_model=config["embeddings"]["model"],
        openai_api_key=settings.openai_api_key,
    )

    loader = DocumentLoader(supported_formats=config["ingestion"]["supported_formats"])
    chunker = Chunker(
        ChunkConfig(
            chunk_size=config["ingestion"]["chunk_size"],
            chunk_overlap=config["ingestion"]["chunk_overlap"],
        )
    )

    retriever = HybridRetriever(
        store=vector_store,
        top_k=config["retrieval"]["top_k"],
        rrf_k=config["retrieval"]["rrf_k"],
    )

    if vector_store.documents:
        retriever.index_corpus(vector_store.documents)
        log.info(
            "Indexed existing corpus on startup", doc_count=len(vector_store.documents)
        )

    cohere_key = os.getenv("COHERE_API_KEY") or settings.cohere_api_key
    if config["reranker"]["enabled"] and cohere_key:
        reranker = CohereReranker(top_n=config["reranker"]["top_n"])
        log.info("Reranker enabled using Cohere")
    else:
        log.warning(
            "Cohere API key not found or reranker disabled. Reranking will be skipped."
        )
        reranker = None

    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        openai_api_key=settings.openai_api_key,
    )

    yield
    log.info("shutdown")


def create_app():
    app = FastAPI(title="RAG Enterprise Kit", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/v1/ingest")
    def ingest(p: dict):
        source = p.get("source")
        if not source:
            return {"error": "Source path is required"}

        path = Path(source)
        if not path.exists():
            return {"error": f"Source path does not exist: {source}"}

        if path.is_dir():
            docs = loader.load_directory(path)
        else:
            docs = loader.load(path)

        if not docs:
            return {"message": "No documents found to ingest", "source": source}

        chunks = chunker.split(docs)
        num_added = vector_store.add_documents(chunks)

        retriever.index_corpus(vector_store.documents)

        return {
            "message": f"Successfully ingested {len(docs)} documents ({num_added} chunks)",
            "source": source,
            "chunks_count": num_added,
        }

    @app.post("/api/v1/query")
    def query(p: dict):
        query_str = p.get("query")
        if not query_str:
            return {"error": "Query string is required"}

        docs = retriever.retrieve(query_str)

        if reranker and docs:
            docs = reranker.rerank(query_str, docs)

        context = "\n\n".join([d.page_content for d in docs])
        messages = [
            SystemMessage(
                content="You are a helpful enterprise assistant. Answer the user's question using only the provided context. If you don't know the answer or the context doesn't contain it, say you don't know."
            ),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query_str}"),
        ]

        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "query": query_str,
            "contexts": [d.page_content for d in docs],
        }

    return app


app = create_app()
