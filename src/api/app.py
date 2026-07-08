from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
log = structlog.get_logger()
@asynccontextmanager
async def lifespan(app):
    log.info("startup"); yield; log.info("shutdown")
def create_app():
    app = FastAPI(title="RAG Enterprise Kit", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    @app.get("/health")
    def health(): return {"status":"ok"}
    @app.post("/api/v1/ingest")
    def ingest(p: dict): return {"message":"Ingestion queued","source":p.get("source")}
    @app.post("/api/v1/query")
    def query(p: dict): return {"answer":"Wire up HybridRetriever+LLM here","query":p.get("query")}
    return app
app = create_app()
