from pathlib import Path
import chromadb
from chromadb.config import Settings as CS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
class ChromaStore:
    def __init__(self, collection_name, persist_directory, embedding_model="text-embedding-3-small", openai_api_key=""):
        self.embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory, settings=CS(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space":"cosine"})
    def add_documents(self, documents):
        if not documents: return 0
        ids=[f"{d.metadata.get('source','d')}_{d.metadata.get('chunk_index',i)}" for i,d in enumerate(documents)]
        texts=[d.page_content for d in documents]
        self.collection.upsert(ids=ids, embeddings=self.embeddings.embed_documents(texts), documents=texts, metadatas=[d.metadata for d in documents])
        return len(texts)
    def search(self, query, top_k=10):
        r=self.collection.query(query_embeddings=[self.embeddings.embed_query(query)], n_results=min(top_k,self.collection.count()), include=["documents","metadatas","distances"])
        return [Document(page_content=t,metadata={**m,"score":1-d}) for t,m,d in zip(r["documents"][0],r["metadatas"][0],r["distances"][0])]
    def count(self): return self.collection.count()
