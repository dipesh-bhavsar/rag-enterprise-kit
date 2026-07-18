from pathlib import Path
import faiss
import pickle

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class ChromaStore:
    """
    Temporary implementation backed by FAISS.
    Class name is kept for compatibility with the rest of the project.
    """

    def __init__(
        self,
        collection_name,
        persist_directory,
        embedding_model="text-embedding-3-small",
        openai_api_key=""
    ):
        if not openai_api_key:
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY")

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.index_file = self.persist_directory / f"{collection_name}.faiss"
        self.docs_file = self.persist_directory / f"{collection_name}.pkl"

        self.documents = []

        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))

            if self.docs_file.exists():
                with open(self.docs_file, "rb") as f:
                    self.documents = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(1536)

    def add_documents(self, documents):
        if not documents:
            return 0

        texts = [d.page_content for d in documents]

        embeddings = self.embeddings.embed_documents(texts)

        import numpy as np

        vectors = np.array(embeddings, dtype="float32")

        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        self.documents.extend(documents)

        faiss.write_index(self.index, str(self.index_file))

        with open(self.docs_file, "wb") as f:
            pickle.dump(self.documents, f)

        return len(documents)

    def search(self, query, top_k=10):
        if len(self.documents) == 0:
            return []

        import numpy as np

        query_vector = np.array(
            [self.embeddings.embed_query(query)],
            dtype="float32"
        )

        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.documents))
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]

            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "score": float(score)
                    }
                )
            )

        return results

    def count(self):
        return len(self.documents)