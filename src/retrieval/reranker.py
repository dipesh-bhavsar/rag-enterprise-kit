import os
import cohere


class CohereReranker:
    def __init__(self, model="rerank-english-v3.0", top_n=4):
        self.model = model
        self.top_n = top_n
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY", ""))

    def rerank(self, query, documents):
        if not documents:
            return []
        resp = self.client.rerank(
            model=self.model,
            query=query,
            documents=[d.page_content for d in documents],
            top_n=min(self.top_n, len(documents)),
        )
        result = []
        for r in resp.results:
            doc = documents[r.index]
            doc.metadata["rerank_score"] = r.relevance_score
            result.append(doc)
        return result
