from collections import defaultdict
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, store, top_k=10, rrf_k=60):
        self.store = store
        self.top_k = top_k
        self.rrf_k = rrf_k
        self._corpus = []
        self._bm25 = None

    def index_corpus(self, docs):
        self._corpus = docs
        if docs:
            self._bm25 = BM25Okapi([d.page_content.lower().split() for d in docs])
        else:
            self._bm25 = None

    def retrieve(self, query):
        if not self._bm25:
            return self.store.search(query, top_k=self.top_k)
        dense = self.store.search(query, top_k=self.top_k * 2)
        scores = self._bm25.get_scores(query.lower().split())
        sparse = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[
            : self.top_k * 2
        ]
        rrf = defaultdict(float)
        dm = {}
        for rank, doc in enumerate(dense):
            k = doc.page_content[:80]
            rrf[k] += 1 / (self.rrf_k + rank + 1)
            dm[k] = doc
        for rank, (idx, _) in enumerate(sparse):
            doc = self._corpus[idx]
            k = doc.page_content[:80]
            rrf[k] += 1 / (self.rrf_k + rank + 1)
            dm.setdefault(k, doc)
        seen = set()
        result = []
        for k in sorted(rrf, key=rrf.__getitem__, reverse=True)[: self.top_k]:
            h = hash(dm[k].page_content)
            if h not in seen:
                seen.add(h)
                result.append(dm[k])
        return result
