"""ReasoningBank-style top-k historical-query retrieval, without a mandatory Vertex client."""
import math
import re
from collections import Counter
from .common import read_json, write_json


class MemoryBank:
    def __init__(self, path, embedding_model=None):
        self.path = path
        self.records = read_json(path) if path.exists() else []
        self.embedding_model = embedding_model

    def retrieve(self, query, k=1):
        if not self.records:
            return []
        texts = [query] + [r["query"] for r in self.records]
        if self.embedding_model:
            import litellm
            vectors = [r["embedding"] for r in litellm.embedding(model=self.embedding_model, input=texts).data]
            def score(v):
                return sum(a*b for a, b in zip(vectors[0], v)) / max(1e-12, math.sqrt(sum(a*a for a in vectors[0]) * sum(b*b for b in v)))
            scores = [score(v) for v in vectors[1:]]
        else:
            # Explicit lightweight baseline, not the paper's semantic retrieval.
            vectors = [Counter(re.findall(r"\w+", t.lower())) for t in texts]
            def score(v):
                return sum(n*v[t] for t, n in vectors[0].items()) / max(1e-12, math.sqrt(sum(n*n for n in vectors[0].values()) * sum(n*n for n in v.values())))
            scores = [score(v) for v in vectors[1:]]
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.records[i] for i in indices]

    def add(self, task_id, query, items):
        if items:
            self.records.append({"task_id": task_id, "query": query, "memory_items": items})
            write_json(self.path, self.records)
