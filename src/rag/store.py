import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.schemas import RetrievedChunk


class SimpleRAG:
    def __init__(self, corpus: str, documents: list[dict]):
        self.corpus = corpus
        self.documents = documents
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
        texts = [f"{item['title']} {item['text']}" for item in documents]
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        if self.matrix is None or not query.strip():
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        indices = scores.argsort()[::-1][:k]
        return [RetrievedChunk(
            corpus=self.corpus, document_id=str(self.documents[i]["id"]),
            title=self.documents[i]["title"], text=self.documents[i]["text"],
            source=self.documents[i]["source"], score=round(float(scores[i]), 4),
        ) for i in indices if scores[i] > 0]


class MultiRAG:
    CORPORA = ("clinical", "disease", "patient_memory", "decision")

    def __init__(self, stores: dict[str, SimpleRAG]):
        self.stores = stores

    @classmethod
    def from_directory(cls, directory: Path) -> "MultiRAG":
        stores = {}
        for name in cls.CORPORA:
            path = directory / f"{name}.json"
            documents = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            stores[name] = SimpleRAG(name, documents)
        return cls(stores)

    def retrieve(self, query: str, k: int = 3, patient_documents: list[dict] | None = None) -> list[RetrievedChunk]:
        chunks = []
        for name, store in self.stores.items():
            if name == "patient_memory" and patient_documents:
                chunks.extend(SimpleRAG("patient_memory", patient_documents).search(query, k))
            else:
                chunks.extend(store.search(query, k))
        return sorted(chunks, key=lambda item: item.score, reverse=True)
