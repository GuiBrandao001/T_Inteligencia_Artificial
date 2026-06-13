"""Recuperador KNN/TF-IDF usando similaridade do cosseno."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import doc_text


class KNNRetriever:
    def __init__(self, docs: list[dict], max_features: int = 30000):
        self.docs = docs
        self.texts = [doc_text(doc) for doc in docs]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)

    def score_all(self, query: str) -> np.ndarray:
        if not self.docs:
            return np.array([], dtype=float)
        q_vec = self.vectorizer.transform([query])
        return cosine_similarity(q_vec, self.doc_matrix).ravel()

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        scores = self.score_all(query)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in order]
