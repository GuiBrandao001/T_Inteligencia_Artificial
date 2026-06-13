"""Ranking hibrido BM25 + KNN/TF-IDF."""
from __future__ import annotations

import numpy as np

from bm25_retriever import BM25Retriever
from knn_retriever import KNNRetriever


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))
    if max_s == min_s:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_s) / (max_s - min_s)


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, knn: KNNRetriever, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha deve estar entre 0 e 1")
        self.bm25 = bm25
        self.knn = knn
        self.docs = bm25.docs
        self.alpha = alpha

    def score_all(self, query: str) -> np.ndarray:
        bm25_scores = minmax_normalize(self.bm25.score_all(query))
        knn_scores = minmax_normalize(self.knn.score_all(query))
        return self.alpha * bm25_scores + (1.0 - self.alpha) * knn_scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        scores = self.score_all(query)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in order]
