"""Recuperador BM25 sem dependencia externa.

A implementacao segue a formula classica do BM25 Okapi:
score(d, q) = sum IDF(t) * (tf(t,d)*(k1+1))/(tf(t,d)+k1*(1-b+b*|d|/avgdl))
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np

from preprocess import preprocess_for_bm25, preprocess_query


class BM25Retriever:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.tokenized_docs = preprocess_for_bm25(docs)
        self.doc_lens = np.array([len(tokens) for tokens in self.tokenized_docs], dtype=float)
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.tokenized_docs]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n_docs = len(self.tokenized_docs)
        df: dict[str, int] = defaultdict(int)
        for tokens in self.tokenized_docs:
            for term in set(tokens):
                df[term] += 1
        # IDF com suavizacao. Evita valores negativos para termos muito frequentes.
        return {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score_all(self, query: str) -> np.ndarray:
        query_terms = preprocess_query(query)
        scores = np.zeros(len(self.docs), dtype=float)
        if not query_terms or not self.docs:
            return scores

        for i, tf_counter in enumerate(self.term_freqs):
            dl = self.doc_lens[i]
            denom_length = self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
            score = 0.0
            for term in query_terms:
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * (tf * (self.k1 + 1.0)) / (tf + denom_length)
            scores[i] = score
        return scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        scores = self.score_all(query)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in order]
