"""Gera os rankings TREC de BM25, KNN/TF-IDF e hibrido."""
from __future__ import annotations

import argparse
from pathlib import Path

from bm25_retriever import BM25Retriever
from config import (
    BM25_B,
    BM25_K1,
    BM25_RUN_PATH,
    CORPUS_PATH,
    DEFAULT_TOP_K,
    HYBRID_ALPHA,
    HYBRID_RUN_PATH,
    KNN_RUN_PATH,
    QUERIES_PATH,
)
from hybrid_retriever import HybridRetriever
from io_utils import load_queries, read_jsonl, write_trec_run
from knn_retriever import KNNRetriever


def build_run_rows(queries, retriever, top_k: int) -> list[tuple[str, str, int, float]]:
    rows: list[tuple[str, str, int, float]] = []
    for qid, query in queries:
        results = retriever.search(query, top_k=top_k)
        for rank, (doc, score) in enumerate(results, start=1):
            rows.append((qid, doc["id"], rank, score))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    docs = read_jsonl(args.corpus)
    queries = load_queries(args.queries)
    print(f"Documentos: {len(docs)}")
    print(f"Queries: {len(queries)}")

    bm25 = BM25Retriever(docs, k1=BM25_K1, b=BM25_B)
    knn = KNNRetriever(docs)
    hybrid = HybridRetriever(bm25, knn, alpha=HYBRID_ALPHA)

    bm25_rows = build_run_rows(queries, bm25, args.top_k)
    knn_rows = build_run_rows(queries, knn, args.top_k)
    hybrid_rows = build_run_rows(queries, hybrid, args.top_k)

    write_trec_run(BM25_RUN_PATH, "bm25", bm25_rows)
    write_trec_run(KNN_RUN_PATH, "knn_tfidf", knn_rows)
    write_trec_run(HYBRID_RUN_PATH, "hybrid", hybrid_rows)

    print(f"Run BM25 salvo em: {BM25_RUN_PATH}")
    print(f"Run KNN salvo em: {KNN_RUN_PATH}")
    print(f"Run hibrido salvo em: {HYBRID_RUN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
