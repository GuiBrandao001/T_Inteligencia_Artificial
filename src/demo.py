"""Demonstra uma consulta nos tres recuperadores."""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

from bm25_retriever import BM25Retriever
from config import BM25_B, BM25_K1, CORPUS_PATH, DEFAULT_TOP_K, HYBRID_ALPHA
from hybrid_retriever import HybridRetriever
from io_utils import read_jsonl
from knn_retriever import KNNRetriever


def print_results(title: str, results: list[tuple[dict, float]]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for i, (doc, score) in enumerate(results, start=1):
        print(f"{i:02d}. {doc.get('title', 'Sem titulo')}")
        print(f"    id: {doc.get('id')} | data: {doc.get('date', '')} | score: {score:.4f}")
        abstract = doc.get("abstract", "")
        print("    " + textwrap.shorten(abstract, width=220, placeholder="..."))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="generative AI in education")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    docs = read_jsonl(args.corpus)
    bm25 = BM25Retriever(docs, k1=BM25_K1, b=BM25_B)
    knn = KNNRetriever(docs)
    hybrid = HybridRetriever(bm25, knn, alpha=HYBRID_ALPHA)

    print(f"Consulta: {args.query}")
    print_results("BM25", bm25.search(args.query, args.top_k))
    print_results("KNN com TF-IDF", knn.search(args.query, args.top_k))
    print_results("Hibrido BM25 + KNN", hybrid.search(args.query, args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
