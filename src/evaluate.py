"""Avaliacao simples de runs TREC com P@k, R@k, AP, MAP e nDCG@k."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import log2
from pathlib import Path

from config import (
    BM25_RUN_PATH,
    DEFAULT_TOP_K,
    HYBRID_RUN_PATH,
    KNN_RUN_PATH,
    QRELS_PATH,
    RESULTS_CSV_PATH,
)


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            qid, _iter, doc_id, rel = line.split("\t")
            qrels[qid][doc_id] = int(rel)
    return qrels


def load_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    run: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _q0, doc_id, rank, score, _system = line.split()
            run[qid].append((doc_id, float(score)))
    return run


def precision_at_k(ranked_doc_ids: list[str], rels: dict[str, int], k: int) -> float:
    top = ranked_doc_ids[:k]
    if not top:
        return 0.0
    return sum(1 for doc_id in top if rels.get(doc_id, 0) > 0) / k


def recall_at_k(ranked_doc_ids: list[str], rels: dict[str, int], k: int) -> float:
    total_relevant = sum(1 for rel in rels.values() if rel > 0)
    if total_relevant == 0:
        return 0.0
    found = sum(1 for doc_id in ranked_doc_ids[:k] if rels.get(doc_id, 0) > 0)
    return found / total_relevant


def average_precision(ranked_doc_ids: list[str], rels: dict[str, int]) -> float:
    total_relevant = sum(1 for rel in rels.values() if rel > 0)
    if total_relevant == 0:
        return 0.0
    precisions = []
    found = 0
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if rels.get(doc_id, 0) > 0:
            found += 1
            precisions.append(found / i)
    return sum(precisions) / total_relevant if precisions else 0.0


def dcg_at_k(ranked_doc_ids: list[str], rels: dict[str, int], k: int) -> float:
    total = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = rels.get(doc_id, 0)
        total += rel / log2(i + 1)
    return total


def ndcg_at_k(ranked_doc_ids: list[str], rels: dict[str, int], k: int) -> float:
    dcg = dcg_at_k(ranked_doc_ids, rels, k)
    ideal_rels = sorted(rels.values(), reverse=True)
    ideal_ids = [f"ideal-{i}" for i in range(len(ideal_rels))]
    ideal_map = dict(zip(ideal_ids, ideal_rels))
    idcg = dcg_at_k(ideal_ids, ideal_map, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(run_path: Path, qrels: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    run = load_run(run_path)
    p_values = []
    r_values = []
    ap_values = []
    ndcg_values = []

    for qid, rels in qrels.items():
        ranked_doc_ids = [doc_id for doc_id, _score in run.get(qid, [])]
        p_values.append(precision_at_k(ranked_doc_ids, rels, k))
        r_values.append(recall_at_k(ranked_doc_ids, rels, k))
        ap_values.append(average_precision(ranked_doc_ids, rels))
        ndcg_values.append(ndcg_at_k(ranked_doc_ids, rels, k))

    n = max(len(qrels), 1)
    return {
        "system": run_path.stem,
        f"P@{k}": sum(p_values) / n,
        f"R@{k}": sum(r_values) / n,
        "MAP": sum(ap_values) / n,
        f"nDCG@{k}": sum(ndcg_values) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, default=QRELS_PATH)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    run_paths = [BM25_RUN_PATH, KNN_RUN_PATH, HYBRID_RUN_PATH]
    rows = [evaluate_run(path, qrels, args.k) for path in run_paths]

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Resultados:")
    for row in rows:
        print(row)
    print(f"CSV salvo em: {RESULTS_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
