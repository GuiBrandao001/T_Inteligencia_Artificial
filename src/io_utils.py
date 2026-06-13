"""Funcoes auxiliares para leitura e escrita de arquivos do projeto."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries(path: Path) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Linha invalida em queries.tsv: {line}")
            queries.append((parts[0], parts[1]))
    return queries


def write_trec_run(path: Path, run_name: str, rows: list[tuple[str, str, int, float]]) -> None:
    """Escreve run TREC: qid Q0 doc_id rank score system."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for qid, doc_id, rank, score in rows:
            f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
