"""Coleta artigos do ArXiv para o tema IA na Educacao.

Uso:
    python src/collect_arxiv.py --max-results 2000

Observacao: este script precisa de internet.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

try:
    import arxiv
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Instale as dependencias com: pip install -r requirements.txt") from exc

from config import (
    ARXIV_CATEGORIES,
    CORPUS_PATH,
    END_YEAR,
    KEYWORDS,
    MAX_RESULTS,
    START_YEAR,
)
from io_utils import write_jsonl


def build_query() -> str:
    keyword_query = " OR ".join(
        [f'ti:"{kw}" OR abs:"{kw}"' for kw in KEYWORDS]
    )
    category_query = " OR ".join([f"cat:{cat}" for cat in ARXIV_CATEGORIES])
    date_query = f"submittedDate:[{START_YEAR}01010000 TO {END_YEAR}12312359]"
    return f"({keyword_query}) AND ({category_query}) AND {date_query}"


def normalize_arxiv_id(entry_id: str) -> str:
    # Exemplo: http://arxiv.org/abs/2401.12345v2 -> 2401.12345
    arxiv_id = entry_id.rstrip("/").split("/")[-1]
    if "v" in arxiv_id:
        base, maybe_version = arxiv_id.rsplit("v", 1)
        if maybe_version.isdigit():
            return base
    return arxiv_id


def collect(max_results: int, output_path: Path) -> list[dict]:
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=build_query(),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    docs_by_id: dict[str, dict] = {}
    for result in client.results(search):
        doc_id = normalize_arxiv_id(result.entry_id)
        if doc_id in docs_by_id:
            continue
        published = result.published
        if published and not (START_YEAR <= published.year <= END_YEAR):
            continue
        docs_by_id[doc_id] = {
            "id": doc_id,
            "title": " ".join(result.title.split()),
            "abstract": " ".join(result.summary.split()),
            "authors": [author.name for author in result.authors],
            "categories": list(result.categories),
            "date": published.date().isoformat() if published else "",
            "url": result.entry_id,
            "source": "arxiv",
        }

    docs = list(docs_by_id.values())
    write_jsonl(output_path, docs)
    return docs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS)
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    args = parser.parse_args()

    print("Coletando artigos do ArXiv...")
    print(f"Tema: Uso de IA na Educacao")
    print(f"Janela temporal: {START_YEAR}-{END_YEAR}")
    print(f"Saida: {args.output}")
    docs = collect(args.max_results, args.output)
    print(f"Total salvo: {len(docs)} documentos")
    if len(docs) < 1000:
        print("AVISO: o trabalho pede entre 1.000 e 5.000 artigos. Ajuste palavras-chave/categorias se necessario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
