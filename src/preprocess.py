"""Pre-processamento textual simples para recuperacao de informacao."""
from __future__ import annotations

import re
from typing import Iterable

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    def unidecode(text: str) -> str:
        return text

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DOMAIN_STOPWORDS = {
    "paper", "study", "studies", "result", "results", "method", "methods",
    "approach", "approaches", "model", "models", "data", "using", "based",
}

STOPWORDS = set(ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS)
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]+")


def normalize_text(text: str) -> str:
    text = unidecode(text or "")
    text = text.lower()
    text = text.replace("/", " ")
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    text = normalize_text(text)
    tokens = TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    return tokens


def doc_text(doc: dict) -> str:
    return f"{doc.get('title', '')} {doc.get('abstract', '')}".strip()


def preprocess_for_bm25(docs: Iterable[dict]) -> list[list[str]]:
    return [tokenize(doc_text(doc)) for doc in docs]


def preprocess_query(query: str) -> list[str]:
    return tokenize(query)
