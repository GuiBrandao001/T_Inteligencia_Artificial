"""Configuracoes principais do projeto."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

THEME = "Uso de Inteligencia Artificial na Educacao"
START_YEAR = 2018
END_YEAR = 2026
MAX_RESULTS = 2000

# Categorias relacionadas ao tema. Ajuste se a coleta vier muito ampla ou muito pequena.
ARXIV_CATEGORIES = [
    "cs.AI",  # Artificial Intelligence
    "cs.LG",  # Machine Learning
    "cs.CL",  # Computation and Language
    "cs.HC",  # Human-Computer Interaction
    "stat.ML",
]

KEYWORDS = [
    "artificial intelligence in education",
    "AI in education",
    "intelligent tutoring systems",
    "learning analytics",
    "educational data mining",
    "adaptive learning",
    "personalized learning",
    "automated feedback",
    "generative AI in education",
    "ChatGPT in education",
    "large language models in education",
    "student performance prediction",
    "AI tutoring",
    "educational technology",
]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"
EVAL_DIR = PROJECT_ROOT / "eval"

CORPUS_PATH = DATA_PROCESSED_DIR / "corpus.jsonl"
QUERIES_PATH = EVAL_DIR / "queries.tsv"
QRELS_PATH = EVAL_DIR / "qrels.tsv"

BM25_RUN_PATH = RUNS_DIR / "bm25.trec"
KNN_RUN_PATH = RUNS_DIR / "knn.trec"
HYBRID_RUN_PATH = RUNS_DIR / "hybrid.trec"
RESULTS_CSV_PATH = RUNS_DIR / "evaluation_results.csv"

DEFAULT_TOP_K = 10
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_ALPHA = 0.5  # 0.5 = mesmo peso para BM25 e KNN/TF-IDF
