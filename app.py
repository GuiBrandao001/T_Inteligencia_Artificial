"""Interface grafica do projeto.

Execute com:
    python -m streamlit run app.py

A interface foi pensada como um buscador cientifico, nao como chatbot.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bm25_retriever import BM25Retriever
from config import (
    BM25_B,
    BM25_K1,
    CORPUS_PATH,
    DEFAULT_TOP_K,
    HYBRID_ALPHA,
    RESULTS_CSV_PATH,
    THEME,
)
from hybrid_retriever import HybridRetriever
from io_utils import read_jsonl
from knn_retriever import KNNRetriever


st.set_page_config(
    page_title="Buscador Cientifico - IA na Educacao",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --accent: #009eb8;
        --accent-dark: #007f94;
        --accent-deep: #005f70;
        --accent-light: #e6f8fb;
        --accent-soft: #f4fbfc;
        --border-soft: #d8eef2;
    }

    /* Forca a cor principal dos elementos HTML nativos */
    * {
        accent-color: var(--accent) !important;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #0f172a;
    }
    .subtitle {
        color: #54606f;
        font-size: 1.02rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f7f9fc;
        border: 1px solid #e6eaf0;
        border-radius: 14px;
        padding: 1rem;
    }
    .result-card {
        border: 1px solid var(--border-soft);
        border-left: 5px solid var(--accent);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
        background: white;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .rank-badge {
        display: inline-block;
        background: var(--accent);
        color: white;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.45rem;
    }
    .score-badge {
        display: inline-block;
        background: var(--accent-light);
        color: var(--accent-dark);
        padding: 0.16rem 0.5rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 650;
        margin-left: 0.35rem;
    }
    .small-muted {
        color: #667085;
        font-size: 0.88rem;
    }
    .doc-title {
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }
    .search-help {
        color: #667085;
        font-size: 0.9rem;
        margin-top: -0.25rem;
        margin-bottom: 0.6rem;
    }
    .example-title {
        font-size: 0.95rem;
        font-weight: 650;
        color: #344054;
        margin-bottom: 0.35rem;
    }

    /* Botoes, inclusive Buscar e download */
    div.stButton > button,
    div.stDownloadButton > button,
    div[data-testid="stLinkButton"] a {
        border-radius: 10px !important;
        border-color: var(--accent) !important;
    }
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        color: white !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background-color: var(--accent-dark) !important;
        border-color: var(--accent-dark) !important;
        color: white !important;
    }
    div.stButton > button:not([kind="primary"]),
    div.stDownloadButton > button:not([kind="primary"]),
    div[data-testid="stLinkButton"] a {
        border-color: var(--accent) !important;
        color: var(--accent-dark) !important;
        background: white !important;
    }
    div.stButton > button:not([kind="primary"]):hover,
    div.stDownloadButton > button:not([kind="primary"]):hover,
    div[data-testid="stLinkButton"] a:hover {
        border-color: var(--accent-dark) !important;
        color: var(--accent-dark) !important;
        background: var(--accent-soft) !important;
    }

    /* Campo de consulta, select e foco */
    .stTextInput input:focus,
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 158, 184, 0.15) !important;
    }
    .stSelectbox label,
    .stSlider label,
    .stTextInput label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Slider: quantidade de resultados */
    div[data-testid="stSlider"] {
        color: var(--accent-dark) !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 0.18rem rgba(0, 158, 184, 0.16) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] div {
        border-color: var(--accent) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
    div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="#ff4b4b"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        color: var(--accent-dark) !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBar"],
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: var(--accent-dark) !important;
    }

    /* Abas: Buscar artigos, Corpus, Avaliacao, Sobre */
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        color: var(--accent-dark) !important;
        border-bottom-color: transparent !important;
        font-weight: 700 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {
        color: var(--accent-dark) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {
        color: var(--accent) !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    button[data-baseweb="tab"] {
        color: var(--accent-dark) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
    }

    /* Dataframes, links, expanders e metricas */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--accent-dark) !important;
        font-weight: 700 !important;
    }
    details summary {
        color: var(--accent-dark) !important;
        font-weight: 700 !important;
    }
    a, a:visited {
        color: var(--accent-dark) !important;
    }
    div[data-testid="stToolbar"] svg,
    div[data-testid="stBaseButton-secondary"] svg,
    div[data-testid="stBaseButton-primary"] svg {
        color: var(--accent) !important;
    }

    /* Remove qualquer vermelho padrao do Streamlit quando ele aparece via style inline */
    [style*="rgb(255, 75, 75)"] {
        color: var(--accent-dark) !important;
        border-color: var(--accent) !important;
    }
    [style*="background-color: rgb(255, 75, 75)"] {
        background-color: var(--accent) !important;
    }
    [style*="#ff4b4b"] {
        color: var(--accent-dark) !important;
        border-color: var(--accent) !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EXAMPLE_QUERIES = [
    "generative AI in education",
    "ChatGPT use in higher education",
    "AI-based tutoring systems",
    "learning analytics for student performance prediction",
]


@st.cache_resource(show_spinner=False)
def load_models(corpus_path: str, mtime: float, k1: float, b: float, alpha: float):
    """Carrega corpus e constroi os tres recuperadores."""
    docs = read_jsonl(Path(corpus_path))
    bm25 = BM25Retriever(docs, k1=k1, b=b)
    knn = KNNRetriever(docs)
    hybrid = HybridRetriever(bm25, knn, alpha=alpha)
    return docs, bm25, knn, hybrid


def safe_get(doc: dict[str, Any], key: str, default: str = "") -> str:
    value = doc.get(key, default)
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value) if value is not None else default


def make_arxiv_link(doc_id: str, explicit_url: str = "") -> str:
    if explicit_url:
        return explicit_url
    if doc_id.startswith("sample-"):
        return ""
    return f"https://arxiv.org/abs/{doc_id}"


def search_with_model(model_name: str, query: str, top_k: int, bm25, knn, hybrid):
    if model_name == "BM25":
        return bm25.search(query, top_k)
    if model_name == "KNN/TF-IDF":
        return knn.search(query, top_k)
    return hybrid.search(query, top_k)


def results_to_dataframe(results: list[tuple[dict, float]]) -> pd.DataFrame:
    rows = []
    for rank, (doc, score) in enumerate(results, start=1):
        doc_id = safe_get(doc, "id")
        rows.append(
            {
                "rank": rank,
                "id": doc_id,
                "titulo": safe_get(doc, "title"),
                "data": safe_get(doc, "date"),
                "categorias": safe_get(doc, "categories"),
                "score": round(float(score), 6),
                "link": make_arxiv_link(doc_id, safe_get(doc, "url")),
            }
        )
    return pd.DataFrame(rows)


def render_result_cards(results: list[tuple[dict, float]]) -> None:
    for rank, (doc, score) in enumerate(results, start=1):
        doc_id = safe_get(doc, "id")
        title = safe_get(doc, "title", "Sem titulo")
        date = safe_get(doc, "date", "Sem data")
        categories = safe_get(doc, "categories", "Sem categoria")
        authors = safe_get(doc, "authors", "Autores nao informados")
        abstract = safe_get(doc, "abstract", "Resumo nao disponivel")
        url = make_arxiv_link(doc_id, safe_get(doc, "url"))

        st.markdown(
            f"""
            <div class="result-card">
                <div class="doc-title">
                    <span class="rank-badge">#{rank}</span>{title}
                    <span class="score-badge">score {score:.4f}</span>
                </div>
                <div class="small-muted">
                    ID: {doc_id} · Data: {date} · Categorias: {categories}
                </div>
                <div class="small-muted">Autores: {authors}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Ver resumo do artigo"):
            st.write(abstract)
            if url:
                st.link_button("Abrir artigo", url)


def render_evaluation_tab() -> None:
    st.subheader("Resultados de avaliacao")
    if RESULTS_CSV_PATH.exists():
        df = pd.read_csv(RESULTS_CSV_PATH)
        st.dataframe(df, use_container_width=True, hide_index=True)
        numeric_cols = [c for c in df.columns if c.lower() not in {"model", "sistema", "run"}]
        if numeric_cols:
            first_text_col = next((c for c in df.columns if c not in numeric_cols), df.columns[0])
            st.caption("Tabela gerada pelo script src/evaluate.py a partir dos arquivos TREC.")
            selected_metric = st.selectbox("Metrica para visualizar", numeric_cols)
            chart_df = df[[first_text_col, selected_metric]].set_index(first_text_col)
            st.bar_chart(chart_df)
    else:
        st.info("Ainda nao existe runs/evaluation_results.csv. Rode python run_pipeline.py primeiro.")

    st.markdown(
        """
        **Como interpretar:** P@10 mede a qualidade dos dez primeiros resultados, R@10 mede a cobertura,
        MAP recompensa rankings que colocam documentos relevantes no topo e nDCG@10 e util quando ha
        relevancia graduada, como 0, 1 e 2.
        """
    )


def main() -> None:
    st.markdown('<div class="main-title">Buscador Cientifico de Artigos</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="subtitle">Tema: {THEME}. Interface demonstrativa para o componente de recuperacao do RAG.</div>',
        unsafe_allow_html=True,
    )

    if not CORPUS_PATH.exists():
        st.error("Corpus nao encontrado. Verifique data/processed/corpus.jsonl ou rode a coleta.")
        st.stop()

    corpus_mtime = CORPUS_PATH.stat().st_mtime

    with st.spinner("Carregando corpus e modelos de recuperacao..."):
        docs, bm25, knn, hybrid = load_models(
            str(CORPUS_PATH), corpus_mtime, BM25_K1, BM25_B, HYBRID_ALPHA
        )

    with st.sidebar:
        st.header("Configuracao da busca")
        model_name = st.selectbox(
            "Modelo de recuperacao",
            ["Hibrido BM25 + KNN", "BM25", "KNN/TF-IDF"],
            index=0,
        )
        top_k = st.slider("Quantidade de resultados", min_value=3, max_value=30, value=DEFAULT_TOP_K)
        st.divider()
        st.caption("Parametros atuais")
        st.write(f"BM25 k1: {BM25_K1}")
        st.write(f"BM25 b: {BM25_B}")
        st.write(f"Alpha hibrido: {HYBRID_ALPHA}")
        st.divider()
        st.caption("Base carregada")
        st.write(f"Documentos: {len(docs)}")
        st.write(f"Arquivo: {CORPUS_PATH.name}")

    tab_search, tab_corpus, tab_eval, tab_about = st.tabs(
        ["Buscar artigos", "Corpus", "Avaliacao", "Sobre o projeto"]
    )

    with tab_search:
        if "query_text" not in st.session_state:
            st.session_state.query_text = "generative AI in education"

        st.markdown('<div class="example-title">Consultas prontas para demonstracao</div>', unsafe_allow_html=True)
        example_cols = st.columns(4)
        for idx, example_query in enumerate(EXAMPLE_QUERIES):
            with example_cols[idx]:
                if st.button(example_query, use_container_width=True, key=f"example_{idx}"):
                    st.session_state.query_text = example_query

        st.markdown(
            '<div class="search-help">A consulta e processada pelos recuperadores BM25, KNN/TF-IDF e Hibrido.</div>',
            unsafe_allow_html=True,
        )

        col_query, col_button = st.columns([5, 1.15], vertical_alignment="bottom")
        with col_query:
            query = st.text_input(
                "Consulta cientifica",
                key="query_text",
                placeholder="Ex.: intelligent tutoring systems for mathematics learning",
            )
        with col_button:
            search_button = st.button("Buscar", type="primary", use_container_width=True)

        if search_button or query:
            results = search_with_model(model_name, query, top_k, bm25, knn, hybrid)
            st.markdown(f"**Modelo:** {model_name} · **Consulta:** `{query}`")
            table_df = results_to_dataframe(results)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Baixar resultados em CSV",
                data=table_df.to_csv(index=False).encode("utf-8"),
                file_name="resultados_busca.csv",
                mime="text/csv",
            )
            st.subheader("Resultados detalhados")
            render_result_cards(results)

    with tab_corpus:
        st.subheader("Visao geral do corpus")
        corpus_df = pd.DataFrame(
            [
                {
                    "id": safe_get(doc, "id"),
                    "titulo": safe_get(doc, "title"),
                    "data": safe_get(doc, "date"),
                    "categorias": safe_get(doc, "categories"),
                    "fonte": safe_get(doc, "source"),
                }
                for doc in docs
            ]
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Documentos", len(docs))
        c2.metric("Categorias distintas", corpus_df["categorias"].nunique())
        c3.metric("Periodo", f"{corpus_df['data'].min()} a {corpus_df['data'].max()}")
        st.dataframe(corpus_df, use_container_width=True, hide_index=True)

    with tab_eval:
        render_evaluation_tab()

    with tab_about:
        st.subheader("Por que essa interface nao parece um chatbot?")
        st.write(
            "A interface foi estruturada como um buscador academico: o usuario informa uma consulta, "
            "escolhe o recuperador, visualiza rankings, scores, metadados do artigo e metricas de avaliacao. "
            "Isso reforca que o trabalho implementa o componente de recuperacao do RAG, nao um assistente conversacional."
        )
        st.markdown(
            """
            **Fluxo implementado:**

            1. Corpus de artigos cientificos sobre IA na Educacao.
            2. Pre-processamento textual de titulo e resumo.
            3. Recuperacao por BM25.
            4. Recuperacao por KNN com TF-IDF.
            5. Combinacao hibrida BM25 + KNN.
            6. Avaliacao com queries, qrels e metricas de IR.
            """
        )
        st.info("Para a entrega final, use a coleta real do ArXiv e substitua os qrels de exemplo por julgamentos manuais.")


if __name__ == "__main__":
    main()
