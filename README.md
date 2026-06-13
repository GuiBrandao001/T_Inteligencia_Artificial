# Trabalho de IA - Recuperacao de Artigos sobre Uso de IA na Educacao

Este projeto inicial implementa o componente de **recuperacao** do RAG para artigos cientificos sobre o uso de Inteligencia Artificial na Educacao.

O projeto ja vem com um corpus pequeno de exemplo para testar o fluxo. Para a entrega final, substitua esse corpus pela coleta real do ArXiv com entre 1.000 e 5.000 artigos.

## O que ja esta pronto

- Estrutura de pastas do projeto
- Configuracao do tema IA na Educacao
- Corpus de exemplo em `data/processed/corpus.jsonl`
- Queries iniciais em `eval/queries.tsv`
- Qrels de exemplo em `eval/qrels.tsv`
- Recuperador BM25
- Recuperador KNN com TF-IDF
- Ranking hibrido BM25 + KNN
- Avaliacao com P@10, R@10, MAP e nDCG@10
- Script de demonstracao por consulta

## Instalar no macOS

Entre na pasta do projeto:

```bash
cd trabalho_ia_educacao_inicial
```

Crie o ambiente virtual:

```bash
python3 -m venv .venv
```

Ative o ambiente:

```bash
source .venv/bin/activate
```

Atualize o pip:

```bash
python -m pip install --upgrade pip
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Testar o projeto inicial

Com o ambiente ativado, rode:

```bash
python run_pipeline.py
```

Esse comando gera os rankings, avalia os modelos e executa uma consulta de demonstracao.

## Rodar cada etapa separadamente

Gerar os rankings:

```bash
python src/generate_runs.py --top-k 10
```

Avaliar os rankings:

```bash
python src/evaluate.py --k 10
```

Testar uma consulta:

```bash
python src/demo.py "generative AI in education" --top-k 5
```

## Coletar artigos reais no ArXiv

Este comando precisa de internet:

```bash
python src/collect_arxiv.py --max-results 2000
```

Depois da coleta, o arquivo `data/processed/corpus.jsonl` sera substituido por artigos reais.

Em seguida, gere novamente os rankings:

```bash
python src/generate_runs.py --top-k 100
```

## Arquivos principais

```text
src/config.py            Configuracoes do tema, caminhos e parametros
src/collect_arxiv.py     Coleta artigos do ArXiv
src/preprocess.py        Pre-processamento textual
src/bm25_retriever.py    Recuperador BM25
src/knn_retriever.py     Recuperador KNN/TF-IDF
src/hybrid_retriever.py  Ranking hibrido
src/generate_runs.py     Gera runs no formato TREC
src/evaluate.py          Calcula metricas de avaliacao
src/demo.py              Demonstra consulta textual
```

## Proximos passos para transformar em entrega final

1. Ajustar palavras-chave em `src/config.py`, se necessario.
2. Rodar `src/collect_arxiv.py` para gerar corpus real.
3. Conferir se o corpus tem entre 1.000 e 5.000 artigos.
4. Rodar BM25, KNN e hibrido.
5. Fazer pooling dos top-10 ou top-20 resultados.
6. Substituir o `eval/qrels.tsv` de exemplo por julgamentos manuais reais.
7. Rodar `src/evaluate.py`.
8. Montar tabelas e analise no relatorio.
9. Gravar video de ate 8 minutos.
10. Compactar tudo em `.zip` para submissao.

## Interface atualizada

A interface grafica usa a cor institucional `#009eb8`, possui consultas prontas para demonstracao e mantem o botao de busca alinhado ao campo de consulta.

Para abrir a interface:

```bash
python -m streamlit run app.py
```

Consultas prontas disponiveis na tela:

- generative AI in education
- ChatGPT use in higher education
- AI-based tutoring systems
- learning analytics for student performance prediction


## Tema visual

A interface usa o tema azul `#009eb8`, configurado em `.streamlit/config.toml`, para evitar elementos vermelhos no Streamlit.


## Interface v4

Esta versao aplica a identidade visual azul `#009eb8` em botoes, abas, slider de quantidade de resultados, links, metricas e destaques visuais da interface.
