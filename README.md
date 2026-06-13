# Sistema de Recuperação de Artigos Científicos sobre IA na Educação

Este repositório contém um projeto inicial para o trabalho prático da disciplina de Inteligência Artificial. O objetivo é construir o componente **R** de um sistema RAG, isto é, o módulo de **recuperação de documentos** responsável por receber uma consulta textual e retornar uma lista ranqueada de artigos científicos.

O tema escolhido é:

> **Uso de Inteligência Artificial na Educação**

O projeto possui um fluxo funcional com corpus de exemplo, recuperação por **BM25**, recuperação por **KNN com TF-IDF**, ranking híbrido, avaliação experimental e interface gráfica em **Streamlit**.

---

## Visão geral do projeto

O sistema segue o seguinte pipeline:

```text
Consulta do usuário
        ↓
Pré-processamento textual
        ↓
Recuperador BM25
        ↓
Recuperador KNN/TF-IDF
        ↓
Ranking híbrido BM25 + KNN
        ↓
Lista ranqueada de artigos científicos
        ↓
Avaliação com P@10, R@10, MAP e nDCG
```

A interface gráfica foi criada apenas para facilitar a demonstração do sistema. O núcleo do trabalho está nos algoritmos de recuperação, na construção da coleção, na geração dos rankings e na avaliação experimental.

---

## Tema da coleção

O tema da coleção é o uso de Inteligência Artificial em contextos educacionais.

Exemplos de tópicos contemplados:

- Inteligência Artificial na Educação;
- uso de ChatGPT no ensino superior;
- sistemas tutores inteligentes;
- learning analytics;
- educational data mining;
- aprendizagem personalizada;
- feedback automático;
- grandes modelos de linguagem aplicados à educação;
- predição de desempenho estudantil;
- evasão escolar com aprendizado de máquina.

---

## Tecnologias utilizadas

O projeto utiliza:

- Python;
- Streamlit;
- pandas;
- numpy;
- scikit-learn;
- rank-bm25;
- nltk;
- matplotlib;
- scipy.

---

## Modelos implementados

### BM25

O BM25 é utilizado como recuperador esparso. Ele considera a frequência dos termos da consulta nos documentos e a raridade desses termos na coleção.

No projeto, o BM25 é aplicado sobre o texto formado por:

```text
título + abstract
```

### KNN com TF-IDF

O segundo recuperador utiliza representação vetorial com TF-IDF e busca por similaridade.

A consulta e os documentos são transformados em vetores, e o sistema retorna os documentos mais próximos da consulta.

### Ranking híbrido

O ranking híbrido combina os resultados do BM25 e do KNN/TF-IDF. Esse módulo serve como aprofundamento, permitindo comparar uma estratégia combinada contra os dois recuperadores individuais.

---

## Estrutura de pastas

```text
trabalho_ia_educacao_interface_v5/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── INTERFACE.md
├── LINKS.txt
│
├── data/
│   ├── raw/
│   └── processed/
│       └── corpus.jsonl
│
├── eval/
│   ├── queries.tsv
│   └── qrels.tsv
│
├── runs/
│   ├── bm25.trec
│   ├── knn.trec
│   ├── hybrid.trec
│   └── evaluation_results.csv
│
├── reports/
│   └── relatorio.tex
│
├── src/
│   ├── bm25_retriever.py
│   ├── collect_arxiv.py
│   ├── config.py
│   ├── demo.py
│   ├── evaluate.py
│   ├── hybrid_retriever.py
│   ├── knn_retriever.py
│   └── preprocess.py
│
└── .streamlit/
    └── config.toml
```

---

## Instalação no macOS

Entre na pasta do projeto:

```bash
cd T-GUILHERME_BRANDAO
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

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Execução do pipeline

Para executar o pipeline completo com o corpus disponível:

```bash
python run_pipeline.py
```

Esse comando gera ou atualiza os arquivos de ranking:

```text
runs/bm25.trec
runs/knn.trec
runs/hybrid.trec
runs/evaluation_results.csv
```

---

## Executar a interface gráfica

Para abrir a interface no navegador:

```bash
python -m streamlit run app.py
```

A interface permite:

- digitar uma consulta científica;
- escolher o modelo de recuperação;
- ajustar a quantidade de resultados;
- visualizar os artigos ranqueados;
- consultar informações do corpus;
- visualizar resultados da avaliação;
- baixar os resultados em CSV.

---

## Exemplo de consulta

Exemplos de consultas que podem ser usadas na demonstração:

```text
generative AI in education
ChatGPT use in higher education
AI-based tutoring systems
learning analytics for student performance prediction
educational data mining for dropout prediction
```

---

## Coleta de artigos reais

O projeto vem com um corpus inicial de exemplo para facilitar os testes.

Para coletar artigos reais do ArXiv:

```bash
python src/collect_arxiv.py --max-results 2000
```

Após a coleta, execute novamente:

```bash
python run_pipeline.py
```

---

## Arquivos de avaliação

### `eval/queries.tsv`

Contém as consultas de avaliação.

Exemplo:

```text
q1	generative AI in education
q2	ChatGPT use in higher education
q3	AI-based tutoring systems
```

### `eval/qrels.tsv`

Contém os julgamentos manuais de relevância.

Formato:

```text
qid	0	doc_id	relevancia
```

Exemplo:

```text
q1	0	2401.12345	2
q1	0	2305.99999	1
q1	0	2207.11111	0
```

Escala utilizada:

| Valor | Significado |
|---|---|
| 2 | Altamente relevante |
| 1 | Relevante |
| 0 | Não relevante |

---

## Métricas de avaliação

O projeto calcula métricas clássicas de Recuperação de Informação:

| Métrica | Descrição |
|---|---|
| P@10 | Precisão entre os 10 primeiros resultados |
| R@10 | Revocação entre os 10 primeiros resultados |
| MAP | Média da precisão média das consultas |
| nDCG@10 | Qualidade do ranking considerando relevância graduada |

---

## Demonstração via terminal

Além da interface gráfica, o projeto também permite executar consultas pelo terminal.

Exemplo:

```bash
python src/demo.py "generative AI in education" --top-k 5
```

---

## Relação com o trabalho prático

Este projeto atende à estrutura principal solicitada no trabalho:

| Exigência | Implementação no projeto |
|---|---|
| Definição de tema | Uso de IA na Educação |
| Coleção de artigos | `data/processed/corpus.jsonl` |
| Coleta de dados | `src/collect_arxiv.py` |
| Pré-processamento | `src/preprocess.py` |
| BM25 | `src/bm25_retriever.py` |
| KNN/TF-IDF | `src/knn_retriever.py` |
| Ranking híbrido | `src/hybrid_retriever.py` |
| Queries | `eval/queries.tsv` |
| Qrels | `eval/qrels.tsv` |
| Runs TREC | Pasta `runs/` |
| Avaliação | `src/evaluate.py` |
| Demonstração | `app.py` e `src/demo.py` |
| Relatório | `reports/relatorio.tex` |

---

## Como reproduzir os resultados

Execute os comandos abaixo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
python -m streamlit run app.py
```

---

## Próximos passos para a entrega final

Para transformar esta base em entrega final, ainda é necessário:

- coletar a coleção real com 1.000 a 5.000 artigos;
- revisar as palavras-chave e categorias usadas na coleta;
- gerar os rankings finais com BM25, KNN e híbrido;
- construir o `qrels.tsv` manualmente a partir dos resultados;
- preencher o relatório em formato SBC;
- inserir tabelas e resultados reais no relatório;
- gravar o vídeo de apresentação;
- gerar o arquivo `.zip` final para submissão.

---

## Observação sobre uso de IA generativa

Este projeto pode ter recebido apoio de ferramentas de IA generativa para organização do código, documentação e redação inicial.

No relatório final, recomenda-se incluir uma seção curta declarando esse uso, por exemplo:

> Ferramentas de IA generativa foram utilizadas como apoio para organização inicial do projeto, estruturação do README, sugestões de código e revisão textual. Todas as decisões metodológicas, execução dos experimentos, análise dos resultados e validação dos julgamentos de relevância foram realizadas pelo autor.

---

## Autor

Guilherme Brandão

---

## Status do projeto

Versão inicial funcional para desenvolvimento e demonstração.

O projeto ainda deve ser expandido com a coleta real, avaliação manual dos resultados e finalização do relatório.
