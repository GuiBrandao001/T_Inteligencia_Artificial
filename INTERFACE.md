# Interface grafica

Esta interface foi criada para demonstrar o projeto como um buscador cientifico de artigos sobre uso de IA na Educacao.

Ela evita formato de chatbot. A organizacao e baseada em:

- Campo de busca academica.
- Selecao de recuperador.
- Ranking com scores.
- Tabela de metadados.
- Cards com resumo dos artigos.
- Aba de avaliacao.
- Aba de corpus.

## Comando

```bash
python -m streamlit run app.py
```

## Uso recomendado no video

1. Apresente o tema e o objetivo do trabalho.
2. Abra a interface.
3. Digite uma query, por exemplo: `generative AI in education`.
4. Mostre a diferenca entre BM25, KNN/TF-IDF e Hibrido.
5. Abra a aba Avaliacao e mostre as metricas.
6. Explique que a interface e apenas a camada visual; o nucleo do trabalho esta nos scripts em `src/`.
