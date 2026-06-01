# ADR-004: Definicao do Modelo Baseline (Regressao Logistica)

## Contexto
Para o sistema de ranking de telefones, precisamos de um modelo de referencia que seja simples, interpretavel e rapido. O baseline serve para garantir que qualquer complexidade futura (como XGBoost ou Redes Neurais) traga um ganho real de performance que justifique o custo operacional.

## Decisao e Motivacao

1. **Escolha do Algoritmo (Regressao Logistica):**
    - **Decisao:** Utilizamos a `LogisticRegression` com balanceamento de classes (`class_weight='balanced'`).
    - **Motivacao:** A Regressao Logistica e o padrao ouro para baselines em classificacao binaria. Ela e linear, facil de interpretar via coeficientes e lida bem com grandes volumes de dados de forma eficiente. O balanceamento de classes e crucial, ja que o CPC e um evento raro (aprox. 0.34% dos casos).

2. **Metrica Principal (AUC-ROC):**
    - **Decisao:** Focamos no AUC-ROC como metrica de sucesso.
    - **Motivacao:** Como o objetivo final e gerar um **ranking** (os 5 melhores telefones), a capacidade do modelo em distinguir entre as classes (ordenacao) e mais importante do que a acuracia pura. O AUC-ROC mede exatamente essa capacidade de ordenacao.

3. **Resultados Obtidos:**
    - **AUC-ROC:** 0.8131
    - **Interpretacao:** O modelo ja demonstra um forte poder preditivo. Coeficientes negativos altos para `recencia` confirmam que contatos antigos sao menos produtivos. O coeficiente positivo para `tipo_linha_Movel` sugere que celulares sao mais propensos ao CPC no contexto deste dataset.

## Consequencia
- Temos um benchmark de 0.8131 de AUC-ROC. Qualquer modelo futuro deve superar essa marca.
- O modelo baseline esta salvo em `baseline_logistic_model.joblib` para comparacoes futuras.
- O pipeline de preprocessamento (OneHotEncoding + Scaling) foi padronizado no arquivo `baseline_model.py`.
