# ADR-005: Definicao de Metricas de Referencia (AUC e Top-5)

## Contexto
O objetivo de negocio e gerar um ranking dos 5 melhores telefones por cliente. Precisamos de metricas que avaliem nao apenas se o modelo acerta a classe (CPC vs nao CPC), mas se ele coloca os telefones produtivos no topo da lista de cada cliente.

## Decisao e Motivacao

1. **Hit Rate @ K (Metrica de Negocio Principal):**
    - **Decisao:** Utilizaremos o `Hit Rate @ 1`, `Hit Rate @ 3` e `Hit Rate @ 5`.
    - **Motivacao:** O `Hit Rate @ K` mede a porcentagem de clientes onde conseguimos pelo menos um CPC realizando apenas K chamadas. Isso reflete diretamente a economia de recursos e o aumento de eficiencia operacional.

2. **AUC-ROC (Metrica de Estabilidade):**
    - **Decisao:** Manteremos o `AUC-ROC` global.
    - **Motivacao:** Garante que o modelo tem uma boa separacao geral entre as classes, independente dos agrupamentos por cliente.

3. **Resultados do Baseline (Referencia):**
    - **AUC-ROC Global:** 0.8131
    - **Hit Rate @ 1:** 96.73%
    - **Hit Rate @ 5:** 100.00%
    - **Analise:** O alto Hit Rate sugere que a Regressao Logistica ja consegue priorizar muito bem o telefone principal do cliente. Para clientes com poucos telefones, o modelo e quase perfeito. O desafio sera manter ou superar essa marca em clientes com alta volatilidade de numeros (> 5 telefones).

## Consequencia
- Qualquer novo modelo (Sprint 2) sera avaliado primariamente pelo ganho no `Hit Rate @ 1` em clientes complexos.
- O script `evaluation_metrics.py` sera o padrao para validacao de todos os modelos subsequentes.
