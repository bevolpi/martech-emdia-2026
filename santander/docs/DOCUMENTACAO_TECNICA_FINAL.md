# Documentacao Tecnica Final

## Projeto Martech Emdia 2026 - Ranking de Telefones

### Cliente

Emdia / Santander

### Objetivo

Desenvolver uma solucao de dados e Machine Learning para priorizar os melhores telefones de contato por cliente, reduzindo esforco operacional de discagem e aumentando a chance de contato produtivo.

O produto final do projeto e um arquivo padronizado com os Top 5 telefones por cliente, ordenados por score de conversao, pronto para consumo pela Emdia na camada de discagem.

---

## 1. Contexto do Problema

A operacao possui clientes com multiplos telefones cadastrados. Sem uma priorizacao inteligente, a discagem tende a consumir tempo e recursos ligando para numeros com baixa probabilidade de sucesso, como telefones inativos, caixa postal, quedas recorrentes ou contatos sem historico produtivo.

O desafio do projeto foi transformar o historico operacional de acionamentos em uma solucao capaz de responder:

> "Qual telefone de cada cliente deve ser priorizado para maximizar a chance de contato certo e conversao?"

Para isso, foi construido um pipeline que consolida dados historicos, cria variaveis preditivas, treina modelos, gera scores e exporta um ranking final validado.

---

## 2. Escopo Entregue

O projeto contempla:

- leitura e consolidacao das bases Emdia;
- analise exploratoria de dados;
- tratamento de inconsistencias;
- criacao de features de comportamento telefonico;
- definicao da variavel-alvo;
- treinamento de modelo baseline;
- treinamento de modelo XGBoost otimizado;
- avaliacao com metricas globais e de ranking;
- pipeline diario de scoring;
- exportacao no schema acordado com a Emdia;
- versionamento do modelo treinado;
- dashboard analitico em Streamlit;
- documentacao tecnica das decisoes.

---

## 3. Tecnologias Utilizadas

| Componente | Tecnologia |
| --- | --- |
| Linguagem | Python 3 |
| Manipulacao de dados | Pandas, NumPy |
| Formato de dados | Parquet, CSV |
| Modelagem | scikit-learn, XGBoost |
| Otimizacao | Optuna |
| Persistencia de modelo | Joblib |
| Visualizacao | Streamlit, Plotly |
| Documentacao | Markdown, ADRs |

Dependencias do projeto:

```text
pyarrow
streamlit
plotly
numpy
optuna
xgboost
scikit-learn
pandas
```

---

## 4. Estrutura do Projeto

```text
santander/
├── base_emdia/                         # bases brutas em Parquet
├── docs/
│   ├── adr/                            # registros de decisoes tecnicas
│   └── DOCUMENTACAO_TECNICA_FINAL.md   # esta documentacao
├── models/                             # modelos versionados
├── querys/                             # consultas e analises auxiliares
├── leitura.py                          # leitura e consolidacao dos dados
├── eda_analysis.py                     # analise exploratoria
├── eda_report.md                       # relatorio da EDA
├── feature_engineering.py              # features e target
├── baseline_model.py                   # modelo baseline
├── hyperparameter_tuning.py            # tuning XGBoost com Optuna
├── evaluation_metrics.py               # metricas de ranking
├── feature_importance.py               # importancia de features
├── pipeline_diario.py                  # pipeline end-to-end
├── export_output.py                    # exportacao padrao Emdia
├── model_registry.py                   # versionamento de modelos
├── version_existing_model.py           # registro de modelo existente
├── dashboard_pandas.py                 # dashboard Streamlit
└── requirements.txt                    # dependencias
```

---

## 5. Pipeline Funcional

O fluxo implementado segue as etapas abaixo:

```text
Dados brutos
  -> leitura e consolidacao
  -> tratamento inicial
  -> feature engineering
  -> treinamento / carregamento do modelo
  -> scoring dos telefones
  -> ordenacao por cliente
  -> Top 5 telefones
  -> exportacao validada
  -> arquivo final para Emdia
```

### 5.1. Leitura e Consolidacao

Responsavel:

```text
leitura.py
```

Entrada:

```text
base_emdia/*.parquet
```

Saida:

```text
dataset_consolidado.parquet
```

Esta etapa carrega os arquivos brutos, consolida a base e aplica tratamentos iniciais de nulos e padronizacoes.

### 5.2. Analise Exploratoria

Responsaveis:

```text
eda_analysis.py
eda_report.md
```

Principais achados:

- idades negativas ou impossiveis;
- acionamentos com valores extremos;
- telefones com alto volume de tentativas e nenhum CPC;
- necessidade de criacao de variaveis de recencia, frequencia e contexto do cliente.

### 5.3. Feature Engineering

Responsavel:

```text
feature_engineering.py
```

Saida:

```text
dataset_features.parquet
```

Principais features criadas:

| Feature | Descricao |
| --- | --- |
| `recencia` | aproxima o tempo desde o ultimo acionamento |
| `frequencia` | volume de acionamentos |
| `tipo_linha` | classifica telefone como Movel ou Fixo |
| `operadora_prefix` | proxy de operadora pelo prefixo |
| `ddd` | DDD do telefone |
| `total_telefones_cliente` | quantidade de telefones por cliente |
| variaveis historicas | alo, CPC, produtivo, promessa, quedas, caixa postal etc. |

### 5.4. Definicao da Target

A variavel-alvo criada foi:

```text
target_label = 1 se cpc_futuro == 1 OU promessa_futuro == 1
```

Conceitos:

- `cpc_futuro`: indica se o telefone gerou Contato com a Pessoa Certa em periodo futuro.
- `promessa_futuro`: indica se o telefone gerou promessa de pagamento ou sinalizacao positiva de conversao em periodo futuro.

Justificativa:

O objetivo operacional nao e apenas fazer alguem atender a chamada. O objetivo e priorizar telefones que gerem contato produtivo, seja por CPC ou por promessa/conversao.

---

## 6. Modelagem

### 6.1. Modelo Baseline

Responsavel:

```text
baseline_model.py
```

Algoritmo:

```text
Regressao Logistica com class_weight='balanced'
```

Motivo da escolha:

- simples;
- interpretavel;
- rapido;
- adequado como referencia inicial.

Resultado documentado:

```text
AUC-ROC: 0.8131
```

Artefato:

```text
baseline_logistic_model.joblib
```

### 6.2. Modelo Otimizado

Responsavel:

```text
hyperparameter_tuning.py
```

Algoritmo:

```text
XGBoost
```

Otimizacao:

```text
Optuna
```

O pipeline do modelo inclui:

- imputacao de valores ausentes;
- escalonamento de variaveis numericas;
- OneHotEncoding de variaveis categoricas;
- classificador XGBoost.

Artefato principal:

```text
tuned_xgboost_model.joblib
```

Arquivos auxiliares:

```text
best_hyperparams.json
docs/optuna_optimization_history.html
docs/optuna_param_importances.html
```

---

## 7. Metricas de Avaliacao

Foram utilizadas metricas globais e metricas orientadas ao ranking.

### 7.1. Metricas Globais

- AUC-ROC;
- accuracy;
- precision;
- recall;
- F1.

### 7.2. Metricas de Ranking

- Hit Rate @ 1;
- Hit Rate @ 3;
- Hit Rate @ 5.

O Hit Rate @ K mede se ao menos um telefone produtivo aparece entre os K primeiros telefones recomendados para um cliente.

Essa metrica e adequada porque o objetivo do projeto nao e apenas classificar telefones, mas ordenar os melhores contatos para a operacao.

### 7.3. Modelo Versionado Atual

Versao:

```text
tuned_xgboost_20260531_204055
```

Metricas registradas:

| Metrica | Valor |
| --- | ---: |
| AUC-ROC | 0.875087 |
| Accuracy | 0.840134 |
| Precision | 0.016330 |
| Recall | 0.775362 |
| F1 | 0.031986 |

Observacao:

A baixa precision e esperada em cenarios de evento raro. O recall elevado indica que o modelo consegue recuperar boa parte dos casos positivos, o que e relevante para um problema de priorizacao e ranking.

---

## 8. Pipeline Diario de Scoring

Responsavel:

```text
pipeline_diario.py
```

Etapas executadas:

1. leitura dos dados brutos;
2. tratamento de nulos;
3. geracao de features;
4. carregamento do modelo treinado;
5. predicao de probabilidade de conversao;
6. predicao da classe;
7. ordenacao por cliente e score;
8. selecao dos Top 5 telefones;
9. exportacao no padrao Emdia.

Output intermediario:

```text
dataset_scored_diario_top5.parquet
```

---

## 9. Exportacao Padrao Emdia

Responsavel:

```text
export_output.py
```

### 9.1. Schema Final

| Coluna | Tipo | Descricao |
| --- | --- | --- |
| `id_pessoa` | string | identificador unico do cliente |
| `telefone` | string | telefone com DDD |
| `rank_telefone` | int8 | posicao no ranking do cliente |
| `score_conversao` | float32 | probabilidade prevista pelo modelo |
| `flag_conversao` | int8 | classe prevista |
| `sexo` | category | sexo do cliente |
| `uf` | category | unidade federativa |
| `tipo_linha` | category | Movel ou Fixo |
| `idade` | Int16 | idade nullable |
| `dt_processamento` | string | data de processamento |

### 9.2. Regras de Validacao

O arquivo final deve atender as seguintes regras:

- conter somente Top 5 telefones por cliente;
- `rank_telefone` deve variar de 1 a 5;
- nao pode haver duplicata de `(id_pessoa, telefone)`;
- `score_conversao` deve estar no intervalo `[0, 1]`;
- colunas devem seguir exatamente a ordem acordada.

### 9.3. Naming dos Arquivos

```text
output_scored_YYYYMMDD.parquet
output_scored_YYYYMMDD.csv
```

O CSV e gerado com:

```text
separador: ;
encoding: utf-8-sig
```

### 9.4. Resultado Validado

Arquivos gerados:

```text
output_scored_20260531.parquet
output_scored_20260531.csv
```

Validacoes:

```text
684.388 registros exportados
499.581 clientes unicos
0 duplicatas de (id_pessoa, telefone)
rank_telefone entre 1 e 5
score_conversao entre 0.0384 e 0.9289
```

---

## 10. Versionamento de Modelo

Responsavel:

```text
model_registry.py
```

O versionamento foi criado para garantir rastreabilidade entre artefato, parametros e metricas.

Cada versao gera:

```text
models/<model_name>/<model_name>_YYYYMMDD_HHMMSS.joblib
models/<model_name>/<model_name>_YYYYMMDD_HHMMSS.json
model_versions.csv
```

O JSON contem:

- id da versao;
- data/hora de treino ou registro;
- caminho do artefato;
- dataset usado;
- target;
- features;
- parametros;
- metricas.

O arquivo `model_versions.csv` consolida as metricas por versao.

Modelo versionado atual:

```text
models/tuned_xgboost/tuned_xgboost_20260531_204055.joblib
```

Metadata:

```text
models/tuned_xgboost/tuned_xgboost_20260531_204055.json
```

Log:

```text
model_versions.csv
```

---

## 11. Dashboard

Responsavel:

```text
dashboard_pandas.py
```

O dashboard foi desenvolvido com Streamlit e Plotly para apoiar analises exploratorias e comunicacao dos resultados.

Ele permite visualizar:

- indicadores gerais da base;
- distribuicao de variaveis;
- comportamento dos scores;
- analises por UF;
- ranking de telefones por cliente.

Execucao:

```powershell
streamlit run dashboard_pandas.py
```

---

## 12. Como Executar o Projeto

Diretorio principal:

```powershell
cd C:\Users\betina.nazario\OneDrive\ClubeDogues\martech-emdia-2026\santander
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Gerar base consolidada:

```powershell
python leitura.py
```

Gerar features:

```powershell
python feature_engineering.py
```

Treinar baseline:

```powershell
python baseline_model.py
```

Treinar XGBoost com Optuna:

```powershell
python hyperparameter_tuning.py
```

Executar pipeline diario:

```powershell
python pipeline_diario.py
```

Executar apenas exportacao:

```powershell
python export_output.py
```

Registrar modelo existente:

```powershell
python version_existing_model.py
```

Executar dashboard:

```powershell
streamlit run dashboard_pandas.py
```

---

## 13. Decisoes Tecnicas Documentadas

As principais decisoes foram registradas em ADRs:

| ADR | Tema |
| --- | --- |
| 0001 | escolha do ambiente Python |
| 0002 | tratamento de outliers |
| 0003 | feature engineering |
| 0004 | modelo baseline |
| 0005 | metricas de referencia |
| 0006 | definicao da target |
| 0007 | schema de exportacao |
| 0008 | versionamento do modelo |

Caminho:

```text
docs/adr/
```

---

## 14. Entregaveis

### Dados

```text
dataset_consolidado.parquet
dataset_features.parquet
dataset_scored_diario.parquet
dataset_scored_diario_top5.parquet
```

### Outputs Emdia

```text
output_scored_20260531.parquet
output_scored_20260531.csv
```

### Modelos

```text
baseline_logistic_model.joblib
tuned_xgboost_model.joblib
tuned_xgboost_reduced_model.joblib
models/tuned_xgboost/tuned_xgboost_20260531_204055.joblib
```

### Governanca

```text
model_versions.csv
models/tuned_xgboost/tuned_xgboost_20260531_204055.json
docs/adr/
```

### Aplicacao Analitica

```text
dashboard_pandas.py
```

---

## 15. Limitacoes Conhecidas

- O pipeline ainda depende de execucao manual via script.
- Nao ha orquestrador agendado configurado.
- Os testes automatizados de schema e qualidade ainda nao foram formalizados em uma suite.
- O monitoramento de drift ainda nao foi implementado.
- O desempenho do modelo deve ser acompanhado por safra, UF, tipo de linha e mudancas operacionais.

---

## 16. Recomendacoes para Sprint 3

Recomendacoes de continuidade:

1. Automatizar execucao diaria do pipeline.
2. Criar testes automaticos para validacao do output.
3. Implementar monitoramento de drift e estabilidade dos scores.
4. Evoluir o dashboard para visao executiva.
5. Criar relatorio automatico por execucao.
6. Revisar performance por segmento, UF e tipo de linha.
7. Formalizar processo de aprovacao de novas versoes de modelo.

---

## 17. Conclusao

O projeto entregou uma solucao completa para priorizacao de telefones por cliente, passando por todas as etapas tecnicas essenciais: ingestao, tratamento, feature engineering, modelagem, avaliacao, scoring, ranking, exportacao e versionamento.

A solucao atual permite gerar um arquivo final validado para a Emdia, contendo os Top 5 telefones por cliente, com score de conversao e schema padronizado. Alem disso, o modelo treinado possui rastreabilidade por versao, metricas registradas e compatibilidade com o pipeline diario.

O projeto esta pronto para apresentacao e continuidade na Sprint 3, com foco em automacao, testes, monitoramento e operacionalizacao.
