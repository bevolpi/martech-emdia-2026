# Martech Emdia 2026 - Ranking de Telefones

Projeto de Machine Learning para priorizar os melhores telefones de contato por cliente, com foco em CPC e conversao operacional para a Emdia/Santander.

## 1. Problema de Negocio

A base de clientes possui varios telefones associados a cada `id_pessoa`. Na operacao de discagem, ligar para todos os numeros sem prioridade aumenta custo, tempo improdutivo e volume de tentativas em telefones com baixa chance de sucesso.

O objetivo do projeto e criar um pipeline que:

- consolide e trate os dados historicos de acionamento;
- crie features relevantes para prever sucesso de contato;
- treine modelos de classificacao/ranking;
- gere score de conversao por telefone;
- entregue os Top 5 telefones por cliente;
- exporte o resultado em um formato padronizado acordado com a Emdia.

## 2. Solucao Proposta

A solucao foi estruturada como um pipeline de dados e modelo:

```text
base_emdia/
  -> leitura e consolidacao
  -> tratamento de nulos e inconsistencias
  -> feature engineering
  -> treino e avaliacao de modelos
  -> scoring diario
  -> ranking Top 5 por cliente
  -> exportacao padrao Emdia
  -> versionamento do modelo treinado
```

O projeto usa principalmente:

- Python 3
- Pandas e NumPy
- PyArrow/Parquet
- scikit-learn
- XGBoost
- Optuna
- Joblib
- Streamlit e Plotly

## 3. Estrutura do Projeto

```text
.
├── santander/
│   ├── base_emdia/                         # arquivos brutos em Parquet
│   ├── docs/adr/                           # decisoes tecnicas do projeto
│   ├── models/                             # modelos versionados
│   ├── querys/                             # arquivos auxiliares de consultas
│   ├── leitura.py                          # leitura, consolidacao e limpeza inicial
│   ├── eda_analysis.py                     # analise exploratoria
│   ├── eda_report.md                       # relatorio da EDA
│   ├── feature_engineering.py              # criacao de features e target
│   ├── baseline_model.py                   # baseline com Regressao Logistica
│   ├── hyperparameter_tuning.py            # tuning com Optuna e XGBoost
│   ├── evaluation_metrics.py               # metricas de ranking
│   ├── feature_importance.py               # analise de importancia das features
│   ├── pipeline_diario.py                  # pipeline end-to-end de scoring
│   ├── export_output.py                    # exportacao no schema Emdia
│   ├── model_registry.py                   # versionamento de modelos
│   ├── version_existing_model.py           # registra modelo ja treinado
│   ├── dashboard_pandas.py                 # dashboard Streamlit
│   └── requirements.txt                    # dependencias
├── README.md
└── martech-emdia-2026.code-workspace
```

## 4. Principais Etapas Executadas

### 4.1. Escolha do Ambiente

Foi definido o uso de Python 3 para garantir compatibilidade com bibliotecas modernas de ciencia de dados, f-strings, encoding UTF-8 e arquivos Parquet.

A decisao esta documentada em:

```text
santander/docs/adr/0001-escolha-ambiente-python.md
```

### 4.2. Leitura e Consolidacao dos Dados

O script `leitura.py` carrega os arquivos da pasta `base_emdia/`, consolida os dados e aplica tratamentos iniciais.

O resultado principal desta etapa e:

```text
santander/dataset_consolidado.parquet
```

### 4.3. Analise Exploratoria

A EDA identificou pontos importantes de qualidade dos dados:

- idades negativas ou impossiveis;
- telefones com volume extremo de acionamentos;
- telefones com muitas tentativas e nenhum CPC;
- necessidade de padronizacao e tratamento de nulos.

Os achados estao em:

```text
santander/eda_analysis.py
santander/eda_report.md
santander/docs/adr/0002-tratamento-outliers.md
```

### 4.4. Feature Engineering

O script `feature_engineering.py` transforma a base consolidada em uma base pronta para modelagem.

Features criadas ou organizadas:

- `recencia`: aproximacao do tempo desde o ultimo acionamento;
- `frequencia`: volume de acionamentos;
- `tipo_linha`: Movel ou Fixo;
- `operadora_prefix`: proxy de operadora pelo prefixo;
- `ddd`;
- `total_telefones_cliente`;
- features historicas de alo, CPC, produtivo, promessa, quedas, caixa postal e cliente desliga.

Tambem foi criada a variavel alvo:

```text
target_label = 1 quando cpc_futuro == 1 OU promessa_futuro == 1
```

Isso faz o modelo priorizar telefones com maior chance de contato produtivo e conversao.

Output da etapa:

```text
santander/dataset_features.parquet
```

Documentacao:

```text
santander/docs/adr/0003-feature-engineering.md
santander/docs/adr/0006-definicao-target.md
```

### 4.5. Modelo Baseline

Foi treinado um baseline com Regressao Logistica balanceada.

Motivos:

- modelo simples;
- rapido;
- interpretavel;
- bom ponto de comparacao para modelos mais complexos.

Resultado documentado:

```text
AUC-ROC baseline: 0.8131
```

Artefato:

```text
santander/baseline_logistic_model.joblib
```

Documentacao:

```text
santander/docs/adr/0004-modelo-baseline.md
```

### 4.6. Metricas de Referencia

Como o objetivo final e ranking, as metricas escolhidas foram:

- AUC-ROC global;
- Hit Rate @ 1;
- Hit Rate @ 3;
- Hit Rate @ 5.

O Hit Rate @ K mede se pelo menos um telefone produtivo aparece entre os K primeiros telefones do cliente.

Script:

```text
santander/evaluation_metrics.py
```

Documentacao:

```text
santander/docs/adr/0005-metricas-referencia.md
```

### 4.7. Modelo XGBoost com Otimizacao

Foi treinado um modelo XGBoost, com hiperparametros otimizados via Optuna.

O pipeline de modelagem inclui:

- imputacao de valores nulos;
- padronizacao de variaveis numericas;
- OneHotEncoding de variaveis categoricas;
- classificador XGBoost.

Artefato principal:

```text
santander/tuned_xgboost_model.joblib
```

Arquivos auxiliares:

```text
santander/best_hyperparams.json
santander/docs/optuna_optimization_history.html
santander/docs/optuna_param_importances.html
```

### 4.8. Pipeline Diario de Scoring

O `pipeline_diario.py` executa o fluxo end-to-end:

1. le dados brutos;
2. trata nulos;
3. gera features;
4. carrega o modelo treinado;
5. gera probabilidade de conversao;
6. gera classe prevista;
7. ordena telefones por cliente;
8. filtra Top 5;
9. chama a exportacao no padrao Emdia.

Output intermediario:

```text
santander/dataset_scored_diario_top5.parquet
```

### 4.9. Exportacao no Padrao Emdia

O script `export_output.py` gera o output final validado.

Schema acordado:

| Coluna | Tipo | Descricao |
| --- | --- | --- |
| `id_pessoa` | string | Identificador do cliente |
| `telefone` | string | Telefone com DDD |
| `rank_telefone` | int8 | Posicao no ranking |
| `score_conversao` | float32 | Probabilidade prevista |
| `flag_conversao` | int8 | Classe prevista |
| `sexo` | category | Sexo do cliente |
| `uf` | category | UF |
| `tipo_linha` | category | Movel ou Fixo |
| `idade` | Int16 | Idade nullable |
| `dt_processamento` | string | Data YYYY-MM-DD |

Regras validadas:

- maximo de 5 telefones por cliente;
- `rank_telefone` entre 1 e 5;
- sem duplicatas de `(id_pessoa, telefone)`;
- `score_conversao` sempre entre 0 e 1;
- colunas na ordem acordada.

Outputs finais:

```text
santander/output_scored_20260531.parquet
santander/output_scored_20260531.csv
```

Resultado validado:

```text
684.388 linhas
0 duplicatas de (id_pessoa, telefone)
rank maximo = 5
score dentro do intervalo [0, 1]
```

Documentacao:

```text
santander/docs/adr/0007-schema-exportacao-output.md
```

### 4.10. Versionamento do Modelo

Foi criado um registry local de modelos em `model_registry.py`.

Cada versao salva:

- modelo `.joblib`;
- metadata `.json`;
- log consolidado em `model_versions.csv`;
- alias legado, como `tuned_xgboost_model.joblib`, para manter compatibilidade com o pipeline.

Formato:

```text
santander/models/<model_name>/<model_name>_YYYYMMDD_HHMMSS.joblib
```

Versao registrada:

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

Arquivos:

```text
santander/model_registry.py
santander/version_existing_model.py
santander/model_versions.csv
santander/models/tuned_xgboost/tuned_xgboost_20260531_204055.joblib
santander/models/tuned_xgboost/tuned_xgboost_20260531_204055.json
```

Documentacao:

```text
santander/docs/adr/0008-versionamento-modelo.md
```

## 5. Como Executar

Entre na pasta do projeto:

```powershell
cd C:\Users\betina.nazario\OneDrive\ClubeDogues\martech-emdia-2026\santander
```

Instale as dependencias:

```powershell
pip install -r requirements.txt
```

Execute a consolidacao/leitura:

```powershell
python leitura.py
```

Execute a engenharia de features:

```powershell
python feature_engineering.py
```

Treine o baseline:

```powershell
python baseline_model.py
```

Treine o modelo XGBoost com Optuna:

```powershell
python hyperparameter_tuning.py
```

Execute o pipeline diario completo:

```powershell
python pipeline_diario.py
```

Execute apenas a exportacao formatada:

```powershell
python export_output.py
```

Registre uma versao de um modelo ja treinado:

```powershell
python version_existing_model.py
```

Execute o dashboard:

```powershell
streamlit run dashboard_pandas.py
```

## 6. Entregaveis Atuais

Datasets:

```text
dataset_consolidado.parquet
dataset_features.parquet
dataset_scored_diario.parquet
dataset_scored_diario_top5.parquet
```

Outputs para Emdia:

```text
output_scored_20260531.parquet
output_scored_20260531.csv
```

Modelos:

```text
baseline_logistic_model.joblib
tuned_xgboost_model.joblib
tuned_xgboost_reduced_model.joblib
models/tuned_xgboost/tuned_xgboost_20260531_204055.joblib
```

Governanca:

```text
model_versions.csv
docs/adr/
```

Dashboard:

```text
dashboard_pandas.py
```

## 7. Status das Sprints

### Sprint 1

Concluida:

- leitura e consolidacao da base;
- analise exploratoria;
- tratamento de problemas de qualidade;
- feature engineering;
- definicao de features e dataset de treinamento.

### Sprint 2

Concluida:

- definicao da target `target_label`;
- baseline com Regressao Logistica;
- metricas de ranking;
- XGBoost com tuning via Optuna;
- pipeline diario de scoring;
- exportacao formatada no padrao Emdia;
- versionamento do modelo treinado.

### Sprint 3

Itens recomendados para continuidade:

- automatizar execucao do pipeline por data de processamento;
- criar testes automaticos para schema, ranking e exportacao;
- adicionar monitoramento de drift e distribuicao de scores;
- evoluir o dashboard para visao executiva e operacional;
- documentar instalacao e execucao para usuarios nao tecnicos;
- revisar performance do modelo por UF, tipo de linha e grupos de clientes;
- preparar apresentacao final com problema, metodologia, resultados e proximos passos.

## 8. Resumo Executivo

O projeto transforma uma base bruta de historico de telefones em uma solucao completa de ranking para discagem. Ele cria variaveis de comportamento, treina modelos preditivos, calcula scores por telefone, seleciona os Top 5 por cliente e exporta o resultado em um schema validado para a Emdia.

A solucao atual ja possui pipeline end-to-end, output padronizado, modelo versionado e documentacao das principais decisoes tecnicas.
