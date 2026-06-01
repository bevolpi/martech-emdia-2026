# ADR-008: Versionamento do Modelo Treinado

## Contexto
O modelo treinado precisa ser rastreavel entre ciclos de treino, comparacoes de
metricas e uso no pipeline diario. Antes desta decisao, os scripts sobrescreviam
apenas os artefatos `baseline_logistic_model.joblib` e `tuned_xgboost_model.joblib`,
sem historico persistente de versoes.

## Decisao
Todo modelo treinado deve ser salvo com `joblib` em um diretorio versionado:

```text
models/<model_name>/<model_name>_YYYYMMDD_HHMMSS.joblib
```

Cada versao tambem gera:

- `models/<model_name>/<model_name>_YYYYMMDD_HHMMSS.json` com metadados,
  features, target, parametros e metricas.
- `model_versions.csv` com uma linha por versao para auditoria e comparacao.
- Artefato legado opcional, como `tuned_xgboost_model.joblib`, para manter
  compatibilidade com o `pipeline_diario.py`.

## Metricas Registradas
- `auc_roc`
- `accuracy`
- `precision`
- `recall`
- `f1`

## Implementacao
O modulo `model_registry.py` centraliza:

- calculo de metricas de classificacao;
- salvamento versionado do artefato `joblib`;
- escrita de metadados JSON;
- atualizacao do log `model_versions.csv`.

Os scripts `baseline_model.py` e `hyperparameter_tuning.py` usam esse modulo ao
final do treinamento. O script `version_existing_model.py` registra uma versao de
um artefato ja treinado sem necessidade de retreino.

## Consequencia
Novos treinamentos passam a manter historico auditavel de modelos e metricas,
enquanto o pipeline diario continua lendo o alias legado `tuned_xgboost_model.joblib`.
