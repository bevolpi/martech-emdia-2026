# -*- coding: utf-8 -*-
"""
Registra uma versao do modelo ja treinado sem retreinar.

Uso principal: criar o primeiro historico de versoes a partir do
artefato legado tuned_xgboost_model.joblib.
"""

import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from model_registry import calcular_metricas_classificacao, salvar_modelo_versionado


def registrar_modelo_existente(
    model_path="tuned_xgboost_model.joblib",
    model_name="tuned_xgboost",
    dataset_path="dataset_features.parquet",
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo {model_path} nao encontrado.")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_path} nao encontrado.")

    features_num = ['idade', 'recencia', 'frequencia', 'total_telefones_cliente']
    features_cat = ['sexo', 'tipo_linha', 'uf']
    target = 'target_label'
    feature_columns = features_num + features_cat

    print("Carregando dataset e modelo legado...")
    df = pd.read_parquet(dataset_path)
    model = joblib.load(model_path)

    _, X_test, _, y_test = train_test_split(
        df[feature_columns],
        df[target],
        test_size=0.2,
        random_state=42,
        stratify=df[target],
    )

    print("Calculando metricas da versao...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = calcular_metricas_classificacao(y_test, y_pred, y_proba)

    params = {}
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        params = model.named_steps["classifier"].get_params()

    return salvar_modelo_versionado(
        model=model,
        model_name=model_name,
        metrics=metrics,
        params=params,
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        target_column=target,
        legacy_model_path=model_path,
    )


if __name__ == "__main__":
    registrar_modelo_existente()
