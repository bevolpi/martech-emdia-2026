# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
from model_registry import calcular_metricas_classificacao, salvar_modelo_versionado

def train_baseline():
    dataset_path = "dataset_features.parquet"
    if not os.path.exists(dataset_path):
        print(f"Arquivo {dataset_path} nao encontrado. Rode feature_engineering.py primeiro.")
        return

    print("Carregando dados...")
    df = pd.read_parquet(dataset_path)
    
    # Selecionar Features e Target
    features_num = ['idade', 'recencia', 'frequencia', 'total_telefones_cliente']
    features_cat = ['sexo', 'tipo_linha', 'uf']
    target = 'target_label' # Task 6: CPC + Conversao
    
    X = df[features_num + features_cat]
    y = df[target]
    
    print(f"Dataset carregado. Target distribuicao: {y.value_counts(normalize=True).to_dict()}")
    
    # Preprocessamento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconhecido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features_num),
            ('cat', categorical_transformer, features_cat)
        ])
    
    # Pipeline do Modelo
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Treinando Regressao Logistica (Baseline)...")
    model_pipeline.fit(X_train, y_train)
    
    # Avaliacao
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    metrics = calcular_metricas_classificacao(y_test, y_pred, y_proba)
    print(f"\n--- Resultados do Baseline ---")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print("\nRelatorio de Classificacao:")
    print(classification_report(y_test, y_pred))
    
    salvar_modelo_versionado(
        model=model_pipeline,
        model_name="baseline_logistic",
        metrics=metrics,
        params=model_pipeline.named_steps['classifier'].get_params(),
        dataset_path=dataset_path,
        feature_columns=features_num + features_cat,
        target_column=target,
        legacy_model_path="baseline_logistic_model.joblib",
    )

if __name__ == "__main__":
    train_baseline()
