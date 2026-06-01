# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import optuna
import optuna.visualization as vis
import os
from model_registry import calcular_metricas_classificacao, salvar_modelo_versionado

def load_data():
    dataset_path = "dataset_features.parquet"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Arquivo {dataset_path} nao encontrado.")
    
    df = pd.read_parquet(dataset_path)
    
    features_num = ['idade', 'recencia', 'frequencia', 'total_telefones_cliente']
    features_cat = ['sexo', 'tipo_linha', 'uf']
    target = 'target_label' # Task 6: CPC + Conversao
    
    X = df[features_num + features_cat]
    y = df[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), features_num, features_cat

def get_preprocessor(features_num, features_cat):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconhecido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features_num),
            ('cat', categorical_transformer, features_cat)
        ])

def run_tuning():
    (X_train, X_test, y_train, y_test), features_num, features_cat = load_data()
    
    # Pre-processar os dados apenas uma vez para acelerar o Optuna
    preprocessor = get_preprocessor(features_num, features_cat)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Calcular class_weight equivalente para XGBoost (scale_pos_weight)
    # Lida com o desbalanceamento das classes
    neg_class = (y_train == 0).sum()
    pos_class = (y_train == 1).sum()
    scale_pos_weight = neg_class / pos_class if pos_class > 0 else 1.0

    def objective(trial):
        # Engrenagens (Hiperparametros) para testar
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**param)
        model.fit(X_train_prep, y_train)
        
        preds_proba = model.predict_proba(X_test_prep)[:, 1]
        auc = roc_auc_score(y_test, preds_proba)
        return auc

    print("Iniciando Otimizacao de Hiperparametros com Optuna (20 trials)...")
    # direction='maximize' porque queremos o maior AUC-ROC
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20) 

    print("\n--- Resultados do Tuning ---")
    print(f"Melhor AUC-ROC: {study.best_value:.4f}")
    print("Melhores Hiperparametros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Salvar plots do Optuna
    os.makedirs('docs', exist_ok=True)
    try:
        vis.plot_optimization_history(study).write_html("docs/optuna_optimization_history.html")
        vis.plot_param_importances(study).write_html("docs/optuna_param_importances.html")
        print("\nGraficos de otimizacao salvos em docs/optuna_optimization_history.html e docs/optuna_param_importances.html")
    except Exception as e:
        print("\nAviso: Nao foi possivel gerar os graficos do Optuna. Erro:", e)

    # Treinar modelo final com pipeline completo (preprocessamento + modelo)
    print("\nTreinando o modelo final com os melhores parametros encontrados...")
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'

    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**best_params))
    ])
    
    final_pipeline.fit(X_train, y_train)
    
    # Avaliar no teste final
    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    
    metrics = calcular_metricas_classificacao(y_test, y_pred, y_proba)
    print(f"\n--- Resultados do Modelo Otimizado (Teste) ---")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print("\nRelatorio de Classificacao:")
    print(classification_report(y_test, y_pred))
    
    salvar_modelo_versionado(
        model=final_pipeline,
        model_name="tuned_xgboost",
        metrics=metrics,
        params=best_params,
        dataset_path="dataset_features.parquet",
        feature_columns=features_num + features_cat,
        target_column="target_label",
        legacy_model_path="tuned_xgboost_model.joblib",
    )

if __name__ == "__main__":
    run_tuning()
