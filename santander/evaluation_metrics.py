# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def evaluate_ranking_metrics():
    dataset_path = "dataset_features.parquet"
    model_path = "baseline_logistic_model.joblib"
    
    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        print("Dataset ou Modelo nao encontrados.")
        return

    print("Carregando dados e modelo...")
    df = pd.read_parquet(dataset_path)
    model = joblib.load(model_path)
    
    # Replicar o split do baseline (usando mesmo random_state)
    features_num = ['idade', 'recencia', 'frequencia', 'total_telefones_cliente']
    features_cat = ['sexo', 'tipo_linha', 'uf']
    target = 'target_label' # Task 6: CPC + Conversao
    
    _, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    
    X_test = df_test[features_num + features_cat]
    y_test = df_test[target]
    
    print("Gerando predicoes de probabilidade...")
    df_test = df_test.copy()
    df_test['prob_cpc'] = model.predict_proba(X_test)[:, 1]
    
    # --- Ranking por Cliente ---
    print("Calculando ranking por cliente...")
    df_test['rank'] = df_test.groupby('id_pessoa')['prob_cpc'].rank(ascending=False, method='first')
    
    # --- Metricas ---
    
    # 1. AUC-ROC Global
    auc = roc_auc_score(y_test, df_test['prob_cpc'])
    
    # 2. Hit Rate @ K function
    def calculate_hit_rate(df_local, k):
        clients_with_target = df_local[df_local[target] == 1]['id_pessoa'].unique()
        if len(clients_with_target) == 0: return 0
        hit_at_k = df_local[(df_local['rank'] <= k) & (df_local[target] == 1)]['id_pessoa'].unique()
        return (len(hit_at_k) / len(clients_with_target) * 100)

    hit_1 = calculate_hit_rate(df_test, 1)
    hit_3 = calculate_hit_rate(df_test, 3)
    hit_5 = calculate_hit_rate(df_test, 5)
    
    # 3. Metricas para clientes com muitos telefones (> 5)
    df_multi = df_test[df_test['total_telefones_cliente'] > 5].copy()
    hit_5_multi = calculate_hit_rate(df_multi, 5) if not df_multi.empty else 0
    
    print("\n" + "="*40)
    print("RESULTADOS DAS METRICAS DE RANKING (TARGET: CPC + CONVERSAO)")
    print("="*40)
    print(f"AUC-ROC Global:     {auc:.4f}")
    print(f"Hit Rate @ 1:       {hit_1:.2f}%")
    print(f"Hit Rate @ 3:       {hit_3:.2f}%")
    print(f"Hit Rate @ 5:       {hit_5:.2f}%")
    print(f"Hit Rate @ 5 (>5 tel): {hit_5_multi:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_ranking_metrics()
