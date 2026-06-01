# -*- coding: utf-8 -*-
"""
Sprint 2 - Task 7: Feature Importance e Selecao

Analisa a importancia das features no modelo XGBoost treinado,
remove features irrelevantes e avalia o impacto no modelo final.
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# ============================================================
# 1. CARREGAR MODELO TREINADO E EXTRAIR FEATURE IMPORTANCES
# ============================================================
def extrair_feature_importances(model_path="tuned_xgboost_model.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo {model_path} nao encontrado.")

    print(f"Carregando modelo de {model_path}...")
    pipeline = joblib.load(model_path)

    preprocessor: ColumnTransformer = pipeline.named_steps['preprocessor']
    classifier: XGBClassifier = pipeline.named_steps['classifier']

    # Reconstruir os nomes das features apos preprocessamento
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            # Pegar os nomes das categorias geradas pelo OHE
            ohe: OneHotEncoder = transformer.named_steps['onehot']
            cat_names = ohe.get_feature_names_out(cols).tolist()
            feature_names.extend(cat_names)

    importances = classifier.feature_importances_
    assert len(importances) == len(feature_names), "Mismatch entre features e importances!"

    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return df_imp

# ============================================================
# 2. RELATÓRIO DE IMPORTÂNCIAS
# ============================================================
def relatorio_importancias(df_imp):
    print("\n" + "="*55)
    print("FEATURE IMPORTANCES — RANKING COMPLETO")
    print("="*55)
    print(f"{'Rank':<5} {'Feature':<40} {'Importance':>10}  {'% Acumulado':>12}")
    print("-"*70)

    total = df_imp['importance'].sum()
    acumulado = 0
    zero_features = []

    for i, row in df_imp.iterrows():
        acumulado += row['importance']
        pct_acum = acumulado / total * 100
        marker = " ◄ ZERO" if row['importance'] == 0 else ""
        print(f"{i+1:<5} {row['feature']:<40} {row['importance']:>10.5f}  {pct_acum:>10.1f}%{marker}")
        if row['importance'] == 0:
            zero_features.append(row['feature'])

    print("-"*70)
    print(f"\n-> Total de features: {len(df_imp)}")
    print(f"-> Features com importance = 0 (candidatas a remover): {len(zero_features)}")
    if zero_features:
        print(f"  {zero_features}")

    # Threshold 95%: features que cobrem 95% da importância total
    df_imp['imp_acum_pct'] = df_imp['importance'].cumsum() / total * 100
    n_95 = (df_imp['imp_acum_pct'] <= 95).sum() + 1
    print(f"\n-> Features necessarias para cobrir 95% da importancia: {n_95} de {len(df_imp)}")

    return zero_features

# ============================================================
# 3. RETREINAR COM AS TOP FEATURES
# ============================================================
def retreinar_com_top_features(df_imp, dataset_path="dataset_features.parquet", threshold_importance=0.0):
    """
    Retreina o modelo removendo features com importance <= threshold.
    Por padrão remove apenas as de importance == 0.
    """
    if not os.path.exists(dataset_path):
        print(f"\nDataset {dataset_path} nao encontrado. Pulando retreino.")
        return None

    print(f"\n{'='*55}")
    print(f"RETREINO SEM FEATURES DE BAIXA IMPORTÂNCIA (threshold={threshold_importance})")
    print("="*55)

    # Features selecionadas (importance > threshold)
    features_selecionadas = df_imp[df_imp['importance'] > threshold_importance]['feature'].tolist()

    # Recuperar apenas as features originais (não-OHE) de volta
    features_num_orig = ['idade', 'recencia', 'frequencia', 'total_telefones_cliente']
    features_cat_orig = ['sexo', 'tipo_linha', 'uf']

    # Verificar quais originais contribuíram para features selecionadas
    features_num_ok = [f for f in features_num_orig if f in features_selecionadas]
    features_cat_ok = []
    for cat in features_cat_orig:
        # Se ao menos 1 categoria OHE desta variável foi selecionada, mantemos a variável
        if any(f.startswith(f"{cat}_") for f in features_selecionadas):
            features_cat_ok.append(cat)

    print(f"\nFeatures numéricas mantidas: {features_num_ok}")
    print(f"Features categóricas mantidas: {features_cat_ok}")
    print(f"Features numéricas removidas: {set(features_num_orig) - set(features_num_ok)}")
    print(f"Features categóricas removidas: {set(features_cat_orig) - set(features_cat_ok)}")

    # Carregar dados
    df = pd.read_parquet(dataset_path)
    X = df[features_num_ok + features_cat_ok]
    y = df['target_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    neg_class = (y_train == 0).sum()
    pos_class = (y_train == 1).sum()
    scale_pos_weight = neg_class / pos_class if pos_class > 0 else 1.0

    # Pipeline com os melhores hiperparâmetros (reutilizados da Task 3)
    best_params_path = "best_hyperparams.json"
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
        print(f"\nUsando hiperparâmetros salvos de {best_params_path}")
    else:
        # Fallback: parâmetros default robustos
        best_params = {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.011,
                       'subsample': 0.703, 'colsample_bytree': 0.802}
        print("\nUsando hiperparâmetros padrão (best_hyperparams.json não encontrado)")

    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconhecido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, features_num_ok),
        ('cat', categorical_transformer, features_cat_ok)
    ])
    model_reduzido = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**best_params))
    ])

    print("\nTreinando modelo reduzido...")
    model_reduzido.fit(X_train, y_train)

    y_proba = model_reduzido.predict_proba(X_test)[:, 1]
    y_pred  = model_reduzido.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- Resultados do Modelo Reduzido ---")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Salvar modelo reduzido
    out_path = "tuned_xgboost_reduced_model.joblib"
    joblib.dump(model_reduzido, out_path)
    print(f"\nModelo reduzido salvo como {out_path}")

    return auc

# ============================================================
# 4. SALVAR RESULTADO EM CSV
# ============================================================
def salvar_importancias_csv(df_imp, output="feature_importances.csv"):
    df_imp.to_csv(output, index=False)
    print(f"\nImportâncias salvas em {output}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df_imp = extrair_feature_importances()
    zero_features = relatorio_importancias(df_imp)
    salvar_importancias_csv(df_imp)
    retreinar_com_top_features(df_imp, threshold_importance=0.0)
