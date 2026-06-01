# -*- coding: utf-8 -*-
import pandas as pd
import joblib
import os
import time

# Importando modulos do projeto
from leitura import carregar_dados, tratar_nulos
from feature_engineering import processar_features
from export_output import exportar_output

def run_pipeline():
    start_time = time.time()
    print("="*50)
    print("INICIANDO PIPELINE DIARIO (END-TO-END)")
    print("="*50)
    
    # 1. Ingestao de Dados (Leitura)
    print("\n[ETAPA 1/6] Lendo dados brutos...")
    df_raw = carregar_dados()
    
    # 2. Tratamento de Nulos
    print("\n[ETAPA 2/6] Tratando nulos e limpando base...")
    df_clean = tratar_nulos(df_raw)
    
    # 3. Engenharia de Features
    print("\n[ETAPA 3/6] Gerando features para o modelo...")
    df_features = processar_features(df_clean)
    
    # 4. Carregar Modelo e Escorar
    print("\n[ETAPA 4/6] Carregando modelo e gerando scores...")
    model_path = "tuned_xgboost_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Erro: Modelo {model_path} nao encontrado!")
        
    model_pipeline = joblib.load(model_path)
    
    # Fazer a predicao (o modelo ja trata as variaveis via Pipeline do scikit-learn)
    probabilidades = model_pipeline.predict_proba(df_features)[:, 1]
    classes = model_pipeline.predict(df_features)
    
    # Anexar resultados no DataFrame
    df_features['score_probabilidade_conversao'] = probabilidades
    df_features['score_predicao_classe'] = classes
    
    # 5. Geracao do Ranking (Top 5 telefones por cliente)
    print("\n[ETAPA 5/6] Ordenando e filtrando Top 5 telefones por cliente...")
    # Ordenar por cliente e por score decrescente
    df_features = df_features.sort_values(['id_pessoa', 'score_probabilidade_conversao'], ascending=[True, False])
    # Pegar apenas os top 5 telefones de cada cliente
    df_top5 = df_features.groupby('id_pessoa').head(5).reset_index(drop=True)
    
    # 6. Salvar o arquivo final
    output_path = "dataset_scored_diario_top5.parquet"
    print(f"\nSalvando base final (Top 5) em '{output_path}'...")
    df_top5.to_parquet(output_path, index=False)

    # 6. Exportacao no schema acordado com a Emdia
    print("\n[ETAPA 6/6] Exportando output formatado no padrao Emdia...")
    exportar_output(
        input_path=output_path,
        output_dir=".",
        export_csv=True
    )
    
    end_time = time.time()
    print("="*50)
    print(f"PIPELINE CONCLUIDO COM SUCESSO! Tempo total: {end_time - start_time:.2f} segundos")
    print(f"Total de telefones originais (escorados): {df_features.shape[0]}")
    print(f"Total de telefones apos filtro (Top 5 por cliente): {df_top5.shape[0]}")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()
