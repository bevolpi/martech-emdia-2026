# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os


def processar_features(df):
    # --- 1. Limpeza de Dados (ADR-002) ---
    print("Limpando dados...")
    # Idade: Filtrar 18-100
    df = df[(df['idade'] >= 18) & (df['idade'] <= 100)].copy()
    
    # Acionamentos: Capping no percentil 99
    p99 = df['qnt_acionamentos_total'].quantile(0.99)
    df['qnt_acionamentos_total'] = df['qnt_acionamentos_total'].clip(upper=p99)
    
    # --- 2. Criacao de Features ---
    print("Criando novas features...")
    
    # A. Recencia (Buckets de 30, 60, 90, 120+) - Vetorizado
    df['recencia'] = 120
    df.loc[df['qnt_acionamentos_90d'] > 0, 'recencia'] = 90
    df.loc[df['qnt_acionamentos_60d'] > 0, 'recencia'] = 60
    df.loc[df['qnt_acionamentos_30d'] > 0, 'recencia'] = 30
    
    # B. Frequencia
    df['frequencia'] = df['qnt_acionamentos_total']
    
    # C. Total de Telefones por Cliente (ADR-002)
    df['total_telefones_cliente'] = df.groupby('id_pessoa')['telefone'].transform('nunique')
    
    # D. Tipo de Linha (Movel vs Fixo) - Vetorizado
    # Se o nono digito (contando de tras pra frente) for '9', e Movel
    df['tipo_linha'] = np.where(df['telefone'].str.get(-9) == '9', 'Movel', 'Fixo')
    
    # E. Operadora (Simplificado por Prefixo) - Vetorizado
    df['operadora_prefix'] = df['telefone'].str[2:5]
    df.loc[df['telefone'].str.len() < 11, 'operadora_prefix'] = df['telefone'].str[2:4]

    # F. DDD
    df['ddd'] = df['telefone'].str[:2]

    # G. Variavel-Alvo (Task 6)
    # CPC confirmado e/ou conversao (promessa)
    df['target_label'] = ((df['cpc_futuro'] == 1) | (df['promessa_futuro'] == 1)).astype(int)

    return df

def feature_engineering():
    dataset_path = "dataset_consolidado.parquet"
    
    if not os.path.exists(dataset_path):
        print(f"Arquivo {dataset_path} nao encontrado.")
        return

    print("Carregando dados...")
    df = pd.read_parquet(dataset_path)
    
    df = processar_features(df)
    
    # --- 3. Salvar Dataset Final ---
    output_path = "dataset_features.parquet"
    print(f"Salvando dataset com features em {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Feature Engineering finalizada com sucesso!")

if __name__ == "__main__":
    feature_engineering()
