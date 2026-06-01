# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

def run_eda():
    dataset_path = "dataset_consolidado.parquet"
    
    if not os.path.exists(dataset_path):
        print(f"Arquivo {dataset_path} nao encontrado. Por favor, rode leitura.py primeiro.")
        return

    print("Carregando dados...")
    df = pd.read_parquet(dataset_path)
    
    report = []
    report.append("# Relatorio de Analise Exploratoria (EDA) - Sprint 1\n")
    
    # 1. Estatisticas Gerais
    report.append("## 1. Estatisticas Gerais")
    stats = df[['idade', 'qnt_acionamentos_total', 'qnt_cpc_total', 'qnt_alo_total']].describe()
    report.append(stats.to_markdown())
    report.append("\n")

    # 2. Distribuicao de Telefones por Cliente
    report.append("## 2. Distribuicao de Telefones por Cliente")
    phones_per_client = df.groupby('id_pessoa')['telefone'].nunique()
    phone_dist = phones_per_client.value_counts().sort_index()
    report.append("Quantidade de telefones por cliente:")
    report.append(phone_dist.to_markdown())
    report.append("\n")
    
    # 3. Analise de CPC (Contato Pessoa Certa)
    report.append("## 3. Analise de CPC")
    # Taxa de CPC global
    total_calls = df['qnt_acionamentos_total'].sum()
    total_cpc = df['qnt_cpc_total'].sum()
    cpc_rate = (total_cpc / total_calls * 100) if total_calls > 0 else 0
    report.append(f"- **Taxa de CPC Global:** {cpc_rate:.2f}%")
    
    # Taxa de CPC por telefone (top 10)
    df['taxa_cpc'] = (df['qnt_cpc_total'] / df['qnt_acionamentos_total']).replace([np.inf, -np.inf], 0).fillna(0)
    report.append("\n")

    # 4. Identificacao de Outliers
    report.append("## 4. Identificacao de Outliers")
    
    # Outliers de Acionamentos (IQR)
    Q1 = df['qnt_acionamentos_total'].quantile(0.25)
    Q3 = df['qnt_acionamentos_total'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outliers_calls = df[df['qnt_acionamentos_total'] > upper_bound]
    
    report.append(f"- **Limite superior para acionamentos (IQR):** {upper_bound}")
    report.append(f"- **Quantidade de registros acima do limite:** {len(outliers_calls)} ({len(outliers_calls)/len(df)*100:.2f}%)")
    
    # Clientes com muitos telefones
    top_phone_holders = phones_per_client[phones_per_client > 5]
    report.append(f"- **Clientes com mais de 5 telefones:** {len(top_phone_holders)}")
    
    # Telefones "Spam" ou Ineficientes (Muitos acionamentos, 0 CPC)
    inefficient_phones = df[(df['qnt_acionamentos_total'] > 50) & (df['qnt_cpc_total'] == 0)]
    report.append(f"- **Telefones ineficientes (>50 acionamentos e 0 CPC):** {len(inefficient_phones)}")
    
    # Salvando o report
    with open("eda_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print("EDA finalizada. Relatorio gerado em eda_report.md")

if __name__ == "__main__":
    run_eda()
