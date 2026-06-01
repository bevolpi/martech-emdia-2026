# -*- coding: utf-8 -*-
import pandas as pd

# Ler o dicionário de dados
try:
    df_dict = pd.read_excel("Data_Dictionary_score_telefone_COMPLETO (2).xlsx")
    print("--- Dicionario de Dados ---")
    print(df_dict.to_string(index=False))
except Exception as e:
    print(f"Erro ao ler o dicionario: {e}")
