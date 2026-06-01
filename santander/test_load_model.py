# -*- coding: utf-8 -*-
import pandas as pd
import joblib
import os

def run_daily_pipeline_mock():
    model_path = "tuned_xgboost_model.joblib"
    
    if not os.path.exists(model_path):
        print(f"Erro: Modelo {model_path} não encontrado. Treine-o primeiro.")
        return

    print("Carregando o modelo treinado (Joblib)...")
    # Carregamento instantâneo do modelo (sem necessidade de retreino)
    model_pipeline = joblib.load(model_path)
    print("Modelo carregado com sucesso!\n")

    print("Simulando novos clientes para predição diária...")
    # Mock de novos dados chegando no pipeline diário (deve ter as mesmas features do treino)
    novos_clientes = pd.DataFrame({
        'idade': [35, 60, 22],
        'recencia': [10, 150, 5],
        'frequencia': [2, 10, 1],
        'total_telefones_cliente': [1, 3, 1],
        'sexo': ['M', 'F', 'M'],
        'tipo_linha': ['Pre-pago', 'Pos-pago', 'Pre-pago'],
        'uf': ['SP', 'RJ', 'MG']
    })
    
    print(novos_clientes)
    
    print("\nRealizando predições instantâneas...")
    # Predição direta no pipeline que já tem todo o pré-processamento salvo
    predicoes = model_pipeline.predict(novos_clientes)
    probabilidades = model_pipeline.predict_proba(novos_clientes)[:, 1]
    
    # Adicionar resultados no DataFrame mock
    novos_clientes['probabilidade_conversao'] = probabilidades
    novos_clientes['predicao_classe'] = predicoes
    
    print("\nResultados Finais:")
    print(novos_clientes[['idade', 'uf', 'probabilidade_conversao', 'predicao_classe']])

if __name__ == "__main__":
    run_daily_pipeline_mock()
