# -*- coding: utf-8 -*-
import pandas as pd
import glob

def carregar_dados():
    """Lê os arquivos parquet e retorna um único DataFrame."""
    arquivos_parquet = glob.glob("*.parquet")
    # Filtra para não ler o arquivo limpo se ele já existir
    arquivos = [f for f in arquivos_parquet if "limpo" not in f]
    arquivos.sort()
    
    print(f"Lendo {len(arquivos)} arquivos Parquet...")
    dfs = []
    for arq in arquivos:
        try:
            dfs.append(pd.read_parquet(arq))
        except Exception as e:
            print(f"Erro ao ler {arq}: {e}")
            
    if not dfs:
        raise ValueError("Nenhum dado encontrado.")
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Dados carregados com sucesso! Formato: {df.shape}")
    return df

def tratar_nulos(df):
    """Realiza o tratamento de nulos conforme os tipos de dados."""
    print("Iniciando tratamento de nulos...")
    nulos = df.isnull().sum()
    colunas_com_nulos = nulos[nulos > 0]
    
    for col in colunas_com_nulos.index:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Preenche nulos numéricos com 0 (para contagens/quantidades) ou mediana
            if col.startswith("qnt_") or col.endswith("_futuro"):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())
        else:
            # Preenche categóricos com a moda ou 'Desconhecido'
            df[col] = df[col].fillna('Desconhecido')
            
    print(f"Tratamento finalizado. Nulos restantes: {df.isnull().sum().sum()}")
    return df

def rodar_queries(df):
    """Exemplos de queries utilizando o schema fornecido em Pandas."""
    print("\n" + "="*50)
    print("RESULTADOS DAS QUERIES EM PANDAS")
    print("="*50)
    
    # Query 1: Total de Pessoas e Média de Idade por UF
    print("\n--> Query 1: Total de Pessoas e Média de Idade por UF")
    q1 = df.groupby('uf').agg(
        total_pessoas=('id_pessoa', 'count'),
        idade_media=('idade', 'mean')
    ).reset_index().sort_values(by='total_pessoas', ascending=False)
    print(q1.head(10)) # Mostrando top 10
    q1.to_csv("resultado_query1_uf.csv", index=False)
    
    # Query 2: Eficiência de Contato por Carteira (CPC vs Acionamentos)
    print("\n--> Query 2: Eficiência de Contato por Carteira (CPC vs Acionamentos Totais)")
    # Agrupando por desc_carteira e somando as métricas
    q2 = df.groupby('desc_carteira').agg(
        total_acionamentos=('qnt_acionamentos_total', 'sum'),
        total_cpc=('qnt_cpc_total', 'sum')
    ).reset_index()
    # Calculando a taxa de cpc sobre acionamentos
    q2['taxa_cpc_%'] = (q2['total_cpc'] / q2['total_acionamentos'] * 100).round(2)
    q2 = q2.sort_values(by='taxa_cpc_%', ascending=False)
    print(q2.head(10))
    q2.to_csv("resultado_query2_carteira.csv", index=False)
    
    # Query 3: Analisando as promessas de pagamento (futuro) por Sexo e Tipo de Pessoa
    print("\n--> Query 3: Promessas Futuras por Sexo e Tipo de Pessoa")
    q3 = df.groupby(['sexo', 'tipo_pessoa']).agg(
        promessas_futuras=('promessa_futuro', 'sum'),
        media_idade=('idade', 'mean')
    ).reset_index().sort_values(by='promessas_futuras', ascending=False)
    print(q3)
    q3.to_csv("resultado_query3_promessas.csv", index=False)

    # Query 4: Top 5 Cidades com maior quantidade de quedas de ligação (30 dias)
    print("\n--> Query 4: Top 5 Cidades com mais Quedas de Ligação (Últimos 30 dias)")
    q4 = df.groupby(['uf', 'cidade']).agg(
        total_quedas_30d=('qnt_queda_ligacao_30d', 'sum')
    ).reset_index().sort_values(by='total_quedas_30d', ascending=False)
    print(q4.head(5))
    q4.to_csv("resultado_query4_quedas.csv", index=False)
    
    # Query 5: Filtragem específica (Ex: Pessoas de SP com mais de 5 acionamentos que tiveram Alô)
    print("\n--> Query 5: Exemplo de Filtro (SP, Acionamentos > 5, Teve Alô)")
    filtro = (df['uf'] == 'SP') & (df['qnt_acionamentos_total'] > 5) & (df['qnt_alo_total'] > 0)
    q5 = df[filtro]
    print(f"Total de registros encontrados: {q5.shape[0]}")
    if q5.shape[0] > 0:
        resultado_q5 = q5[['id_pessoa', 'idade', 'cidade', 'qnt_acionamentos_total', 'qnt_alo_total']]
        print(resultado_q5.head(5))
        resultado_q5.to_csv("resultado_query5_filtro.csv", index=False)

def main():
    # 1. Carregar os dados (já garantindo a integridade dos arquivos compactados)
    df = carregar_dados()
    
    # 2. Tratamento de Nulos
    df_limpo = tratar_nulos(df)
    
    # 3. Salvar o DataFrame limpo e consolidado em um único arquivo
    output_file = "dataset_consolidado.parquet"
    print(f"\nSalvando o dataset final limpo e consolidado em '{output_file}'...")
    df_limpo.to_parquet(output_file, index=False)
    print("Dataset salvo com sucesso!")    
    # 3. Executar exemplos de queries usando Pandas
    rodar_queries(df_limpo)

if __name__ == "__main__":
    main()
