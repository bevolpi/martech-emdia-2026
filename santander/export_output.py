# -*- coding: utf-8 -*-
"""
Sprint 2 - Task 8: Exportacao do Output Formatado

Padrao acordado com Emdia:
- Colunas: id_pessoa, telefone, rank_telefone, score_conversao, flag_conversao,
           sexo, uf, tipo_linha, idade, dt_processamento
- Tipos: id_pessoa (str), telefone (str), rank_telefone (int8),
         score_conversao (float32), flag_conversao (int8),
         sexo (category), uf (category), tipo_linha (category),
         idade (int16), dt_processamento (date str YYYY-MM-DD)
- Naming do arquivo: output_scored_YYYYMMDD.parquet (producao)
                     output_scored_YYYYMMDD.csv     (compartilhamento)
"""

import pandas as pd
import os
from datetime import date

# ============================================================
# SCHEMA ACORDADO COM EMDIA
# ============================================================
# Mapeamento: coluna_interna -> coluna_output
COLUMN_MAP = {
    'id_pessoa'                     : 'id_pessoa',
    'telefone'                      : 'telefone',
    'score_probabilidade_conversao' : 'score_conversao',
    'score_predicao_classe'         : 'flag_conversao',
    'sexo'                          : 'sexo',
    'uf'                            : 'uf',
    'tipo_linha'                    : 'tipo_linha',
    'idade'                         : 'idade',
}

# Tipos finais de cada coluna (otimizados para tamanho e interoperabilidade)
DTYPE_MAP = {
    'id_pessoa'       : str,
    'telefone'        : str,
    'rank_telefone'   : 'int8',
    'score_conversao' : 'float32',
    'flag_conversao'  : 'int8',
    'sexo'            : 'category',
    'uf'              : 'category',
    'tipo_linha'      : 'category',
    'idade'           : 'Int16',    # Nullable int (aceita NaN)
    'dt_processamento': str,
}

OUTPUT_COLUMNS = [
    'id_pessoa', 'telefone', 'rank_telefone',
    'score_conversao', 'flag_conversao',
    'sexo', 'uf', 'tipo_linha', 'idade',
    'dt_processamento'
]

def exportar_output(
    input_path: str = "dataset_scored_diario_top5.parquet",
    output_dir: str = ".",
    export_csv: bool = True
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Arquivo {input_path} nao encontrado. "
            "Execute pipeline_diario.py primeiro."
        )

    print("="*55)
    print("EXPORTACAO DO OUTPUT FORMATADO (PADRAO EMDIA)")
    print("="*55)

    print(f"\nCarregando {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Registros carregados: {len(df):,}")

    # ----------------------------------------------------------
    # 1. SELECIONAR E RENOMEAR COLUNAS
    # ----------------------------------------------------------
    colunas_faltando = [k for k in COLUMN_MAP if k not in df.columns]
    if colunas_faltando:
        raise ValueError(f"Colunas obrigatorias ausentes no input: {colunas_faltando}")

    df_out = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

    # ----------------------------------------------------------
    # 2. DEDUPLICAR E CALCULAR TOP 5 POR CLIENTE
    # ----------------------------------------------------------
    df_out['id_pessoa'] = df_out['id_pessoa'].astype(str)
    df_out['telefone'] = df_out['telefone'].astype(str)
    df_out = (
        df_out
        .sort_values(['id_pessoa', 'score_conversao'], ascending=[True, False])
        .drop_duplicates(subset=['id_pessoa', 'telefone'], keep='first')
        .reset_index(drop=True)
    )
    df_out['rank_telefone'] = df_out.groupby('id_pessoa').cumcount().add(1).astype('int8')
    df_out = df_out[df_out['rank_telefone'] <= 5].copy()

    # ----------------------------------------------------------
    # 3. ADICIONAR DATA DE PROCESSAMENTO
    # ----------------------------------------------------------
    dt_hoje = date.today().strftime("%Y-%m-%d")
    df_out['dt_processamento'] = dt_hoje

    # ----------------------------------------------------------
    # 4. APLICAR TIPOS CORRETOS
    # ----------------------------------------------------------
    print("\nAplicando schema de tipos...")
    for col, dtype in DTYPE_MAP.items():
        try:
            df_out[col] = df_out[col].astype(dtype)
        except Exception as e:
            raise ValueError(f"Nao foi possivel converter '{col}' para {dtype}: {e}") from e

    # ----------------------------------------------------------
    # 5. ORDENAR (id_pessoa ASC, rank_telefone ASC)
    # ----------------------------------------------------------
    df_out = df_out.sort_values(['id_pessoa', 'rank_telefone']).reset_index(drop=True)

    # ----------------------------------------------------------
    # 6. REORDENAR COLUNAS NA ORDEM FINAL
    # ----------------------------------------------------------
    df_out = df_out[OUTPUT_COLUMNS]

    # ----------------------------------------------------------
    # 7. VALIDACOES DE QUALIDADE
    # ----------------------------------------------------------
    print("\n--- Validacoes de Qualidade ---")

    erros_validacao = []

    # Checar nulos
    nulos = df_out.drop(columns=['idade']).isnull().sum()
    nulos_presentes = nulos[nulos > 0]
    if nulos_presentes.empty:
        print("  [OK] Sem valores nulos nas colunas obrigatorias nao-nullable.")
    else:
        erros_validacao.append(f"Nulos encontrados em colunas obrigatorias:\n{nulos_presentes}")

    # Checar range do rank (deve ser 1 a 5)
    rank_min = df_out['rank_telefone'].min()
    rank_max = df_out['rank_telefone'].max()
    if rank_min == 1 and rank_max <= 5:
        print(f"  [OK] rank_telefone no intervalo esperado: [{rank_min}, {rank_max}]")
    else:
        erros_validacao.append(f"rank_telefone fora do intervalo: [{rank_min}, {rank_max}]")

    # Checar score entre 0 e 1
    score_min = df_out['score_conversao'].min()
    score_max = df_out['score_conversao'].max()
    if 0.0 <= score_min and score_max <= 1.0:
        print(f"  [OK] score_conversao no intervalo [0, 1]: min={score_min:.4f}, max={score_max:.4f}")
    else:
        erros_validacao.append(f"score_conversao fora do intervalo: min={score_min:.4f}, max={score_max:.4f}")

    # Checar duplicatas de (id_pessoa, telefone)
    dups = df_out.duplicated(subset=['id_pessoa', 'telefone']).sum()
    if dups == 0:
        print("  [OK] Sem duplicatas de (id_pessoa, telefone).")
    else:
        erros_validacao.append(f"{dups} duplicatas de (id_pessoa, telefone) encontradas")

    colunas_incorretas = list(df_out.columns) != OUTPUT_COLUMNS
    if colunas_incorretas:
        erros_validacao.append(f"Schema de colunas diferente do padrao Emdia: {list(df_out.columns)}")
    else:
        print("  [OK] Colunas na ordem acordada com Emdia.")

    if erros_validacao:
        raise ValueError("Falha nas validacoes do output Emdia:\n- " + "\n- ".join(erros_validacao))

    # ----------------------------------------------------------
    # 8. SALVAR ARQUIVOS
    # ----------------------------------------------------------
    dt_filename = date.today().strftime("%Y%m%d")
    os.makedirs(output_dir, exist_ok=True)

    # Parquet (producao)
    parquet_path = os.path.join(output_dir, f"output_scored_{dt_filename}.parquet")
    df_out.to_parquet(parquet_path, index=False)

    # CSV (compartilhamento / auditoria)
    if export_csv:
        csv_path = os.path.join(output_dir, f"output_scored_{dt_filename}.csv")
        df_out.to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')

    # ----------------------------------------------------------
    # 9. RESUMO FINAL
    # ----------------------------------------------------------
    print("\n" + "="*55)
    print("OUTPUT GERADO COM SUCESSO")
    print("="*55)
    print(f"  Registros exportados  : {len(df_out):,}")
    print(f"  Clientes unicos       : {df_out['id_pessoa'].nunique():,}")
    print(f"  Data processamento    : {dt_hoje}")
    print(f"  Arquivo Parquet       : {parquet_path}")
    if export_csv:
        print(f"  Arquivo CSV           : {csv_path}")
    print("\nColunas e tipos do output final:")
    for col in df_out.columns:
        print(f"  {col:<22}: {str(df_out[col].dtype):<12}")
    print("="*55)

    return df_out


if __name__ == "__main__":
    exportar_output(
        input_path="dataset_scored_diario_top5.parquet",
        output_dir=".",
        export_csv=True
    )
