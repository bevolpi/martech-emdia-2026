# -*- coding: utf-8 -*-
"""
Este script é a versão em Python (usando Streamlit e Pandas) do dashboard exemplo_R.Rmd.
Para executar, instale as dependências: pip install streamlit pandas plotly scipy
E rode no terminal: streamlit run dashboard_pandas.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import skew, kurtosis

st.set_page_config(page_title="Projeto_SAN — Martech 2026", layout="wide")

@st.cache_data
def carregar_dados():
    # Lê os dados consolidados gerados anteriormente
    df = pd.read_parquet("dataset_consolidado.parquet")
    
    # Enriquecimento base
    # Converte para string e remove ".0" caso tenha sido lido como float
    df['telefone_str'] = df['telefone'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['ndigitos_tel'] = df['telefone_str'].apply(lambda x: len(x) if x != 'nan' else 0)
    
    def get_tipo_tel(n):
        if n == 11: return "Celular (11 dig)"
        elif n == 10: return "Fixo (10 dig)"
        else: return "Outro"
        
    df['tipo_tel'] = df['ndigitos_tel'].apply(get_tipo_tel)
    df['ddd'] = df['telefone_str'].str[:2]
    
    # Evitando divisão por zero
    df['taxa_cpc_total'] = np.where(df['qnt_acionamentos_total'] > 0,
                                    df['qnt_cpc_total'] / df['qnt_acionamentos_total'],
                                    np.nan)
    df['taxa_cpc_30d'] = np.where(df['qnt_acionamentos_30d'] > 0,
                                  df['qnt_cpc_30d'] / df['qnt_acionamentos_30d'],
                                  np.nan)
                                  
    df['taxa_cpc_60d'] = np.where(df['qnt_acionamentos_60d'] > 0,
                                  df['qnt_cpc_60d'] / df['qnt_acionamentos_60d'],
                                  np.nan)
                                  
    df['taxa_cpc_90d'] = np.where(df['qnt_acionamentos_90d'] > 0,
                                  df['qnt_cpc_90d'] / df['qnt_acionamentos_90d'],
                                  np.nan)
    
    if 'cpc_futuro' in df.columns:
        df['label_cpc'] = df['cpc_futuro'].map({0: "Não CPC", 1: "CPC"}).fillna("Desconhecido")
    else:
        df['label_cpc'] = "Desconhecido"
        df['cpc_futuro'] = 0
        
    return df

try:
    dados = carregar_dados()
except Exception as e:
    st.error(f"Erro ao ler o arquivo 'dataset_consolidado.parquet'. Você já rodou o script de consolidação? Detalhe: {e}")
    st.stop()

cores = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000", "#A9D18E"]

st.title("Projeto_SAN — Martech 2026")

@st.cache_data
def carregar_scores():
    """Lê o arquivo de scores gerado pelo pipeline diário."""
    if not __import__('os').path.exists("dataset_scored_diario_top5.parquet"):
        return None
    df = pd.read_parquet("dataset_scored_diario_top5.parquet")
    return df

df_scores = carregar_scores()

# Abas principais
aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs(["📋 Visão Geral", "📞 Telefones", "📊 Taxa de CPC", "🔍 Outliers", "📈 Estatísticas", "🏆 Ranking Top 5"])

# ==========================================
# ABA 1: VISÃO GERAL
# ==========================================
with aba1:
    st.header("Visão Geral")
    
    # Value Boxes
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de Registros", f"{len(dados):,}".replace(",", "."))
    col2.metric("Total de Colunas", dados.shape[1])
    
    pct_celular = (dados['tipo_tel'] == "Celular (11 dig)").mean()
    col3.metric("% Celulares", f"{pct_celular:.1%}")
    
    pct_cpc = dados['cpc_futuro'].mean()
    col4.metric("Taxa CPC Futuro", f"{pct_cpc:.1%}")
    
    n_dup = (dados['telefone'].value_counts() > 1).sum()
    col5.metric("Telefones Duplicados", f"{n_dup:,}".replace(",", "."))
    
    st.markdown("---")
    
    # Gráficos
    c1, c2, c3 = st.columns([2, 1, 1])
    
    with c1:
        st.subheader("Distribuição de Idade")
        df_idade = dados.dropna(subset=['idade'])
        mediana_idade = df_idade['idade'].median()
        
        fig = px.histogram(df_idade, x="idade", nbins=40, color_discrete_sequence=[cores[0]])
        fig.add_vline(x=mediana_idade, line_dash="dash", line_color="red", 
                      annotation_text=f"Mediana = {mediana_idade:.1f}")
        fig.update_layout(xaxis_title="Idade", yaxis_title="Frequência")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Target (cpc_futuro)")
        df_t = dados['label_cpc'].value_counts(normalize=True).reset_index()
        df_t.columns = ['label_cpc', 'pct']
        
        fig = px.bar(df_t, x='label_cpc', y='pct', text=df_t['pct'].apply(lambda x: f"{x:.1%}"),
                     color='label_cpc', color_discrete_sequence=["#ED7D31", "#70AD47"])
        fig.update_layout(xaxis_title="", yaxis_title="Proporção", showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
    with c3:
        st.subheader("Tipo de Telefone")
        df_tel = dados['tipo_tel'].value_counts().reset_index()
        df_tel.columns = ['tipo_tel', 'n']
        
        fig = px.pie(df_tel, names='tipo_tel', values='n', color_discrete_sequence=cores)
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# ABA 2: TELEFONES
# ==========================================
with aba2:
    c1, c2, c3 = st.columns([2, 1, 1])
    
    with c1:
        st.subheader("Distribuição por DDD (Top 20)")
        df_ddd = dados['ddd'].value_counts().reset_index().head(20)
        df_ddd.columns = ['ddd', 'n']
        df_ddd = df_ddd.sort_values('n', ascending=True) # Para barras horizontais
        
        fig = px.bar(df_ddd, x='n', y='ddd', orientation='h', color_discrete_sequence=[cores[0]],
                     labels={'n': 'Registros', 'ddd': 'DDD'})
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Telefones Duplicados")
        df_dup = dados['telefone'].value_counts().reset_index()
        df_dup.columns = ['telefone', 'ocorrencias']
        df_dup = df_dup[df_dup['ocorrencias'] > 1].head(50)
        st.dataframe(df_dup, use_container_width=True)
        
    with c3:
        st.subheader("Top 15 Estados")
        df_uf = dados['uf'].value_counts().reset_index().head(15)
        df_uf.columns = ['uf', 'n']
        df_uf = df_uf.sort_values('n', ascending=True)
        
        fig = px.bar(df_uf, x='n', y='uf', orientation='h', color_discrete_sequence=[cores[2]],
                     labels={'n': 'Registros', 'uf': ''})
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# ABA 3: TAXA DE CPC COM FILTROS
# ==========================================
with aba3:
    st.sidebar.header("Filtros de CPC")
    
    sexo_lista = ["Todos"] + sorted(dados['sexo'].dropna().unique().tolist())
    sexo_sel = st.sidebar.selectbox("Sexo:", sexo_lista)
    
    tipo_lista = ["Todos"] + sorted(dados['tipo_pessoa'].dropna().unique().tolist())
    tipo_sel = st.sidebar.selectbox("Tipo Pessoa:", tipo_lista)
    
    cart_lista = ["Todos"] + sorted(dados['desc_carteira'].dropna().unique().tolist())
    cart_sel = st.sidebar.selectbox("Carteira:", cart_lista)
    
    # Aplicação de Filtros
    df_filt = dados.copy()
    if sexo_sel != "Todos": df_filt = df_filt[df_filt['sexo'] == sexo_sel]
    if tipo_sel != "Todos": df_filt = df_filt[df_filt['tipo_pessoa'] == tipo_sel]
    if cart_sel != "Todos": df_filt = df_filt[df_filt['desc_carteira'] == cart_sel]
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribuição da Taxa de CPC (Total)")
        df_taxa = df_filt.dropna(subset=['taxa_cpc_total'])
        fig = px.histogram(df_taxa, x="taxa_cpc_total", nbins=30, color_discrete_sequence=[cores[2]])
        fig.update_layout(xaxis_title="Taxa CPC", yaxis_title="Frequência", xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Taxa CPC Média por Janela")
        medias = pd.DataFrame({
            'janela': ["Total", "30d", "60d", "90d"],
            'media': [
                df_filt['taxa_cpc_total'].mean(),
                df_filt['taxa_cpc_30d'].mean(),
                df_filt['taxa_cpc_60d'].mean(),
                df_filt['taxa_cpc_90d'].mean()
            ]
        })
        fig = px.bar(medias, x='janela', y='media', text=medias['media'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "0.0%"),
                     color_discrete_sequence=[cores[0]])
        fig.update_layout(yaxis_tickformat=".1%", yaxis_title="Taxa CPC média")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("CPC: Histórico x Futuro por Sexo")
        df_sexo = df_filt.groupby('sexo').agg(
            taxa_cpc_hist=('taxa_cpc_total', 'mean'),
            taxa_cpc_futuro=('cpc_futuro', 'mean')
        ).reset_index()
        
        df_sexo_melt = df_sexo.melt(id_vars='sexo', var_name='tipo', value_name='valor')
        fig = px.bar(df_sexo_melt, x='sexo', y='valor', color='tipo', barmode='group',
                     text=df_sexo_melt['valor'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "0.0%"),
                     color_discrete_sequence=[cores[0], cores[1]])
        fig.update_layout(yaxis_tickformat=".1%", yaxis_title="Taxa")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("CPC: Histórico x Futuro por Tipo de Pessoa")
        df_tipo = df_filt.groupby('tipo_pessoa').agg(
            taxa_cpc_hist=('taxa_cpc_total', 'mean'),
            taxa_cpc_futuro=('cpc_futuro', 'mean')
        ).reset_index()
        
        df_tipo_melt = df_tipo.melt(id_vars='tipo_pessoa', var_name='tipo', value_name='valor')
        fig = px.bar(df_tipo_melt, x='tipo_pessoa', y='valor', color='tipo', barmode='group',
                     text=df_tipo_melt['valor'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "0.0%"),
                     color_discrete_sequence=[cores[0], cores[1]])
        fig.update_layout(yaxis_tickformat=".1%", yaxis_title="Taxa")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# ABA 4: OUTLIERS
# ==========================================
with aba4:
    c1, c2 = st.columns([1, 1])
    
    cols_out = [
        "idade", "qnt_acionamentos_total", "qnt_alo_total",
        "qnt_cpc_total", "qnt_produtivo_total", "qnt_promessa_total",
        "qnt_desconhece_total", "qnt_caixa_postal_total", "qnt_queda_ligacao_total"
    ]
    # Filtrar apenas as colunas que existem no dataframe
    cols_out = [c for c in cols_out if c in dados.columns]
    
    with c1:
        st.subheader("Resumo de Outliers (Método IQR)")
        
        res_outliers = []
        for col in cols_out:
            s = dados[col].dropna()
            if len(s) == 0: continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            inf = q1 - 1.5 * iqr
            sup = q3 + 1.5 * iqr
            n_out = ((s < inf) | (s > sup)).sum()
            
            res_outliers.append({
                "Variável": col,
                "Q1": round(q1, 1),
                "Q3": round(q3, 1),
                "Limite_Inf": round(inf, 1),
                "Limite_Sup": round(sup, 1),
                "N_Outliers": n_out,
                "Pct_Outliers": f"{(n_out / len(s) * 100):.2f}%"
            })
            
        df_outliers = pd.DataFrame(res_outliers)
        st.dataframe(df_outliers, use_container_width=True)
        
        st.subheader("Idades Suspeitas (< 18 ou > 100)")
        df_idade_susp = dados[(dados['idade'] < 18) | (dados['idade'] > 100)]
        col_disp = [c for c in ['id_pessoa', 'telefone', 'idade', 'sexo', 'tipo_pessoa', 'uf', 'desc_carteira'] if c in dados.columns]
        st.dataframe(df_idade_susp[col_disp])
        
    with c2:
        st.subheader("Boxplot — Selecione a Variável")
        var_box = st.selectbox("Selecione:", cols_out)
        
        if var_box:
            df_box = dados[['label_cpc', var_box]].dropna()
            if not df_box.empty:
                p99 = df_box[var_box].quantile(0.99)
                fig = px.box(df_box, x="label_cpc", y=var_box, color="label_cpc",
                             color_discrete_map={"Não CPC": "#ED7D31", "CPC": "#70AD47"})
                # Truncando no p99 para facilitar a visualização (semelhante ao R)
                fig.update_yaxes(range=[0, p99 * 1.1])
                st.plotly_chart(fig, use_container_width=True)


# ==========================================
# ABA 5: ESTATÍSTICAS
# ==========================================
with aba5:
    st.subheader("Estatísticas Descritivas Completas")
    
    num_cols = dados.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_list = []
    for col in num_cols:
        s = dados[col]
        s_val = s.dropna()
        if len(s_val) == 0: continue
        
        stats_list.append({
            "Variável": col,
            "N": len(s_val),
            "Média": round(s_val.mean(), 3),
            "Mediana": round(s_val.median(), 3),
            "DP": round(s_val.std(), 3),
            "Mínimo": round(s_val.min(), 3),
            "Máximo": round(s_val.max(), 3),
            "Q1": round(s_val.quantile(0.25), 3),
            "Q3": round(s_val.quantile(0.75), 3),
            "Assimetria": round(skew(s_val), 3),
            "Curtose": round(kurtosis(s_val), 3),
            "N_Missing": s.isna().sum()
        })
        
    df_stats = pd.DataFrame(stats_list)
    
    # Aplica formatação condicional similar ao DataTables no R
    def highlight_assimetria(val):
        if val > 1 or val < -1:
            return 'background-color: #fde8e8'
        return ''
        
    def highlight_missing(val):
        if val > 0:
            return 'background-color: #fff3cd'
        return ''
        
    styled_stats = df_stats.style.applymap(highlight_assimetria, subset=['Assimetria'])\
                                 .applymap(highlight_missing, subset=['N_Missing'])
                                 
    st.dataframe(styled_stats, use_container_width=True, height=600)


# ==========================================
# ABA 6: RANKING TOP 5 (SPRINT 2 - TASK 6)
# ==========================================
with aba6:
    st.header("🏆 Ranking Top 5 — Scores de Conversão por Cliente")
    st.caption("Fonte: dataset_scored_diario_top5.parquet — gerado pelo Pipeline Diário (Sprint 2)")

    if df_scores is None:
        st.warning("Arquivo 'dataset_scored_diario_top5.parquet' não encontrado. Execute o pipeline_diario.py primeiro.", icon="⚠️")
    else:
        # KPIs
        n_clientes = df_scores['id_pessoa'].nunique()
        n_telefones = len(df_scores)
        score_medio = df_scores['score_probabilidade_conversao'].mean()
        pct_classe1 = df_scores['score_predicao_classe'].mean()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Clientes Únicos", f"{n_clientes:,}".replace(",", "."))
        kpi2.metric("Telefones no Ranking", f"{n_telefones:,}".replace(",", "."))
        kpi3.metric("Score Médio de Conversão", f"{score_medio:.2%}")
        kpi4.metric("% Classificados como Conversão", f"{pct_classe1:.2%}")

        st.markdown("---")

        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Distribuição dos Scores de Conversão")
            fig = px.histogram(
                df_scores, x="score_probabilidade_conversao", nbins=40,
                color_discrete_sequence=["#4472C4"],
                labels={"score_probabilidade_conversao": "Probabilidade de Conversão"}
            )
            fig.update_layout(xaxis_tickformat=".0%", yaxis_title="Frequência")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Score Médio por UF (Top 15)")
            if 'uf' in df_scores.columns:
                df_uf_score = (
                    df_scores.groupby('uf')['score_probabilidade_conversao']
                    .mean()
                    .reset_index()
                    .sort_values('score_probabilidade_conversao', ascending=False)
                    .head(15)
                    .sort_values('score_probabilidade_conversao', ascending=True)
                )
                fig = px.bar(
                    df_uf_score, x='score_probabilidade_conversao', y='uf',
                    orientation='h', color_discrete_sequence=["#70AD47"],
                    labels={"score_probabilidade_conversao": "Score Médio", "uf": "UF"}
                )
                fig.update_layout(xaxis_tickformat=".1%")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Telefones por Posição no Ranking (1º ao 5º melhor)")
        if 'id_pessoa' in df_scores.columns:
            df_scores_ranked = df_scores.copy()
            df_scores_ranked['rank_telefone'] = df_scores_ranked.groupby('id_pessoa')['score_probabilidade_conversao'].rank(method='first', ascending=False).astype(int)
            df_rank_dist = df_scores_ranked['rank_telefone'].value_counts().reset_index().sort_values('rank_telefone')
            df_rank_dist.columns = ['Posição no Ranking', 'Qtd de Telefones']
            fig = px.bar(
                df_rank_dist, x='Posição no Ranking', y='Qtd de Telefones',
                color_discrete_sequence=["#FFC000"],
                text='Qtd de Telefones'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🔎 Buscar Cliente por ID")
        id_busca = st.text_input("Digite o ID do cliente (id_pessoa):")
        if id_busca:
            df_cliente = df_scores[df_scores['id_pessoa'].astype(str) == id_busca.strip()]
            if df_cliente.empty:
                st.info("Nenhum cliente encontrado com esse ID.")
            else:
                st.success(f"Encontrado! {len(df_cliente)} telefone(s) no Top 5 para esse cliente.")
                cols_show = ['telefone', 'score_probabilidade_conversao', 'score_predicao_classe']
                cols_show = [c for c in cols_show if c in df_cliente.columns]
                st.dataframe(df_cliente[cols_show].reset_index(drop=True), use_container_width=True)
