import pandas as pd
import numpy as np
import streamlit as st
from backend.config import ESTADOS_BRASIL, PARAMETROS_CLIMA, MESES_NOMES
from backend.api_openmeteo import carregar_dados_openmeteo_estado
from backend.api_infodengue import carregar_dados_infodengue_estado, classificar_risco_casos_reais


def gerar_dados_estado(estado_nome: str, n_anos: int = 3,
                       usar_dados_reais: bool = True) -> pd.DataFrame:

    #Carrega dados REAIS para um estado
    #OBRIGATÓRIO: InfoDengue (casos reais) + Open-Meteo (clima real)


    estado_info = ESTADOS_BRASIL[estado_nome]
    regiao = estado_info['regiao']

    # Calcular período completo (janeiro do primeiro ano até dezembro do último)
    ano_atual = pd.Timestamp.now().year
    ano_inicio = ano_atual - n_anos + 1
    ano_fim = ano_atual

    # 1. Buscar CLIMA REAL (Open-Meteo) - OBRIGATÓRIO
    df_clima = carregar_dados_openmeteo_estado(estado_nome, n_anos)

    if df_clima is None or len(df_clima) == 0:
        st.error("❌ Falha ao carregar dados climáticos.")
        return pd.DataFrame()

    # FILTRAR apenas os anos solicitados (janeiro a dezembro)
    df_clima = df_clima[
        (df_clima['ano'] >= ano_inicio) &
        (df_clima['ano'] <= ano_fim)
        ].copy()

    # 2. Buscar CASOS REAIS (InfoDengue) - OBRIGATÓRIO
    df_casos = carregar_dados_infodengue_estado(estado_nome, n_anos)

    if df_casos is None or len(df_casos) == 0:
        st.error("❌ Falha ao carregar dados do InfoDengue.")
        return pd.DataFrame()

    # FILTRAR apenas os anos solicitados
    df_casos = df_casos[
        (df_casos['ano'] >= ano_inicio) &
        (df_casos['ano'] <= ano_fim)
        ].copy()

    # 3. COMBINAR dados reais (INNER JOIN)
    df = pd.merge(
        df_clima,
        df_casos[['ano', 'mes', 'casos_dengue']],
        on=['ano', 'mes'],
        how='inner'
    )

    if len(df) == 0:
        st.error("❌ Nenhuma correspondência entre clima e casos de dengue.")
        st.warning(f"🔍 Clima: {df_clima['ano'].unique()} | InfoDengue: {df_casos['ano'].unique()}")
        return pd.DataFrame()

    # Garantir que temos todos os meses do período
    anos_meses_esperados = []
    for ano in range(ano_inicio, ano_fim + 1):
        for mes in range(1, 13):
            # Se o ano for o atual, parar no mês atual
            if ano == ano_atual and mes > pd.Timestamp.now().month:
                break
            anos_meses_esperados.append((ano, mes))

    # Criar DataFrame de referência completo
    df_referencia = pd.DataFrame(anos_meses_esperados, columns=['ano', 'mes'])

    # Merge com referência para identificar meses faltantes
    df_completo = pd.merge(
        df_referencia,
        df,
        on=['ano', 'mes'],
        how='left',
        indicator=True
    )

    # Mostrar estatísticas de cobertura
    meses_com_dados = len(df_completo[df_completo['_merge'] == 'both'])
    meses_faltantes = len(df_completo[df_completo['_merge'] == 'left_only'])

    if meses_faltantes > 0:
        st.warning(f"⚠️ {meses_faltantes} meses sem dados em uma ou ambas as APIs")

    # Remover coluna auxiliar '_merge'
    df = df_completo[df_completo['_merge'] == 'both'].drop(columns=['_merge']).copy()

    if len(df) == 0:
        st.error("❌ Nenhum mês com dados completos disponível.")
        return pd.DataFrame()

    # Garantir que casos_dengue é int e sem NaN
    df['casos_dengue'] = df['casos_dengue'].fillna(0).astype(int)

    # Classificar risco baseado em casos REAIS
    df['risco_dengue'] = df['casos_dengue'].apply(classificar_risco_casos_reais)

    # Adicionar nomes de meses e ano_mes
    if 'mes_nome' not in df.columns:
        df['mes_nome'] = df['mes'].apply(lambda x: MESES_NOMES[int(x) - 1])
    if 'ano_mes' not in df.columns:
        df['ano_mes'] = df.apply(lambda row: f"{int(row['ano'])}-{int(row['mes']):02d}", axis=1)

    # Adicionar informações do estado
    df['estado'] = estado_nome
    df['sigla'] = estado_info['sigla']
    df['regiao'] = regiao
    df['latitude'] = estado_info['lat']
    df['longitude'] = estado_info['lon']

    # Ordenar por ano e mês
    df = df.sort_values(['ano', 'mes']).reset_index(drop=True)

    return df


def calcular_estatisticas(df: pd.DataFrame) -> dict:
    #Calcula estatísticas do dataset

    return {
        'total_casos': int(df['casos_dengue'].sum()),
        'media_temp': float(df['temperatura_media'].mean()),
        'media_umidade': float(df['umidade_relativa'].mean()),
        'media_precipitacao': float(df['precipitacao'].mean()),
        'casos_ultimo_mes': int(df.iloc[-1]['casos_dengue']) if len(df) > 0 else 0,
        'risco_predominante': df['risco_dengue'].mode()[0] if len(df) > 0 else 'Médio',
        'total_registros': len(df),
        'distribuicao_risco': df['risco_dengue'].value_counts().to_dict()
    }