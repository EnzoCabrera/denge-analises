import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple


def adicionar_features_engenheiradas(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    # Amplitude térmica diária
    if 'temperatura_max' in df.columns and 'temperatura_min' in df.columns:
        df['amplitude_termica'] = df['temperatura_max'] - df['temperatura_min']

    # Temperatura ideal para Aedes aegypti (25-30°C)
    if 'temperatura_media' in df.columns:
        df['temp_ideal_aedes'] = df['temperatura_media'].apply(
            lambda x: 1 if 25 <= x <= 30 else 0
        )

        # Desvio da temperatura ideal (27.5°C)
        df['desvio_temp_ideal'] = np.abs(df['temperatura_media'] - 27.5)

        # Temperatura acima de 20°C (limite mínimo para reprodução)
        df['temp_acima_20'] = (df['temperatura_media'] > 20).astype(int)


    if 'umidade_relativa' in df.columns and 'precipitacao' in df.columns:
        # Índice combinado umidade × chuva (normalizado)
        df['indice_umidade_chuva'] = (
                (df['umidade_relativa'] / 100) *
                np.log1p(df['precipitacao'])  # log para reduzir impacto de outliers
        )

    if 'umidade_relativa' in df.columns:
        # Umidade alta (>70% é favorável)
        df['umidade_alta'] = (df['umidade_relativa'] > 70).astype(int)

        # Umidade ideal (70-90%)
        df['umidade_ideal'] = df['umidade_relativa'].apply(
            lambda x: 1 if 70 <= x <= 90 else 0
        )

    if 'precipitacao' in df.columns:
        # Chuva moderada (50-150mm - ideal para criadouros)
        df['chuva_moderada'] = df['precipitacao'].apply(
            lambda x: 1 if 50 <= x <= 150 else 0
        )

        # Chuva intensa (>150mm)
        df['chuva_intensa'] = (df['precipitacao'] > 150).astype(int)

        # Sem chuva (<10mm)
        df['sem_chuva'] = (df['precipitacao'] < 10).astype(int)


    if 'mes' in df.columns:
        # Estação do ano (hemisfério sul)
        def obter_estacao(mes):
            if mes in [12, 1, 2]:
                return 'Verao'
            elif mes in [3, 4, 5]:
                return 'Outono'
            elif mes in [6, 7, 8]:
                return 'Inverno'
            else:
                return 'Primavera'

        df['estacao'] = df['mes'].apply(obter_estacao)

        # Estação favorável (verão/primavera)
        df['estacao_favoravel'] = df['estacao'].isin(['Verao', 'Primavera']).astype(int)

        # Trimestre
        df['trimestre'] = ((df['mes'] - 1) // 3 + 1)

        # Meses de pico (fevereiro, março, abril)
        df['mes_pico'] = df['mes'].isin([2, 3, 4]).astype(int)

        # Componentes cíclicos do mês (captura sazonalidade)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)



    if 'casos_dengue' in df.columns:
        # Ordenar por ano e mês para garantir sequência temporal
        if 'ano' in df.columns and 'mes' in df.columns:
            df = df.sort_values(['ano', 'mes']).reset_index(drop=True)

        # Casos do mês anterior (feature MAIS IMPORTANTE!)
        df['casos_lag1'] = df['casos_dengue'].shift(1).fillna(0)

        # Casos de 2 meses atrás
        df['casos_lag2'] = df['casos_dengue'].shift(2).fillna(0)

        # Casos de 3 meses atrás
        df['casos_lag3'] = df['casos_dengue'].shift(3).fillna(0)

        # Média móvel dos últimos 3 meses
        df['media_movel_3m'] = df['casos_dengue'].rolling(window=3, min_periods=1).mean()

        # Média móvel dos últimos 6 meses
        df['media_movel_6m'] = df['casos_dengue'].rolling(window=6, min_periods=1).mean()

        # Tendência (diferença entre mês atual e anterior)
        df['tendencia'] = df['casos_dengue'].diff().fillna(0)

        # Taxa de crescimento (% mudança)
        df['taxa_crescimento'] = df['casos_dengue'].pct_change().fillna(0)
        df['taxa_crescimento'] = df['taxa_crescimento'].replace([np.inf, -np.inf], 0)


    if 'temperatura_media' in df.columns and 'umidade_relativa' in df.columns:
        # Temperatura × Umidade (condição ideal combinada)
        df['temp_x_umidade'] = df['temperatura_media'] * (df['umidade_relativa'] / 100)

    if 'temperatura_media' in df.columns and 'precipitacao' in df.columns:
        # Temperatura × Precipitação
        df['temp_x_precip'] = df['temperatura_media'] * np.log1p(df['precipitacao'])

    # Score de risco climático composto
    score = 0

    if 'temp_ideal_aedes' in df.columns:
        score += df['temp_ideal_aedes'] * 3
    if 'umidade_ideal' in df.columns:
        score += df['umidade_ideal'] * 2
    if 'chuva_moderada' in df.columns:
        score += df['chuva_moderada'] * 2
    if 'estacao_favoravel' in df.columns:
        score += df['estacao_favoravel'] * 1

    if isinstance(score, pd.Series):
        df['score_climatico'] = score

    # Índice de favorabilidade (0-1)
    if all(col in df.columns for col in ['temp_ideal_aedes', 'umidade_ideal', 'chuva_moderada', 'estacao_favoravel']):
        df['indice_favorabilidade'] = (
                (df['temp_ideal_aedes'] + df['umidade_ideal'] +
                 df['chuva_moderada'] + df['estacao_favoravel']) / 4
        )


    if 'temperatura_media' in df.columns:
        # Desvio padrão da temperatura (últimos 3 meses)
        df['temp_std_3m'] = df['temperatura_media'].rolling(window=3, min_periods=1).std().fillna(0)

    if 'umidade_relativa' in df.columns:
        # Variação de umidade (últimos 3 meses)
        df['umidade_std_3m'] = df['umidade_relativa'].rolling(window=3, min_periods=1).std().fillna(0)

    if 'precipitacao' in df.columns:
        # Total de chuva acumulada (últimos 3 meses)
        df['precip_acum_3m'] = df['precipitacao'].rolling(window=3, min_periods=1).sum()

        # Variabilidade da chuva
        df['precip_std_3m'] = df['precipitacao'].rolling(window=3, min_periods=1).std().fillna(0)

    return df


def selecionar_features_relevantes(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:

    # Features a serem usadas (em ordem de importância estimada)
    features_prioritarias = [
        # LAG FEATURES (MUITO IMPORTANTES - histórico de casos)
        'casos_lag1',  # Casos do mês anterior
        'casos_lag2',  # Casos de 2 meses atrás
        'media_movel_3m',  # Média móvel 3 meses
        'tendencia',  # Tendência de crescimento
        'taxa_crescimento',  # Taxa de crescimento percentual

        # TEMPERATURA (importantes)
        'temperatura_media',
        'temperatura_max',
        'temperatura_min',
        'temp_ideal_aedes',  # Binary: temperatura ideal?
        'desvio_temp_ideal',  # Quão longe do ideal
        'amplitude_termica',  # Variação diária
        'temp_acima_20',  # Acima do mínimo para reprodução

        # UMIDADE E CHUVA (importantes)
        'umidade_relativa',
        'precipitacao',
        'umidade_alta',  # Binary: umidade > 70%
        'umidade_ideal',  # Binary: umidade 70-90%
        'chuva_moderada',  # Binary: chuva ideal para criadouros
        'indice_umidade_chuva',  # Interação umidade × chuva

        # TEMPORAIS (sazonalidade)
        'mes',
        'trimestre',
        'estacao_favoravel',  # Binary: verão/primavera
        'mes_pico',  # Binary: meses de pico histórico
        'mes_sin',  # Componente sazonal (seno)
        'mes_cos',  # Componente sazonal (cosseno)

        # INTERAÇÕES E SCORES
        'temp_x_umidade',  # Interação temperatura × umidade
        'score_climatico',  # Score composto de risco
        'indice_favorabilidade',  # Índice de condições favoráveis

        # ESTATÍSTICAS (variabilidade)
        'temp_std_3m',  # Variabilidade da temperatura
        'precip_acum_3m'  # Chuva acumulada 3 meses
    ]

    # Verificar quais features existem no DataFrame
    features_disponiveis = [f for f in features_prioritarias if f in df.columns]

    # Logging
    total_disponiveis = len(features_disponiveis)
    total_criadas = len([col for col in df.columns if col not in
                         ['ano', 'mes', 'casos_dengue', 'risco_dengue', 'ano_mes', 'mes_nome']])

    # Retornar apenas as colunas disponíveis
    X = df[features_disponiveis].copy()

    return X, features_disponiveis


def validar_features(df: pd.DataFrame) -> bool:

    # Colunas mínimas necessárias
    colunas_minimas = ['temperatura_media', 'umidade_relativa', 'precipitacao', 'mes']

    for col in colunas_minimas:
        if col not in df.columns:
            st.error(f"❌ Coluna obrigatória ausente: {col}")
            return False

    # Verificar se tem casos de dengue
    if 'casos_dengue' not in df.columns:
        st.warning("⚠️ Coluna 'casos_dengue' não encontrada. Features de lag não serão criadas.")

    return True