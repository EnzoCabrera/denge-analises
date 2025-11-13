import pandas as pd
import numpy as np
import streamlit as st
from backend.config import ESTADOS_BRASIL, PARAMETROS_CLIMA, MESES_NOMES
from backend.api_openmeteo import carregar_dados_openmeteo_estado, simular_casos_dengue
from backend.api_infodengue import carregar_dados_infodengue_estado, classificar_risco_casos_reais


@st.cache_data
def gerar_dados_estado(estado_nome: str, n_anos: int = 3,
                       usar_dados_reais: bool = True) -> pd.DataFrame:

    #Gera ou carrega dados para um estado
    #PRIORIDADE: InfoDengue (casos reais) + Open-Meteo (clima real)


    estado_info = ESTADOS_BRASIL[estado_nome]
    regiao = estado_info['regiao']

    if usar_dados_reais:
        st.info(f"🌐 Tentando carregar dados REAIS para {estado_nome}...")

        # 1. Buscar CLIMA REAL (Open-Meteo)
        df_clima = carregar_dados_openmeteo_estado(estado_nome, n_anos)

        # 2. Buscar CASOS REAIS (InfoDengue)
        df_casos = carregar_dados_infodengue_estado(estado_nome, n_anos)

        # 3. Se ambos forem obtidos com sucesso, COMBINAR
        if df_clima is not None and df_casos is not None:
            st.success(f"✅ Dados 100% REAIS obtidos! Clima: Open-Meteo | Casos: InfoDengue")

            df = _combinar_clima_e_casos_reais(df_clima, df_casos, estado_info)

            # Adicionar informações do estado
            df['estado'] = estado_nome
            df['sigla'] = estado_info['sigla']
            df['regiao'] = regiao
            df['latitude'] = estado_info['lat']
            df['longitude'] = estado_info['lon']

            return df

        # 4. Se só conseguiu CLIMA, simular casos baseado no clima real
        elif df_clima is not None:
            st.warning(f"⚠️ Usando clima REAL (Open-Meteo) + casos SIMULADOS")

            df = simular_casos_dengue(df_clima, regiao)
            df['estado'] = estado_nome
            df['sigla'] = estado_info['sigla']
            df['regiao'] = regiao
            df['latitude'] = estado_info['lat']
            df['longitude'] = estado_info['lon']

            return df

    st.warning(f"⚠️ Usando dados 100% SIMULADOS para {estado_nome}")
    return _gerar_dados_simulados(estado_nome, n_anos)


def _combinar_clima_e_casos_reais(df_clima: pd.DataFrame,
                                  df_casos: pd.DataFrame,
                                  estado_info: dict) -> pd.DataFrame:

    #Combina dados REAIS de clima e casos


    # Fazer merge por ano/mês
    df = pd.merge(
        df_clima,
        df_casos[['ano', 'mes', 'casos_dengue']],
        on=['ano', 'mes'],
        how='left'
    )

    # Preencher casos faltantes com 0
    df['casos_dengue'] = df['casos_dengue'].fillna(0).astype(int)

    # Classificar risco baseado em casos REAIS
    df['risco_dengue'] = df['casos_dengue'].apply(classificar_risco_casos_reais)

    # Adicionar nomes de meses e ano_mes (se não existir)
    if 'mes_nome' not in df.columns:
        df['mes_nome'] = df['mes'].apply(lambda x: MESES_NOMES[x - 1])
    if 'ano_mes' not in df.columns:
        df['ano_mes'] = df.apply(lambda row: f"{row['ano']}-{row['mes']:02d}", axis=1)

    return df


def _gerar_dados_simulados(estado_nome: str, n_anos: int) -> pd.DataFrame:

    #Gera dados completamente simulados (fallback)

    np.random.seed(42)

    estado_info = ESTADOS_BRASIL[estado_nome]
    regiao = estado_info['regiao']
    params = PARAMETROS_CLIMA[regiao]

    dados = []
    anos = list(range(2022, 2022 + n_anos))

    for ano in anos:
        for mes in range(1, 13):

            fator_temp, fator_chuva, _ = _calcular_fatores_sazonais(mes)
            fator_ano = 1 + (ano - 2022) * 0.15

            temp = np.random.normal(params['temp_base'] * fator_temp, params['temp_var'] * 0.8)
            umidade = np.random.normal(params['umidade_base'], params['umidade_var'] * 0.7)
            precip = np.random.normal(params['precip_base'] * fator_chuva, params['precip_var'] * 0.6)

            temp = np.clip(temp, 10, 40)
            umidade = np.clip(umidade, 30, 100)
            precip = np.clip(precip, 0, 500)

            score_risco = _calcular_score_risco(temp, umidade, precip, mes)

            if score_risco >= 7:
                risco = 'Alto'
                casos_base = params['casos_base'] * 0.5
            elif score_risco >= 4:
                risco = 'Médio'
                casos_base = params['casos_base'] * 0.3
            else:
                risco = 'Baixo'
                casos_base = params['casos_base'] * 0.1

            casos = int(np.random.normal(casos_base * fator_ano, casos_base * 0.25))
            casos = max(100, casos)

            dados.append({
                'ano': ano,
                'mes': mes,
                'mes_nome': MESES_NOMES[mes - 1],
                'ano_mes': f"{ano}-{mes:02d}",
                'temperatura_media': round(temp, 1),
                'temperatura_max': round(temp + np.random.uniform(3, 8), 1),
                'temperatura_min': round(temp - np.random.uniform(3, 8), 1),
                'umidade_relativa': round(umidade, 1),
                'precipitacao': round(precip, 1),
                'casos_dengue': casos,
                'risco_dengue': risco,
                'estado': estado_nome,
                'sigla': estado_info['sigla'],
                'regiao': regiao,
                'latitude': estado_info['lat'],
                'longitude': estado_info['lon']
            })

    return pd.DataFrame(dados)


def _calcular_score_risco(temp: float, umidade: float, precip: float, mes: int) -> int:
    #Calcula score de risco
    score = 0

    if 25 <= temp <= 30:
        score += 3
    elif 20 <= temp <= 35:
        score += 2
    elif temp > 18:
        score += 1

    if umidade > 80:
        score += 3
    elif umidade > 70:
        score += 2
    elif umidade > 60:
        score += 1

    if precip > 150:
        score += 3
    elif precip > 100:
        score += 2
    elif precip > 50:
        score += 1

    if mes in [12, 1, 2, 3]:
        score += 2
    elif mes in [10, 11]:
        score += 1

    return score


def _calcular_fatores_sazonais(mes: int) -> tuple:
    #Calcula fatores sazonais
    if mes in [12, 1, 2]:
        return 1.15, 1.5, 2.5
    elif mes in [3, 4]:
        return 1.05, 1.3, 1.8
    elif mes in [6, 7, 8]:
        return 0.80, 0.5, 0.4
    else:
        return 0.95, 0.9, 1.2


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