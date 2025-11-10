import pandas as pd
import numpy as np
import streamlit as st
from backend.config import ESTADOS_BRASIL, PARAMETROS_CLIMA, MESES_NOMES
from backend.api_openmeteo import carregar_dados_openmeteo_estado, simular_casos_dengue


@st.cache_data
def gerar_dados_estado(estado_nome: str, n_anos: int = 3,
                       usar_dados_reais: bool = True) -> pd.DataFrame:
    #Gera ou carrega dados para um estado

    estado_info = ESTADOS_BRASIL[estado_nome]
    regiao = estado_info['regiao']

    # TENTAR CARREGAR DADOS REAIS DO OPEN-METEO
    if usar_dados_reais:
        st.info(f"🌐 Carregando dados REAIS do Open-Meteo para {estado_nome}...")

        df_openmeteo = carregar_dados_openmeteo_estado(estado_nome, n_anos)

        if df_openmeteo is not None and len(df_openmeteo) > 0:
            st.success(f"✅ Dados REAIS carregados! {len(df_openmeteo)} meses de Open-Meteo")

            # Simular casos de dengue baseado no clima REAL
            df = simular_casos_dengue(df_openmeteo, regiao)

            # Adicionar informações do estado
            df['estado'] = estado_nome
            df['sigla'] = estado_info['sigla']
            df['regiao'] = regiao
            df['latitude'] = estado_info['lat']
            df['longitude'] = estado_info['lon']

            # Multiplicar dados (simular múltiplas localidades)
            df = _multiplicar_amostras(df, n_anos)

            return df

    # FALLBACK: Dados simulados
    st.warning(f"⚠️ Usando dados SIMULADOS para {estado_nome}")
    return _gerar_dados_simulados(estado_nome, n_anos)


def _multiplicar_amostras(df: pd.DataFrame, n_anos: int) -> pd.DataFrame:

    #Multiplica amostras para ter dados suficientes para ML


    amostras_por_mes = max(20, int(150 / n_anos))
    dados_expandidos = []

    for _, row in df.iterrows():
        for i in range(amostras_por_mes):
            variacao = np.random.uniform(0.90, 1.10)

            nova_linha = row.copy()
            nova_linha['temperatura_media'] *= variacao
            nova_linha['temperatura_max'] *= variacao
            nova_linha['temperatura_min'] *= variacao
            nova_linha['umidade_relativa'] *= np.random.uniform(0.95, 1.05)
            nova_linha['precipitacao'] *= np.random.uniform(0.85, 1.15)
            nova_linha['casos_dengue'] = int(nova_linha['casos_dengue'] * variacao)

            # Recalcular risco
            temp = nova_linha['temperatura_media']
            umidade = nova_linha['umidade_relativa']
            precip = nova_linha['precipitacao']

            score = 0
            if 25 <= temp <= 30: score += 3
            elif 20 <= temp <= 35: score += 2
            elif temp > 18: score += 1

            if umidade > 80: score += 3
            elif umidade > 70: score += 2
            elif umidade > 60: score += 1

            if precip > 150: score += 3
            elif precip > 100: score += 2
            elif precip > 50: score += 1

            if score >= 7:
                nova_linha['risco_dengue'] = 'Alto'
            elif score >= 4:
                nova_linha['risco_dengue'] = 'Médio'
            else:
                nova_linha['risco_dengue'] = 'Baixo'

            dados_expandidos.append(nova_linha)

    return pd.DataFrame(dados_expandidos)


def _gerar_dados_simulados(estado_nome: str, n_anos: int) -> pd.DataFrame:
    #Gera dados completamente simulados (fallback)

    np.random.seed(42)

    estado_info = ESTADOS_BRASIL[estado_nome]
    regiao = estado_info['regiao']
    params = PARAMETROS_CLIMA[regiao]

    dados = []
    anos = list(range(2022, 2022 + n_anos))
    amostras_por_mes = max(30, int(200 / n_anos))

    for ano in anos:
        for mes in range(1, 13):
            for _ in range(amostras_por_mes):

                fator_temp, fator_chuva, _ = _calcular_fatores_sazonais(mes)
                fator_ano = 1 + (ano - 2022) * 0.15
                variacao_local = np.random.uniform(0.85, 1.15)

                temp = np.random.normal(params['temp_base'] * fator_temp * variacao_local,
                                       params['temp_var'] * 0.8)
                umidade = np.random.normal(params['umidade_base'] * variacao_local,
                                          params['umidade_var'] * 0.7)
                precip = np.random.normal(params['precip_base'] * fator_chuva * variacao_local,
                                         params['precip_var'] * 0.6)

                temp = np.clip(temp, 10, 40)
                umidade = np.clip(umidade, 30, 100)
                precip = np.clip(precip, 0, 500)

                score_risco = _calcular_score_risco(temp, umidade, precip, mes)

                if score_risco >= 7:
                    risco = 'Alto'
                    casos_base = params['casos_base'] * 2.0
                elif score_risco >= 4:
                    risco = 'Médio'
                    casos_base = params['casos_base'] * 1.0
                else:
                    risco = 'Baixo'
                    casos_base = params['casos_base'] * 0.3

                casos = int(np.random.normal(casos_base * fator_ano * variacao_local,
                                            casos_base * 0.25))
                casos = max(5, casos)

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
                    'latitude': estado_info['lat'] + np.random.uniform(-1, 1),
                    'longitude': estado_info['lon'] + np.random.uniform(-1, 1)
                })

    return pd.DataFrame(dados)


def _calcular_score_risco(temp: float, umidade: float, precip: float, mes: int) -> int:
    #Calcula score de risco
    score = 0

    if 25 <= temp <= 30: score += 3
    elif 20 <= temp <= 35: score += 2
    elif temp > 18: score += 1

    if umidade > 80: score += 3
    elif umidade > 70: score += 2
    elif umidade > 60: score += 1

    if precip > 150: score += 3
    elif precip > 100: score += 2
    elif precip > 50: score += 1

    if mes in [12, 1, 2, 3]: score += 2
    elif mes in [10, 11]: score += 1

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
    df_agregado = df.groupby(['ano', 'mes', 'mes_nome', 'ano_mes']).agg({
        'casos_dengue': 'sum',
        'temperatura_media': 'mean',
        'umidade_relativa': 'mean',
        'precipitacao': 'mean',
        'risco_dengue': lambda x: x.mode()[0] if len(x) > 0 else 'Médio'
    }).reset_index()

    return {
        'total_casos': int(df['casos_dengue'].sum()),
        'media_temp': float(df['temperatura_media'].mean()),
        'media_umidade': float(df['umidade_relativa'].mean()),
        'media_precipitacao': float(df['precipitacao'].mean()),
        'casos_ultimo_mes': int(df_agregado.iloc[-1]['casos_dengue']) if len(df_agregado) > 0 else 0,
        'risco_predominante': df['risco_dengue'].mode()[0] if len(df) > 0 else 'Médio',
        'total_registros': len(df),
        'distribuicao_risco': df['risco_dengue'].value_counts().to_dict()
    }