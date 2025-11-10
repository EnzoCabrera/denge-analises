"""
Módulo para integração com Open-Meteo API
Dados climáticos históricos gratuitos e confiáveis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Optional
from backend.config import ESTADOS_BRASIL


class OpenMeteoClient:
    """Cliente para Open-Meteo API"""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self):
        self.session = requests.Session()

    def buscar_dados_climaticos(self, latitude: float, longitude: float,
                                data_inicio: str, data_fim: str) -> Optional[pd.DataFrame]:
        """
        Busca dados climáticos históricos

        Args:
            latitude: Latitude do local
            longitude: Longitude do local
            data_inicio: Data inicial (YYYY-MM-DD)
            data_fim: Data final (YYYY-MM-DD)

        Returns:
            DataFrame com dados climáticos
        """

        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': data_inicio,
            'end_date': data_fim,
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'temperature_2m_mean',
                'precipitation_sum',
                'rain_sum',
                'windspeed_10m_max',
                'relative_humidity_2m_mean'
            ],
            'timezone': 'America/Sao_Paulo'
        }

        try:
            st.info(f"🌍 Buscando dados climáticos: {data_inicio} até {data_fim}")

            response = self.session.get(self.BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                dados = response.json()

                if 'daily' not in dados:
                    st.error("❌ Resposta da API sem dados diários")
                    return None

                df = pd.DataFrame(dados['daily'])

                st.success(f"✅ {len(df)} dias de dados climáticos obtidos")

                return self._processar_dados(df)

            else:
                st.error(f"❌ Erro na API: Status {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout ao conectar com Open-Meteo")
            return None

        except Exception as e:
            st.error(f"❌ Erro ao buscar dados: {str(e)}")
            return None

    def _processar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa dados da API"""

        # Renomear colunas
        df.rename(columns={
            'time': 'data',
            'temperature_2m_max': 'temperatura_max',
            'temperature_2m_min': 'temperatura_min',
            'temperature_2m_mean': 'temperatura_media',
            'precipitation_sum': 'precipitacao',
            'rain_sum': 'chuva',
            'windspeed_10m_max': 'velocidade_vento',
            'relative_humidity_2m_mean': 'umidade_relativa'
        }, inplace=True)

        # Converter data
        df['data'] = pd.to_datetime(df['data'])

        # Preencher valores ausentes
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)

        return df


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados_openmeteo_estado(estado_nome: str, n_anos: int = 3) -> Optional[pd.DataFrame]:
    """
    Carrega dados climáticos do Open-Meteo

    Args:
        estado_nome: Nome do estado
        n_anos: Número de anos de histórico

    Returns:
        DataFrame com dados mensais
    """

    if estado_nome not in ESTADOS_BRASIL:
        st.error(f"❌ Estado {estado_nome} não encontrado")
        return None

    # Obter coordenadas da capital do estado
    estado_info = ESTADOS_BRASIL[estado_nome]
    latitude = estado_info['lat']
    longitude = estado_info['lon']

    # Calcular período
    data_fim = datetime(2025, 10, 30)  # Data fixa para consistência
    data_inicio = data_fim - timedelta(days=365 * n_anos)

    data_inicio_str = data_inicio.strftime('%Y-%m-%d')
    data_fim_str = data_fim.strftime('%Y-%m-%d')

    st.info(f"📍 Local: {estado_nome} (Lat: {latitude:.2f}, Lon: {longitude:.2f})")

    # Buscar dados
    client = OpenMeteoClient()
    df_clima = client.buscar_dados_climaticos(latitude, longitude,
                                              data_inicio_str, data_fim_str)

    if df_clima is None or len(df_clima) == 0:
        st.error("❌ Falha ao obter dados do Open-Meteo")
        return None

    # Agregar por mês
    df_mensal = _agregar_dados_mensais(df_clima)

    st.success(f"✅ {len(df_mensal)} meses de dados processados")

    return df_mensal


def _agregar_dados_mensais(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega dados diários em mensais"""

    df['ano'] = df['data'].dt.year
    df['mes'] = df['data'].dt.month

    # Agregação mensal
    df_mensal = df.groupby(['ano', 'mes']).agg({
        'temperatura_media': 'mean',
        'temperatura_max': 'max',
        'temperatura_min': 'min',
        'umidade_relativa': 'mean',
        'precipitacao': 'sum',
        'velocidade_vento': 'mean'
    }).reset_index()

    # Preencher NaN
    for col in df_mensal.select_dtypes(include=[np.number]).columns:
        df_mensal[col].fillna(df_mensal[col].median(), inplace=True)

    # Garantir valores positivos
    df_mensal['precipitacao'] = df_mensal['precipitacao'].clip(lower=0)
    df_mensal['umidade_relativa'] = df_mensal['umidade_relativa'].clip(0, 100)
    df_mensal['temperatura_media'] = df_mensal['temperatura_media'].clip(10, 45)

    return df_mensal


def simular_casos_dengue(df_clima: pd.DataFrame, regiao: str) -> pd.DataFrame:
    """Simula casos de dengue baseado em dados climáticos REAIS"""

    from backend.config import PARAMETROS_CLIMA, MESES_NOMES

    df = df_clima.copy()
    params = PARAMETROS_CLIMA.get(regiao, PARAMETROS_CLIMA['Sudeste'])

    casos_lista = []
    riscos_lista = []

    for _, row in df.iterrows():
        temp = row['temperatura_media']
        umidade = row['umidade_relativa']
        precip = row['precipitacao']
        mes = row['mes']

        # Calcular score de risco
        score = 0

        # Temperatura ótima: 25-30°C
        if 25 <= temp <= 30:
            score += 3
        elif 20 <= temp <= 35:
            score += 2
        elif temp > 18:
            score += 1

        # Umidade ideal: >70%
        if umidade > 80:
            score += 3
        elif umidade > 70:
            score += 2
        elif umidade > 60:
            score += 1

        # Precipitação
        if precip > 150:
            score += 3
        elif precip > 100:
            score += 2
        elif precip > 50:
            score += 1

        # Sazonalidade (verão/primavera)
        if mes in [12, 1, 2, 3]:
            score += 2
        elif mes in [10, 11]:
            score += 1

        # Classificar risco
        if score >= 7:
            risco = 'Alto'
            casos_base = params['casos_base'] * 2.0
        elif score >= 4:
            risco = 'Médio'
            casos_base = params['casos_base'] * 1.0
        else:
            risco = 'Baixo'
            casos_base = params['casos_base'] * 0.3

        # Gerar casos com variação
        casos = int(np.random.normal(casos_base, casos_base * 0.25))
        casos = max(10, casos)

        casos_lista.append(casos)
        riscos_lista.append(risco)

    df['casos_dengue'] = casos_lista
    df['risco_dengue'] = riscos_lista
    df['mes_nome'] = df['mes'].apply(lambda x: MESES_NOMES[x - 1])
    df['ano_mes'] = df.apply(lambda row: f"{row['ano']}-{row['mes']:02d}", axis=1)

    return df