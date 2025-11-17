import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Optional
from backend.config import ESTADOS_BRASIL


class OpenMeteoClient:
    #Cliente para Open-Meteo API

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self):
        self.session = requests.Session()

    def buscar_dados_climaticos(self, latitude: float, longitude: float,
                                data_inicio: str, data_fim: str) -> Optional[pd.DataFrame]:
        #Busca dados climáticos históricos

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
            response = self.session.get(self.BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                dados = response.json()

                if 'daily' not in dados:
                    st.error("❌ Resposta da API sem dados diários")
                    return None

                df = pd.DataFrame(dados['daily'])

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
        #Processa dados da API

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



def carregar_dados_openmeteo_estado(estado_nome: str, n_anos: int = 3) -> Optional[pd.DataFrame]:

    #Carrega dados climáticos REAIS do Open-Meteo


    if estado_nome not in ESTADOS_BRASIL:
        st.warning(f"⚠️ Estado '{estado_nome}' não encontrado")
        return None

    estado_info = ESTADOS_BRASIL[estado_nome]

    from datetime import datetime

    data_atual = datetime.now()
    ano_atual = data_atual.year
    ano_inicio = ano_atual - n_anos

    start_date = f"{ano_inicio}-01-01"
    end_date = data_atual.strftime('%Y-%m-%d')

    # Verificar qual chave existe
    if 'latitude' in estado_info:
        lat = estado_info['latitude']
        lon = estado_info['longitude']
    elif 'lat' in estado_info:
        lat = estado_info['lat']
        lon = estado_info['lon']
    else:
        st.error(f"❌ Coordenadas não encontradas para {estado_nome}")
        return None

    # Buscar dados
    client = OpenMeteoClient()

    df_clima = client.buscar_dados_climaticos(
        latitude=lat,
        longitude=lon,
        data_inicio=start_date,
        data_fim=end_date
    )

    if df_clima is None or len(df_clima) == 0:
        st.warning("⚠️ Falha ao obter dados climáticos do Open-Meteo")
        return None

    # Agregar por mês
    df_mensal = _agregar_dados_mensais(df_clima)

    if len(df_mensal) == 0:
        st.error("❌ Nenhum dado após agregação mensal")
        return None

    return df_mensal


def _agregar_dados_mensais(df: pd.DataFrame) -> pd.DataFrame:
    #Agrega dados diários em mensais

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