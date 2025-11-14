import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Optional
from backend.config import ESTADOS_BRASIL, MESES_NOMES


# =====================================================
# GEOCÓDIGOS IBGE (Capitais dos Estados)
# =====================================================

GEOCODES_INFODENGUE = {
    'Acre': 1200401,           # Rio Branco
    'Alagoas': 2704302,        # Maceió
    'Amapá': 1600303,          # Macapá
    'Amazonas': 1302603,       # Manaus
    'Bahia': 2927408,          # Salvador
    'Ceará': 2304400,          # Fortaleza
    'Distrito Federal': 5300108,  # Brasília
    'Espírito Santo': 3205309, # Vitória
    'Goiás': 5208707,          # Goiânia
    'Maranhão': 2111300,       # São Luís
    'Mato Grosso': 5103403,    # Cuiabá
    'Mato Grosso do Sul': 5002704,  # Campo Grande
    'Minas Gerais': 3106200,   # Belo Horizonte
    'Pará': 1501402,           # Belém
    'Paraíba': 2507507,        # João Pessoa
    'Paraná': 4106902,         # Curitiba
    'Pernambuco': 2611606,     # Recife
    'Piauí': 2211001,          # Teresina
    'Rio de Janeiro': 3304557, # Rio de Janeiro
    'Rio Grande do Norte': 2408102,  # Natal
    'Rio Grande do Sul': 4314902,    # Porto Alegre
    'Rondônia': 1100205,       # Porto Velho
    'Roraima': 1400100,        # Boa Vista
    'Santa Catarina': 4205407, # Florianópolis
    'São Paulo': 3550308,      # São Paulo (Capital)
    'Sergipe': 2800308,        # Aracaju
    'Tocantins': 1721000       # Palmas
}


class InfoDengueClient:
    """Cliente para API do InfoDengue"""
    
    BASE_URL = "https://info.dengue.mat.br/api/alertcity"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
    
    def buscar_casos_dengue(self, geocode: int, ano_inicio: int, ano_fim: int) -> Optional[pd.DataFrame]:
        """
        Busca casos reais de dengue do InfoDengue
        
        Args:
            geocode: Código IBGE do município
            ano_inicio: Ano inicial
            ano_fim: Ano final
            
        Returns:
            DataFrame com casos reais ou None
        """
        
        try:
            params = {
                'geocode': geocode,
                'disease': 'dengue',
                'format': 'json',
                'ew_start': f'{ano_inicio}01',
                'ew_end': f'{ano_fim}53'
            }
            
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                dados = response.json()
                
                if not dados or len(dados) == 0:
                    return None
                
                df = pd.DataFrame(dados)
                return self._processar_dados_infodengue(df)
            
            else:
                return None
                
        except requests.exceptions.Timeout:
            st.warning("⏱️ Timeout ao conectar com InfoDengue")
            return None
        
        except Exception as e:
            st.warning(f"⚠️ Erro ao acessar InfoDengue: {str(e)}")
            return None
    
    def _processar_dados_infodengue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa dados brutos do InfoDengue
        
        Args:
            df: DataFrame bruto da API
            
        Returns:
            DataFrame processado
        """
        
        # Converter timestamp de milissegundos para datetime
        if 'data_iniSE' in df.columns:
            df['data'] = pd.to_datetime(df['data_iniSE'], unit='ms', errors='coerce')
        else:
            return pd.DataFrame()
        
        # Processar casos (prioridade: casos_est > casos)
        if 'casos_est' in df.columns:
            df['casos_dengue'] = pd.to_numeric(df['casos_est'], errors='coerce').fillna(0)
        elif 'casos' in df.columns:
            df['casos_dengue'] = pd.to_numeric(df['casos'], errors='coerce').fillna(0)
        else:
            return pd.DataFrame()
        
        # Extrair ano e mês
        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
        df['semana'] = df['SE']
        
        # Remover linhas inválidas
        df = df.dropna(subset=['ano', 'mes', 'casos_dengue'])
        
        # Filtrar apenas anos válidos (2020-2030)
        df = df[(df['ano'] >= 2020) & (df['ano'] <= 2030)]
        
        return df


@st.cache_data(ttl=86400, show_spinner=False)
def carregar_dados_infodengue_estado(estado_nome: str, n_anos: int = 3) -> Optional[pd.DataFrame]:
    """
    Carrega dados REAIS de dengue do InfoDengue
    Busca desde JANEIRO do ano inicial até o mês atual

    Args:
        estado_nome: Nome do estado
        n_anos: Número de anos para retroceder (ex: 2 = desde Jan/2023)

    Returns:
        DataFrame com casos REAIS agregados por mês ou None
    """

    if estado_nome not in GEOCODES_INFODENGUE:
        return None

    geocode = GEOCODES_INFODENGUE[estado_nome]

    # =====================================================
    # CORREÇÃO: Calcular ano inicial e buscar desde Janeiro
    # =====================================================

    from datetime import datetime

    data_atual = datetime.now()
    ano_atual = data_atual.year

    # Retroceder N anos
    ano_inicio = ano_atual - n_anos
    ano_fim = ano_atual

    st.info(f"🌐 Buscando dados REAIS de dengue para **{estado_nome}** (Capital)")
    st.info(f"📅 Período: **Janeiro/{ano_inicio}** a **{data_atual.strftime('%B/%Y')}**")
    st.info(f"🏙️ Geocode IBGE: **{geocode}**")

    # Buscar dados da API (pega ano completo)
    client = InfoDengueClient()
    df_casos = client.buscar_casos_dengue(geocode, ano_inicio, ano_fim)

    if df_casos is None or len(df_casos) == 0:
        return None

    # =====================================================
    # FILTRAR: Desde Janeiro do ano_inicio até hoje
    # =====================================================

    df_casos['data'] = pd.to_datetime(df_casos['data'])

    # Data de início: 1º de Janeiro do ano inicial
    from datetime import datetime
    data_inicio_periodo = datetime(ano_inicio, 1, 1)

    # Filtrar apenas registros dentro do período desejado
    df_casos = df_casos[
        (df_casos['data'] >= data_inicio_periodo) &
        (df_casos['data'] <= data_atual)
        ]

    # Agregar por mês
    df_mensal = _agregar_casos_por_mes(df_casos)

    if len(df_mensal) == 0:
        return None

    st.success(f"✅ {len(df_mensal)} meses de dados REAIS processados")
    st.success(f"📊 Período final: {df_mensal['ano_mes'].min()} a {df_mensal['ano_mes'].max()}")

    return df_mensal


def _agregar_casos_por_mes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega casos semanais em mensais
    
    Args:
        df: DataFrame com dados semanais
        
    Returns:
        DataFrame agregado por mês
    """
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Agrupar por ano e mês (somando casos das semanas)
    df_mensal = df.groupby(['ano', 'mes']).agg({
        'casos_dengue': 'sum'
    }).reset_index()
    
    # Adicionar informações extras
    df_mensal['mes_nome'] = df_mensal['mes'].apply(lambda x: MESES_NOMES[x - 1])
    df_mensal['ano_mes'] = df_mensal.apply(lambda row: f"{row['ano']}-{row['mes']:02d}", axis=1)
    
    # Ordenar
    df_mensal = df_mensal.sort_values(['ano', 'mes']).reset_index(drop=True)
    
    return df_mensal


def classificar_risco_casos_reais(casos: int) -> str:
    """
    Classifica risco baseado no número de casos REAIS
    
    Args:
        casos: Número de casos no mês
        
    Returns:
        'Alto', 'Médio' ou 'Baixo'
    """
    
    if casos > 5000:
        return 'Alto'
    elif casos > 1000:
        return 'Médio'
    else:
        return 'Baixo'


def obter_estatisticas_infodengue(df: pd.DataFrame) -> dict:
    """
    Calcula estatísticas dos dados do InfoDengue
    
    Args:
        df: DataFrame com dados mensais
        
    Returns:
        Dicionário com estatísticas
    """
    
    if df is None or len(df) == 0:
        return {}
    
    return {
        'total_casos': int(df['casos_dengue'].sum()),
        'media_casos_mes': float(df['casos_dengue'].mean()),
        'max_casos_mes': int(df['casos_dengue'].max()),
        'min_casos_mes': int(df['casos_dengue'].min()),
        'meses_disponveis': len(df),
        'periodo_inicio': f"{df['ano'].min()}-{df['mes'].min():02d}",
        'periodo_fim': f"{df['ano'].max()}-{df['mes'].max():02d}",
        'anos_disponiveis': df['ano'].nunique()
    }