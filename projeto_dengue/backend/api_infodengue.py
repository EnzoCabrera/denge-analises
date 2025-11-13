"""
Módulo para integração com InfoDengue API - VERSÃO DEBUG
Dados REAIS de casos de dengue no Brasil
Fonte: https://info.dengue.mat.br
"""

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
    'Acre': 1200401,  # Rio Branco
    'Alagoas': 2704302,  # Maceió
    'Amapá': 1600303,  # Macapá
    'Amazonas': 1302603,  # Manaus
    'Bahia': 2927408,  # Salvador
    'Ceará': 2304400,  # Fortaleza
    'Distrito Federal': 5300108,  # Brasília
    'Espírito Santo': 3205309,  # Vitória
    'Goiás': 5208707,  # Goiânia
    'Maranhão': 2111300,  # São Luís
    'Mato Grosso': 5103403,  # Cuiabá
    'Mato Grosso do Sul': 5002704,  # Campo Grande
    'Minas Gerais': 3106200,  # Belo Horizonte
    'Pará': 1501402,  # Belém
    'Paraíba': 2507507,  # João Pessoa
    'Paraná': 4106902,  # Curitiba
    'Pernambuco': 2611606,  # Recife
    'Piauí': 2211001,  # Teresina
    'Rio de Janeiro': 3304557,  # Rio de Janeiro
    'Rio Grande do Norte': 2408102,  # Natal
    'Rio Grande do Sul': 4314902,  # Porto Alegre
    'Rondônia': 1100205,  # Porto Velho
    'Roraima': 1400100,  # Boa Vista
    'Santa Catarina': 4205407,  # Florianópolis
    'São Paulo': 3550308,  # São Paulo (Capital)
    'Sergipe': 2800308,  # Aracaju
    'Tocantins': 1721000  # Palmas
}


class InfoDengueClient:
    """Cliente para API do InfoDengue com DEBUG"""

    BASE_URL = "https://info.dengue.mat.br/api/alertcity"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def buscar_casos_dengue(self, geocode: int, ano_inicio: int, ano_fim: int) -> Optional[pd.DataFrame]:
        """
        Busca casos reais de dengue do InfoDengue com DEBUG
        """

        try:
            params = {
                'geocode': geocode,
                'disease': 'dengue',
                'format': 'json',
                'ew_start': f'{ano_inicio}01',
                'ew_end': f'{ano_fim}53'
            }

            st.info(f"📡 **Requisição InfoDengue:**")
            st.code(f"""
URL: {self.BASE_URL}
Geocode: {geocode}
Período: {ano_inicio}-{ano_fim}
Doença: dengue
            """)

            response = self.session.get(self.BASE_URL, params=params, timeout=30)

            st.info(f"📥 **Status HTTP:** {response.status_code}")

            if response.status_code == 200:
                dados = response.json()

                st.info(f"📊 **Dados recebidos:** {len(dados)} registros")

                if not dados or len(dados) == 0:
                    st.error("❌ API retornou lista vazia")
                    return None

                # DEBUG: Mostrar primeiros 3 registros
                st.success("✅ Primeiros 3 registros da API:")
                st.json(dados[:3] if len(dados) >= 3 else dados)

                df = pd.DataFrame(dados)

                # DEBUG: Mostrar colunas disponíveis
                st.info(f"📋 **Colunas disponíveis:** {list(df.columns)}")

                # DEBUG: Mostrar primeiras linhas
                st.dataframe(df.head(3))

                return self._processar_dados_infodengue(df)

            elif response.status_code == 404:
                st.error("❌ Endpoint não encontrado (404). API pode ter mudado.")
                return None

            else:
                st.error(f"❌ Erro na API: Status {response.status_code}")
                st.text(response.text[:500])  # Primeiros 500 chars da resposta
                return None

        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout ao conectar com InfoDengue")
            return None

        except Exception as e:
            st.error(f"❌ Erro ao acessar InfoDengue: {str(e)}")
            st.exception(e)
            return None

    def _processar_dados_infodengue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa dados brutos do InfoDengue com correção de timestamp
        """

        st.info("🔄 **Processando dados...**")

        # =====================================================
        # CORREÇÃO: Timestamp em MILISSEGUNDOS
        # =====================================================

        # Converter data_iniSE (timestamp em milissegundos)
        if 'data_iniSE' in df.columns:
            # Converter de milissegundos para datetime
            df['data'] = pd.to_datetime(df['data_iniSE'], unit='ms', errors='coerce')
            st.success(f"✅ Data convertida de timestamp (ms)")
        else:
            st.error("❌ Coluna 'data_iniSE' não encontrada")
            return pd.DataFrame()

        # =====================================================
        # PROCESSAR CASOS
        # =====================================================

        # Ordem de prioridade para casos:
        # 1. casos_est (casos estimados - mais preciso)
        # 2. casos (casos notificados)
        # 3. notif_accum_year (acumulado do ano)

        if 'casos_est' in df.columns:
            df['casos_dengue'] = pd.to_numeric(df['casos_est'], errors='coerce').fillna(0)
            st.success("✅ Usando 'casos_est' (casos estimados)")
        elif 'casos' in df.columns:
            df['casos_dengue'] = pd.to_numeric(df['casos'], errors='coerce').fillna(0)
            st.success("✅ Usando 'casos' (casos notificados)")
        else:
            st.error("❌ Nenhuma coluna de casos encontrada!")
            return pd.DataFrame()

        # =====================================================
        # EXTRAIR ANO E MÊS (com data correta)
        # =====================================================

        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
        df['semana'] = df['SE']  # Semana epidemiológica

        # Remover linhas inválidas
        df = df.dropna(subset=['ano', 'mes', 'casos_dengue'])

        # Filtrar apenas anos válidos (2020-2025)
        df = df[(df['ano'] >= 2020) & (df['ano'] <= 2030)]

        # =====================================================
        # DEBUG: Mostrar estatísticas
        # =====================================================

        st.info(f"""
    📊 **Estatísticas dos dados processados:**
    - Total de registros: {len(df)}
    - Soma de casos: {df['casos_dengue'].sum():,.0f}
    - Média de casos/semana: {df['casos_dengue'].mean():.1f}
    - Período: {df['ano'].min()}-{df['ano'].max()}
    - Meses únicos: {df['mes'].nunique()}
        """)

        # Mostrar primeiras linhas processadas
        st.dataframe(df[['data', 'ano', 'mes', 'casos_dengue', 'SE']].head())

        return df


@st.cache_data(ttl=3600, show_spinner=False)  # Cache por 1h (reduzido para debug)
def carregar_dados_infodengue_estado(estado_nome: str, n_anos: int = 3) -> Optional[pd.DataFrame]:
    """
    Carrega dados REAIS de dengue do InfoDengue
    """

    if estado_nome not in GEOCODES_INFODENGUE:
        st.warning(f"⚠️ Dados do InfoDengue não disponíveis para {estado_nome}")
        return None

    geocode = GEOCODES_INFODENGUE[estado_nome]

    # Calcular período (usar ano atual real)
    ano_atual = datetime.now().year
    ano_fim = ano_atual
    ano_inicio = ano_fim - n_anos

    st.info(f"🌐 Buscando dados REAIS de dengue para **{estado_nome}** (Capital)")
    st.info(f"📅 Período: **{ano_inicio}** a **{ano_fim}**")
    st.info(f"🏙️ Geocode IBGE: **{geocode}**")

    # Buscar dados
    client = InfoDengueClient()
    df_casos = client.buscar_casos_dengue(geocode, ano_inicio, ano_fim)

    if df_casos is None or len(df_casos) == 0:
        st.error("❌ Falha ao obter dados do InfoDengue. Usando dados simulados...")
        return None

    # Agregar por mês
    df_mensal = _agregar_casos_por_mes(df_casos)

    if len(df_mensal) == 0:
        st.error("❌ Nenhum dado após agregação")
        return None

    st.success(f"✅ {len(df_mensal)} meses de dados REAIS processados")
    st.success(f"📊 Total de casos: {df_mensal['casos_dengue'].sum():,.0f}")

    return df_mensal


def _agregar_casos_por_mes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega casos semanais em mensais
    """

    st.info("📅 Agregando dados por mês...")

    # Verificar se tem dados
    if len(df) == 0:
        st.error("❌ DataFrame vazio para agregação")
        return pd.DataFrame()

    # Agrupar por ano e mês (SOMANDO casos das semanas)
    df_mensal = df.groupby(['ano', 'mes']).agg({
        'casos_dengue': 'sum'  # Soma de todas as semanas do mês
    }).reset_index()

    # Adicionar informações extras
    df_mensal['mes_nome'] = df_mensal['mes'].apply(lambda x: MESES_NOMES[x - 1])
    df_mensal['ano_mes'] = df_mensal.apply(lambda row: f"{row['ano']}-{row['mes']:02d}", axis=1)

    # Ordenar
    df_mensal = df_mensal.sort_values(['ano', 'mes']).reset_index(drop=True)

    # DEBUG
    st.success(f"✅ Agregação concluída: {len(df_mensal)} meses")
    st.dataframe(df_mensal.head(10))

    return df_mensal


def classificar_risco_casos_reais(casos: int) -> str:
    """
    Classifica risco baseado no número de casos REAIS
    """

    if casos > 5000:
        return 'Alto'
    elif casos > 1000:
        return 'Médio'
    else:
        return 'Baixo'