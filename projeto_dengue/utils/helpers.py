"""
Funções auxiliares e utilitárias
"""

import pandas as pd
from backend.config import ESTADOS_BRASIL


def preparar_dados_mapa() -> pd.DataFrame:
    """
    Prepara dados para visualização no mapa

    Returns:
        DataFrame com informações dos estados
    """
    return pd.DataFrame([
        {
            'estado': nome,
            'lat': info['lat'],
            'lon': info['lon'],
            'regiao': info['regiao']
        }
        for nome, info in ESTADOS_BRASIL.items()
    ])


def exportar_csv(df: pd.DataFrame, estado_nome: str) -> bytes:
    """
    Exporta DataFrame para CSV

    Args:
        df: DataFrame para exportar
        estado_nome: Nome do estado

    Returns:
        Bytes do arquivo CSV
    """
    return df.to_csv(index=False).encode('utf-8')


def formatar_numero(numero: float, decimais: int = 1) -> str:
    """
    Formata número com separador de milhares

    Args:
        numero: Número para formatar
        decimais: Número de casas decimais

    Returns:
        String formatada
    """
    return f"{numero:,.{decimais}f}".replace(',', 'X').replace('.', ',').replace('X', '.')