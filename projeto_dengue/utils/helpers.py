import pandas as pd
from backend.config import ESTADOS_BRASIL


def preparar_dados_mapa() -> pd.DataFrame:
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
    return df.to_csv(index=False).encode('utf-8')


def formatar_numero(numero: float, decimais: int = 1) -> str:
    return f"{numero:,.{decimais}f}".replace(',', 'X').replace('.', ',').replace('X', '.')