import requests
import pandas as pd
from datetime import datetime, timedelta

#Carregar dados do INMET
def carregar_dados_inmet(codigo_estacao, data_inicio, data_fim):
    url = f"https://apitempo.inmet.gov.br/estacao/{data_inicio}/{data_fim}/{codigo_estacao}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            dados = response.json()
            df = pd.DataFrame(dados)
            return df
        else:
            print(f"Erro ao carregar dados: {response.status_code}")
            return None
    except Exception as e:
        print(f"Erro: {e}")
        return None

def carregar_dados_dengue(geocode, data_inicio, data_fim):
    url = f"https://info.dengue.mat.br/api/alertcity"

    params = {
        'geocode': geocode,
        'disease': 'dengue',
        'format': 'json',
        'ew_start': data_inicio,
        'ew_end': data_fim
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            dados = response.json()
            df = pd.DataFrame(dados)
            return df
        else:
            print(f"Erro ao carregar dados: {response.status_code}")
            return None
    except Exception as e:
        print(f"Erro: {e}")
        return None

#Exemplo
if __name__ == "__main__":
    # Exemplo: dados de dengue do Rio de Janeiro
    geocode_rj = 3304557  # Código IBGE do Rio de Janeiro

    print("Carregando dados de dengue...")
    df_dengue = carregar_dados_dengue(
        geocode=geocode_rj,
        data_inicio=202201,  # Ano/Semana epidemiológica
        data_fim=202352
    )

    if df_dengue is not None:
        print(f"✓ {len(df_dengue)} registros carregados")
        print(df_dengue.head())
        df_dengue.to_csv('dados_dengue_rj.csv', index=False)
        print("Dados salvos em 'dados_dengue_rj.csv'")