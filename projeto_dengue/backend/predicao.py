import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb

    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False


class PredicaoDengue:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.features = [
            'temperatura_media', 'temperatura_max', 'temperatura_min',
            'umidade_relativa', 'precipitacao', 'mes',
            'temp_x_umidade', 'temp_x_precip', 'umidade_x_precip',
            'temp_quadrada', 'mes_sin', 'mes_cos',
            'media_movel_3m', 'tendencia'
        ]
        self.melhor_modelo_nome = None
        self.mae = None
        self.r2 = None

    def preparar_features(self, df: pd.DataFrame) -> pd.DataFrame:

        #Cria features para o modelo preditivo

        df_prep = df.copy()

        # Features de interação
        df_prep['temp_x_umidade'] = df['temperatura_media'] * df['umidade_relativa']
        df_prep['temp_x_precip'] = df['temperatura_media'] * df['precipitacao']
        df_prep['umidade_x_precip'] = df['umidade_relativa'] * df['precipitacao']

        # Features não-lineares
        df_prep['temp_quadrada'] = df['temperatura_media'] ** 2

        # Sazonalidade circular
        df_prep['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df_prep['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

        # Ordenar por data
        df_prep = df_prep.sort_values(['ano', 'mes'])

        # Média móvel de casos (últimos 3 meses)
        df_prep['media_movel_3m'] = df_prep['casos_dengue'].rolling(
            window=3, min_periods=1
        ).mean()

        # Tendência (índice normalizado)
        df_prep['tendencia'] = np.arange(len(df_prep)) / len(df_prep)

        return df_prep

    def treinar_modelo(self, df: pd.DataFrame) -> dict:
        #Treina modelos de regressão para predição

        # Preparar features
        df_prep = self.preparar_features(df)

        # Separar features e target
        X = df_prep[self.features]
        y = df_prep['casos_dengue']

        # Dividir em treino e teste (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Treinar múltiplos modelos
        modelos = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }

        if XGBOOST_DISPONIVEL:
            modelos['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )

        resultados = []
        melhor_mae = float('inf')

        for nome, modelo in modelos.items():
            # Treinar
            modelo.fit(X_train_scaled, y_train)

            # Predizer
            y_pred = modelo.predict(X_test_scaled)

            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            resultados.append({
                'Modelo': nome,
                'MAE': mae,
                'R²': r2,
                'RMSE': np.sqrt(np.mean((y_test - y_pred) ** 2))
            })

            # Salvar melhor modelo
            if mae < melhor_mae:
                melhor_mae = mae
                self.modelo = modelo
                self.melhor_modelo_nome = nome
                self.mae = mae
                self.r2 = r2

        return {
            'resultados': pd.DataFrame(resultados),
            'melhor_modelo': self.melhor_modelo_nome,
            'mae': self.mae,
            'r2': self.r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'datas_test': df_prep.iloc[split_idx:][['ano', 'mes', 'ano_mes']].reset_index(drop=True)
        }

    def prever_mes_atual(self, df_historico: pd.DataFrame,
                         clima_atual: dict) -> dict:
        #Prevê casos de dengue para o mês atual

        if self.modelo is None:
            raise ValueError("Modelo não foi treinado ainda!")

        # Preparar dados do mês atual
        df_atual = pd.DataFrame([clima_atual])

        # Calcular features derivadas
        df_atual['temp_x_umidade'] = df_atual['temperatura_media'] * df_atual['umidade_relativa']
        df_atual['temp_x_precip'] = df_atual['temperatura_media'] * df_atual['precipitacao']
        df_atual['umidade_x_precip'] = df_atual['umidade_relativa'] * df_atual['precipitacao']
        df_atual['temp_quadrada'] = df_atual['temperatura_media'] ** 2
        df_atual['mes_sin'] = np.sin(2 * np.pi * df_atual['mes'] / 12)
        df_atual['mes_cos'] = np.cos(2 * np.pi * df_atual['mes'] / 12)

        # Média móvel (últimos 3 meses históricos)
        ultimos_3_meses = df_historico.tail(3)['casos_dengue'].mean()
        df_atual['media_movel_3m'] = ultimos_3_meses

        # Tendência
        df_atual['tendencia'] = 1.0

        # Normalizar
        X_atual = df_atual[self.features]
        X_atual_scaled = self.scaler.transform(X_atual)

        # Predizer
        predicao = self.modelo.predict(X_atual_scaled)[0]

        # Intervalo de confiança
        intervalo_inferior = max(0, predicao - 1.96 * self.mae)
        intervalo_superior = predicao + 1.96 * self.mae

        casos_previstos = int(predicao)

        temp = clima_atual['temperatura_media']
        umidade = clima_atual['umidade_relativa']
        precip = clima_atual['precipitacao']
        mes = clima_atual['mes']

        # Calcular SCORE CLIMÁTICO
        score_climatico = 0

        # Temperatura
        if 25 <= temp <= 30:
            score_climatico += 3
        elif 20 <= temp <= 35:
            score_climatico += 2
        elif temp > 18:
            score_climatico += 1

        # Umidade
        if umidade > 80:
            score_climatico += 3
        elif umidade > 70:
            score_climatico += 2
        elif umidade > 60:
            score_climatico += 1

        # Precipitação
        if precip > 150:
            score_climatico += 3
        elif precip > 100:
            score_climatico += 2
        elif precip > 50:
            score_climatico += 1

        # Sazonalidade
        if mes in [12, 1, 2, 3]:  # Verão (ALTÍSSIMO RISCO)
            score_climatico += 3
        elif mes in [10, 11]:  # Primavera (ALTO RISCO)
            score_climatico += 2
        elif mes in [4, 5]:  # Outono
            score_climatico += 1

        # Obter média histórica do mês (AGREGADA)
        mes_atual = clima_atual['mes']
        df_historico_mes = df_historico[df_historico['mes'] == mes_atual]

        # Calcular total por ano
        casos_por_ano = df_historico_mes.groupby('ano')['casos_dengue'].sum()
        casos_historicos_mes = casos_por_ano.mean()

        # Escalar predição para comparar
        n_amostras = len(df_historico_mes) / len(df_historico['ano'].unique())
        casos_previstos_escalados = predicao * n_amostras

        # Calcular variação percentual
        variacao_percentual = ((casos_previstos_escalados - casos_historicos_mes) / casos_historicos_mes * 100)

        # CLASSIFICAÇÃO HÍBRIDA (Score Climático + Casos Previstos)

        # 1. Se score climático for ALTO
        if score_climatico >= 8:
            # Condições muito favoráveis
            if variacao_percentual > -30:  # Mesmo abaixo, se clima for crítico
                risco_previsto = 'Alto'
                alerta = '🔴 ALERTA: Condições climáticas críticas!'
            else:
                risco_previsto = 'Médio'
                alerta = '🟡 ATENÇÃO: Clima favorável, mas casos muito abaixo do esperado'

        elif score_climatico >= 5:
            # Condições moderadas
            if variacao_percentual > 20:
                risco_previsto = 'Alto'
                alerta = '🔴 ALERTA: Casos previstos muito acima da média!'
            elif variacao_percentual > -20:
                risco_previsto = 'Médio'
                alerta = '🟡 ATENÇÃO: Casos dentro da faixa esperada'
            else:
                risco_previsto = 'Baixo'
                alerta = '🟢 Normal: Casos abaixo da média'

        else:
            # Condições desfavoráveis
            if variacao_percentual > 50:
                risco_previsto = 'Médio'
                alerta = '🟡 ATENÇÃO: Casos elevados apesar de clima desfavorável'
            else:
                risco_previsto = 'Baixo'
                alerta = '🟢 Normal: Clima e casos dentro do esperado'

        return {
            'casos_previstos': int(predicao),  # ← Individual
            'intervalo_inferior': int(intervalo_inferior),
            'intervalo_superior': int(intervalo_superior),
            'risco_previsto': risco_previsto,
            'alerta': alerta,
            'confianca': self.r2,
            'modelo_usado': self.melhor_modelo_nome,
            'casos_historicos_media': int(casos_historicos_mes),  # ← Agregado
            'variacao_percentual': variacao_percentual,
            'score_climatico': score_climatico
        }

    def prever_proximos_meses(self, df_historico: pd.DataFrame,
                              clima_futuro: list, n_meses: int = 3) -> pd.DataFrame:
        #Prevê casos para os próximos N meses

        predicoes = []

        for i, clima in enumerate(clima_futuro[:n_meses]):
            resultado = self.prever_mes_atual(df_historico, clima)

            predicoes.append({
                'mes': clima['mes'],
                'ano': clima.get('ano', 2025),
                'casos_previstos': resultado['casos_previstos'],
                'intervalo_inferior': resultado['intervalo_inferior'],
                'intervalo_superior': resultado['intervalo_superior'],
                'risco_previsto': resultado['risco_previsto']
            })

        return pd.DataFrame(predicoes)


def obter_clima_atual_estimado(estado_nome: str) -> dict:

    #Obtém clima atual estimado (você pode integrar com API real)

    from backend.config import PARAMETROS_CLIMA, ESTADOS_BRASIL

    # Obter mês atual
    mes_atual = datetime.now().month
    ano_atual = datetime.now().year

    # Obter parâmetros da região
    estado_info = ESTADOS_BRASIL[estado_nome]
    regiao = estado_info['regiao']
    params = PARAMETROS_CLIMA[regiao]

    # Calcular fatores sazonais
    if mes_atual in [12, 1, 2]:
        fator_temp, fator_chuva = 1.15, 1.5
    elif mes_atual in [3, 4]:
        fator_temp, fator_chuva = 1.05, 1.3
    elif mes_atual in [6, 7, 8]:
        fator_temp, fator_chuva = 0.80, 0.5
    else:
        fator_temp, fator_chuva = 0.95, 0.9

    # Estimar clima atual
    temp_media = params['temp_base'] * fator_temp

    return {
        'temperatura_media': temp_media,
        'temperatura_max': temp_media + 5,
        'temperatura_min': temp_media - 5,
        'umidade_relativa': params['umidade_base'],
        'precipitacao': params['precip_base'] * fator_chuva,
        'mes': mes_atual,
        'ano': ano_atual
    }