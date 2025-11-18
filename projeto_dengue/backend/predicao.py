import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb

    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False


class PredicaoDengue:

    #Classe para predição de casos de dengue no mês atual


    def __init__(self):
        """Inicializa o modelo de predição"""
        self.melhor_modelo = None
        self.melhor_modelo_nome = None
        self.scaler = StandardScaler()
        self.features = []
        self.r2_score = 0.0
        self.mae_score = 0.0

    def treinar_modelo(self, df: pd.DataFrame) -> dict:

        try:
            from backend.feature_engineering import (
                adicionar_features_engenheiradas,
                selecionar_features_relevantes,
                validar_features
            )

            if validar_features(df):
                df_eng = adicionar_features_engenheiradas(df)
                X, features = selecionar_features_relevantes(df_eng)
            else:
                raise ValueError("Validação falhou")

        except (ImportError, ValueError):
            # Fallback: features básicas
            features = ['temperatura_media', 'temperatura_max', 'temperatura_min',
                        'umidade_relativa', 'precipitacao', 'mes']
            features_disponiveis = [f for f in features if f in df.columns]
            X = df[features_disponiveis].copy()
            features = features_disponiveis

        # Target: casos de dengue
        y = df['casos_dengue'].values

        # Verificar dados suficientes
        if len(X) < 10:
            raise ValueError(f"Dados insuficientes: apenas {len(X)} registros")


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        modelos = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0)
        }

        if XGBOOST_DISPONIVEL:
            modelos['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )

        resultados = []
        melhor_r2 = -np.inf
        melhor_modelo = None
        melhor_nome = None
        melhor_mae = np.inf

        for nome, modelo in modelos.items():
            try:
                # Treinar
                modelo.fit(X_train_scaled, y_train)

                # Predizer
                y_pred = modelo.predict(X_test_scaled)

                # Métricas
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # MAPE
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

                resultados.append({
                    'Modelo': nome,
                    'R²': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE (%)': mape
                })

                # Atualizar melhor modelo
                if r2 > melhor_r2:
                    melhor_r2 = r2
                    melhor_mae = mae
                    melhor_modelo = modelo
                    melhor_nome = nome

            except Exception as e:
                print(f"Erro ao treinar {nome}: {str(e)}")
                continue

        if melhor_modelo is None:
            raise ValueError("Nenhum modelo foi treinado com sucesso")

        self.melhor_modelo = melhor_modelo
        self.melhor_modelo_nome = melhor_nome
        self.features = features
        self.r2_score = melhor_r2
        self.mae_score = melhor_mae

        # Converter para DataFrame
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values('R²', ascending=False).reset_index(drop=True)

        return {
            'resultados': df_resultados,
            'r2': melhor_r2,
            'mae': melhor_mae
        }

    def prever_mes_atual(self, df: pd.DataFrame, clima_atual: dict) -> dict:

        if self.melhor_modelo is None:
            raise ValueError("Modelo não foi treinado. Execute treinar_modelo() primeiro.")

        # Feature engineering no clima atual
        X_novo = pd.DataFrame([{
            'temperatura_media': clima_atual.get('temperatura_media', df['temperatura_media'].mean()),
            'temperatura_max': clima_atual.get('temperatura_max', df['temperatura_max'].mean()),
            'temperatura_min': clima_atual.get('temperatura_min', df['temperatura_min'].mean()),
            'umidade_relativa': clima_atual.get('umidade_relativa', df['umidade_relativa'].mean()),
            'precipitacao': clima_atual.get('precipitacao', df['precipitacao'].mean()),
            'mes': datetime.now().month
        }])

        # Aplicar feature engineering (se disponível)
        try:
            from backend.feature_engineering import adicionar_features_engenheiradas

            # Adicionar histórico recente para features de lag
            df_temp = df.copy().tail(10)
            df_temp = pd.concat([df_temp, X_novo], ignore_index=True)

            # Feature engineering
            df_temp = adicionar_features_engenheiradas(df_temp)

            # Pegar apenas a última linha (predição)
            X_novo_eng = df_temp.tail(1)

            # Selecionar features (apenas as que foram usadas no treino)
            features_disponiveis = [f for f in self.features if f in X_novo_eng.columns]
            X_novo_final = X_novo_eng[features_disponiveis]

        except (ImportError, Exception):
            # Fallback: usar features básicas
            features_basicas = ['temperatura_media', 'temperatura_max', 'temperatura_min',
                                'umidade_relativa', 'precipitacao', 'mes']
            features_disponiveis = [f for f in features_basicas if f in X_novo.columns]
            X_novo_final = X_novo[features_disponiveis]

        # Normalizar
        X_novo_scaled = self.scaler.transform(X_novo_final)

        # Predizer
        casos_previstos = self.melhor_modelo.predict(X_novo_scaled)[0]

        # Garantir que não seja negativo
        casos_previstos = max(0, casos_previstos)

        # Usar erro médio (MAE) do treino para estimar intervalo
        mae = self.mae_score

        intervalo_inferior = max(0, casos_previstos - (1.5 * mae))
        intervalo_superior = casos_previstos + (1.5 * mae)

        # Média histórica do mesmo mês
        mes_atual = datetime.now().month
        df_mesmo_mes = df[df['mes'] == mes_atual]

        if len(df_mesmo_mes) > 0:
            media_historica = df_mesmo_mes['casos_dengue'].mean()
        else:
            media_historica = df['casos_dengue'].mean()

        # Variação percentual
        if media_historica > 0:
            variacao_pct = ((casos_previstos - media_historica) / media_historica) * 100
        else:
            variacao_pct = 0.0

        return {
            'casos_previstos': float(casos_previstos),
            'intervalo_inferior': float(intervalo_inferior),
            'intervalo_superior': float(intervalo_superior),
            'confianca': float(self.r2_score),
            'modelo_usado': self.melhor_modelo_nome,
            'variacao_historico': float(variacao_pct),
            'media_historica': float(media_historica),
            'mes': mes_atual,
            'ano': datetime.now().year
        }


def obter_clima_atual_estimado(estado_nome: str) -> dict:

    # Mês atual
    mes_atual = datetime.now().month

    # Estimativas baseadas em médias brasileiras por região
    # (Idealmente seria buscar API de clima em tempo real)

    climas_estimados = {
        'São Paulo': {
            1: {'temp': 24, 'umid': 75, 'precip': 200},  # Janeiro
            2: {'temp': 25, 'umid': 73, 'precip': 180},
            3: {'temp': 24, 'umid': 72, 'precip': 150},
            4: {'temp': 22, 'umid': 70, 'precip': 80},
            5: {'temp': 19, 'umid': 68, 'precip': 60},
            6: {'temp': 18, 'umid': 66, 'precip': 50},
            7: {'temp': 18, 'umid': 65, 'precip': 40},
            8: {'temp': 20, 'umid': 64, 'precip': 45},
            9: {'temp': 21, 'umid': 67, 'precip': 70},
            10: {'temp': 22, 'umid': 70, 'precip': 110},
            11: {'temp': 23, 'umid': 72, 'precip': 130},  # Novembro
            12: {'temp': 24, 'umid': 74, 'precip': 170}
        }
    }

    # Pegar clima do estado ou usar default
    clima_mes = climas_estimados.get(estado_nome, climas_estimados['São Paulo']).get(mes_atual, {
        'temp': 22, 'umid': 70, 'precip': 100
    })

    return {
        'temperatura_media': clima_mes['temp'],
        'temperatura_max': clima_mes['temp'] + 5,
        'temperatura_min': clima_mes['temp'] - 5,
        'umidade_relativa': clima_mes['umid'],
        'precipitacao': clima_mes['precip']
    }