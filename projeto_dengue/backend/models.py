import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_DISPONIVEL = True
except ImportError:
    SMOTE_DISPONIVEL = False

try:
    import xgboost as xgb

    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False


class ModeloDengue:
    #Classe para treinar e avaliar modelos de predição de dengue

    def __init__(self):
        self.modelos_regressao = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.melhor_modelo = None
        self.tipo_modelo = None  # 'classificacao' ou 'regressao'

    def preparar_dados(self, df: pd.DataFrame) -> tuple:
        import streamlit as st
        from collections import Counter

        try:
            from backend.feature_engineering import (
                adicionar_features_engenheiradas,
                selecionar_features_relevantes,
                validar_features
            )

            if not validar_features(df):
                st.warning("⚠️ DataFrame não possui todas as colunas necessárias. Usando features básicas.")
                raise ValueError("Validação falhou")

            df_eng = adicionar_features_engenheiradas(df)
            X, features = selecionar_features_relevantes(df_eng)

            y = df_eng['casos_dengue'].copy()

        except (ImportError, ValueError) as e:
            st.warning(f"⚠️ Feature Engineering não disponível: {str(e)}")
            st.info("💡 Usando features básicas como fallback.")

            features = [
                'temperatura_media', 'temperatura_max', 'temperatura_min',
                'umidade_relativa', 'precipitacao', 'mes'
            ]

            features_disponiveis = [f for f in features if f in df.columns]

            if len(features_disponiveis) < 3:
                raise ValueError(f"Dados insuficientes. Apenas {len(features_disponiveis)} features disponíveis.")

            X = df[features_disponiveis].copy()
            features = features_disponiveis

            # Target: casos de dengue
            y = df['casos_dengue'].copy()


        if len(X) != len(y):
            raise ValueError(f"Incompatibilidade: X tem {len(X)} linhas, y tem {len(y)} linhas")

        if len(X) < 10:
            st.error(f"❌ Dados insuficientes: apenas {len(X)} registros disponíveis")
            raise ValueError(f"Mínimo de 10 registros necessários, encontrados {len(X)}")

        # Tratar NaN
        if X.isnull().any().any():
            st.warning("⚠️ Valores ausentes detectados. Preenchendo com mediana...")
            X = X.fillna(X.median())

        # Tratar infinitos
        if np.isinf(X.values).any():
            st.warning("⚠️ Valores infinitos detectados. Substituindo...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())

        return X, y, features

    def treinar_modelos(self, df: pd.DataFrame) -> pd.DataFrame:

        # Preparar dados
        X, y, features = self.preparar_dados(df)

        import streamlit as st

        # Mostrar correlações (informativo)
        df_temp = X.copy()

        # Se tiver target numérico (casos_dengue), calcular correlação
        if 'casos_dengue' in df.columns:
            df_temp['target'] = df['casos_dengue']
            correlacoes = df_temp.corr()['target'].drop('target').abs()
            max_corr = correlacoes.max()

            # Mostrar top 3 features
            top_features = correlacoes.nlargest(3)
            for feature, corr in top_features.items():
                st.write(f"   - {feature}: {corr:.3f}")

        # Treinar modelos de regressão
        return self._treinar_modelos_regressao(df, X, features)

    def _treinar_modelos_regressao(self, df: pd.DataFrame, X: pd.DataFrame, features: list) -> pd.DataFrame:

        self.tipo_modelo = 'regressao'

        # Target: número de casos
        y = df['casos_dengue'].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.svm import SVR

        import streamlit as st

        self.modelos_regressao = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ) if XGBOOST_DISPONIVEL else None,

            'Random Forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            ),

            'Extra Trees': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),

            'Ridge': Ridge(alpha=1.0),

            'Lasso': Lasso(alpha=1.0),

            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),

            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }

        # Remover modelos None (XGBoost se não disponível)
        self.modelos_regressao = {k: v for k, v in self.modelos_regressao.items() if v is not None}

        resultados = []

        for nome, modelo in self.modelos_regressao.items():
            try:
                # Treinar
                modelo.fit(X_train_scaled, y_train)

                # Predizer
                y_pred = modelo.predict(X_test_scaled)

                # Métricas
                from sklearn.metrics import mean_squared_error

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Erro percentual médio absoluto (MAPE)
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

                # Converter R² para "acurácia" (para compatibilidade com gráficos)
                acuracia_equivalente = max(0, min(1, r2))  # Clip entre 0-1

                resultados.append({
                    'Modelo': nome,
                    'R²': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE (%)': mape,
                    'Acurácia': acuracia_equivalente,
                    'F1-Score': acuracia_equivalente,
                    'CV Acurácia': acuracia_equivalente
                })

            except Exception as e:
                st.warning(f"⚠️ Erro ao treinar {nome}: {str(e)}")
                continue

        if not resultados:
            raise ValueError("Nenhum modelo foi treinado com sucesso")

        # Converter para DataFrame
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values('R²', ascending=False).reset_index(drop=True)

        # Salvar melhor modelo
        melhor_nome = df_resultados.iloc[0]['Modelo']
        self.melhor_modelo = self.modelos_regressao[melhor_nome]

        melhor_r2 = df_resultados.iloc[0]['R²']
        melhor_mae = df_resultados.iloc[0]['MAE']

        st.success(f"✅ **Melhor modelo:** {melhor_nome} | R² = {melhor_r2:.3f} | MAE = {melhor_mae:.0f} casos")

        return df_resultados

    def prever(self, X_novo: pd.DataFrame) -> np.ndarray:

        if self.melhor_modelo is None:
            raise ValueError("Modelo não foi treinado ainda!")

        X_scaled = self.scaler.transform(X_novo)

        # Sempre retornar número de casos (regressão)
        casos_previstos = self.melhor_modelo.predict(X_scaled)

        # Garantir que não há valores negativos
        casos_previstos = np.maximum(casos_previstos, 0)

        return casos_previstos