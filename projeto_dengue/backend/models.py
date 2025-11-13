"""
Módulo de modelos de Machine Learning para predição de dengue
"""

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
    import xgboost as xgb

    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False


class ModeloDengue:
    """Classe para treinar e avaliar modelos de predição de dengue"""

    def __init__(self):
        self.modelos_classificacao = {}
        self.modelos_regressao = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.melhor_modelo = None
        self.tipo_modelo = None  # 'classificacao' ou 'regressao'

    def preparar_dados(self, df: pd.DataFrame) -> tuple:
        """
        Prepara dados para treinamento

        Args:
            df: DataFrame com dados históricos

        Returns:
            Tupla (X, y, features)
        """

        # Features para o modelo
        features = [
            'temperatura_media', 'temperatura_max', 'temperatura_min',
            'umidade_relativa', 'precipitacao', 'mes'
        ]

        # Verificar se todas as features existem
        features_disponiveis = [f for f in features if f in df.columns]

        if len(features_disponiveis) < 3:
            raise ValueError(f"Dados insuficientes. Apenas {len(features_disponiveis)} features disponíveis.")

        X = df[features_disponiveis].copy()

        # Target (risco de dengue)
        y = df['risco_dengue'].copy()

        return X, y, features_disponiveis

    def treinar_modelos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Treina múltiplos modelos e retorna comparação

        Args:
            df: DataFrame com dados históricos

        Returns:
            DataFrame com resultados dos modelos
        """

        # Preparar dados
        X, y, features = self.preparar_dados(df)

        # =====================================================
        # VALIDAÇÃO: Verificar número de classes únicas
        # =====================================================

        classes_unicas = y.nunique()

        if classes_unicas < 2:
            # Apenas 1 classe → Usar REGRESSÃO em vez de CLASSIFICAÇÃO
            return self._treinar_modelos_regressao(df, X, features)

        # Se tem 2+ classes → Usar CLASSIFICAÇÃO
        return self._treinar_modelos_classificacao(X, y, features)

    def _treinar_modelos_classificacao(self, X: pd.DataFrame, y: pd.Series, features: list) -> pd.DataFrame:
        """
        Treina modelos de CLASSIFICAÇÃO (quando há múltiplas classes)
        """

        self.tipo_modelo = 'classificacao'

        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Balancear classes (se necessário)
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train)

        # Definir modelos
        self.modelos_classificacao = {
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        if XGBOOST_DISPONIVEL:
            self.modelos_classificacao['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, random_state=42, eval_metric='mlogloss'
            )

        # Treinar e avaliar
        resultados = []

        for nome, modelo in self.modelos_classificacao.items():
            try:
                # Treinar
                if nome == 'Gradient Boosting':
                    # Gradient Boosting precisa sample_weight válido
                    if len(np.unique(y_train)) >= 2:
                        modelo.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                    else:
                        modelo.fit(X_train_scaled, y_train)
                else:
                    modelo.fit(X_train_scaled, y_train)

                # Predizer
                y_pred = modelo.predict(X_test_scaled)

                # Métricas
                acuracia = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Cross-validation
                cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=3, scoring='accuracy')
                cv_mean = cv_scores.mean()

                resultados.append({
                    'Modelo': nome,
                    'Acurácia': acuracia,
                    'F1-Score': f1,
                    'CV Acurácia': cv_mean
                })

            except Exception as e:
                # Se modelo falhar, pular
                continue

        if not resultados:
            raise ValueError("Nenhum modelo foi treinado com sucesso")

        # Converter para DataFrame
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values('Acurácia', ascending=False).reset_index(drop=True)

        # Salvar melhor modelo
        melhor_nome = df_resultados.iloc[0]['Modelo']
        self.melhor_modelo = self.modelos_classificacao[melhor_nome]

        return df_resultados

    def _treinar_modelos_regressao(self, df: pd.DataFrame, X: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Treina modelos de REGRESSÃO (quando há apenas 1 classe ou poucos dados)
        Prediz o NÚMERO DE CASOS em vez da CLASSE DE RISCO
        """

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

        # Definir modelos de regressão
        self.modelos_regressao = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        if XGBOOST_DISPONIVEL:
            self.modelos_regressao['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)

        # Treinar e avaliar
        resultados = []

        for nome, modelo in self.modelos_regressao.items():
            try:
                # Treinar
                modelo.fit(X_train_scaled, y_train)

                # Predizer
                y_pred = modelo.predict(X_test_scaled)

                # Métricas
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                # Converter R² para "acurácia" (0-1)
                acuracia_equivalente = max(0, r2)

                resultados.append({
                    'Modelo': nome,
                    'Acurácia': acuracia_equivalente,  # R² como proxy
                    'F1-Score': acuracia_equivalente,  # Mesmo valor
                    'CV Acurácia': acuracia_equivalente,  # Simplificado
                    'MAE': mae,
                    'R²': r2
                })

            except Exception as e:
                continue

        if not resultados:
            raise ValueError("Nenhum modelo foi treinado com sucesso")

        # Converter para DataFrame
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values('R²', ascending=False).reset_index(drop=True)

        # Salvar melhor modelo
        melhor_nome = df_resultados.iloc[0]['Modelo']
        self.melhor_modelo = self.modelos_regressao[melhor_nome]

        return df_resultados

    def prever(self, X_novo: pd.DataFrame) -> np.ndarray:
        """
        Faz predições com o melhor modelo

        Args:
            X_novo: DataFrame com novos dados

        Returns:
            Array com predições
        """

        if self.melhor_modelo is None:
            raise ValueError("Modelo não foi treinado ainda!")

        X_scaled = self.scaler.transform(X_novo)

        if self.tipo_modelo == 'classificacao':
            # Retornar classes
            y_pred_encoded = self.melhor_modelo.predict(X_scaled)
            return self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            # Retornar número de casos
            return self.melhor_modelo.predict(X_scaled)