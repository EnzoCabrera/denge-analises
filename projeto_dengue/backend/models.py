"""
Módulo de Machine Learning - Treinamento e avaliação de modelos
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

from backend.config import ML_CONFIG


class ModeloDengue:
    """Classe para gerenciar modelos de predição de dengue"""

    def __init__(self):
        self.modelos = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.melhor_modelo = None
        self.melhor_nome = None
        self.resultados = None

    def treinar_modelos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Treina múltiplos modelos e retorna comparação

        Args:
            df: DataFrame com dados

        Returns:
            DataFrame com resultados dos modelos
        """
        # FEATURES EXPANDIDAS (incluindo interações)
        df_features = self._criar_features_ml(df)

        # Preparar dados
        features_expandidas = [
            'temperatura_media', 'umidade_relativa', 'precipitacao',
            'mes', 'temperatura_max', 'temperatura_min',
            'temp_x_umidade', 'temp_x_precip', 'umidade_x_precip',
            'temp_quadrada', 'mes_sin', 'mes_cos'
        ]

        X = df_features[features_expandidas]
        y = self.label_encoder.fit_transform(df['risco_dengue'])

        print(f"📊 Dataset: {len(X)} amostras")
        print(f"📊 Distribuição de classes: {Counter(y)}")

        # Split com verificação de classes
        X_train, X_test, y_train, y_test = self._split_seguro(X, y)

        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # BALANCEAMENTO com SMOTE (se disponível)
        if SMOTE_DISPONIVEL and len(X_train) > 20:
            try:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                print(f"✓ SMOTE aplicado: {len(X_train)} → {len(X_train_balanced)} amostras")
            except:
                X_train_balanced = X_train_scaled
                y_train_balanced = y_train
        else:
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train

        # Definir modelos com MELHORES hiperparâmetros
        self.modelos = {
            'Naive Bayes': GaussianNB(var_smoothing=1e-9),

            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=ML_CONFIG['random_state'],
                n_jobs=-1
            ),

            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=ML_CONFIG['random_state']
            )
        }

        if XGBOOST_DISPONIVEL:
            self.modelos['XGBoost'] = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=ML_CONFIG['random_state'],
                eval_metric='mlogloss',
                n_jobs=-1
            )

        # Treinar e avaliar com CROSS-VALIDATION
        resultados = []
        for nome, modelo in self.modelos.items():
            # Treinar
            modelo.fit(X_train_balanced, y_train_balanced)

            # Avaliar no teste
            y_pred = modelo.predict(X_test_scaled)
            acc_test = accuracy_score(y_test, y_pred)
            f1_test = f1_score(y_test, y_pred, average='weighted')

            # Cross-validation (se houver dados suficientes)
            if len(X_train_balanced) >= 30:
                try:
                    cv_scores = cross_val_score(
                        modelo, X_train_balanced, y_train_balanced,
                        cv=min(5, len(X_train_balanced) // 10),
                        scoring='accuracy'
                    )
                    acc_cv = cv_scores.mean()
                except:
                    acc_cv = acc_test
            else:
                acc_cv = acc_test

            resultados.append({
                'Modelo': nome,
                'Acurácia': acc_test,
                'F1-Score': f1_test,
                'CV Acurácia': acc_cv
            })

            print(f"✓ {nome}: Acc={acc_test:.2%}, F1={f1_test:.2%}, CV={acc_cv:.2%}")

        # Ordenar resultados
        self.resultados = pd.DataFrame(resultados).sort_values('Acurácia', ascending=False)

        # Definir melhor modelo
        self.melhor_nome = self.resultados.iloc[0]['Modelo']
        self.melhor_modelo = self.modelos[self.melhor_nome]

        return self.resultados

    def _criar_features_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de Machine Learning

        Args:
            df: DataFrame original

        Returns:
            DataFrame com features expandidas
        """
        df_ml = df.copy()

        # Interações entre variáveis
        df_ml['temp_x_umidade'] = df['temperatura_media'] * df['umidade_relativa']
        df_ml['temp_x_precip'] = df['temperatura_media'] * df['precipitacao']
        df_ml['umidade_x_precip'] = df['umidade_relativa'] * df['precipitacao']

        # Transformações não-lineares
        df_ml['temp_quadrada'] = df['temperatura_media'] ** 2

        # Sazonalidade circular
        df_ml['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df_ml['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

        return df_ml

    def _split_seguro(self, X, y):
        """
        Faz split dos dados com verificação de classes

        Args:
            X: Features
            y: Target

        Returns:
            X_train, X_test, y_train, y_test
        """
        class_counts = Counter(y)
        min_samples = min(class_counts.values())

        # Aumentar test_size se dataset for pequeno
        if len(X) < 50:
            test_size = 0.25
        else:
            test_size = ML_CONFIG['test_size']

        if min_samples >= 2 and len(X) >= 10:
            return train_test_split(
                X, y,
                test_size=test_size,
                random_state=ML_CONFIG['random_state'],
                stratify=y
            )
        else:
            return train_test_split(
                X, y,
                test_size=test_size,
                random_state=ML_CONFIG['random_state']
            )

    def prever(self, dados_novos: pd.DataFrame) -> np.ndarray:
        """
        Faz predições com o melhor modelo

        Args:
            dados_novos: DataFrame com novos dados

        Returns:
            Array com predições
        """
        if self.melhor_modelo is None:
            raise ValueError("Nenhum modelo foi treinado ainda!")

        # Criar features expandidas
        df_features = self._criar_features_ml(dados_novos)

        features_expandidas = [
            'temperatura_media', 'umidade_relativa', 'precipitacao',
            'mes', 'temperatura_max', 'temperatura_min',
            'temp_x_umidade', 'temp_x_precip', 'umidade_x_precip',
            'temp_quadrada', 'mes_sin', 'mes_cos'
        ]

        X = df_features[features_expandidas]
        X_scaled = self.scaler.transform(X)
        predicoes = self.melhor_modelo.predict(X_scaled)

        return self.label_encoder.inverse_transform(predicoes)

    def get_melhor_acuracia(self) -> float:
        """Retorna a acurácia do melhor modelo"""
        if self.resultados is None:
            return 0.0
        return float(self.resultados.iloc[0]['Acurácia'])

    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importância das features (para modelos baseados em árvore)"""
        if self.melhor_modelo is None:
            return pd.DataFrame()

        if hasattr(self.melhor_modelo, 'feature_importances_'):
            features_expandidas = [
                'temperatura_media', 'umidade_relativa', 'precipitacao',
                'mes', 'temperatura_max', 'temperatura_min',
                'temp_x_umidade', 'temp_x_precip', 'umidade_x_precip',
                'temp_quadrada', 'mes_sin', 'mes_cos'
            ]

            importances = pd.DataFrame({
                'feature': features_expandidas,
                'importance': self.melhor_modelo.feature_importances_
            }).sort_values('importance', ascending=False)

            return importances

        return pd.DataFrame()