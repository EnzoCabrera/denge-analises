import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# 1. COLETA E PREPARAÇÃO DOS DADOS
# =====================================================

def carregar_dados():
    """
    Simula carregamento de dados climáticos e casos de dengue
    Na prática, você carregaria de APIs ou arquivos CSV
    """
    np.random.seed(42)
    n_samples = 1000

    # Dados simulados
    data = {
        'latitude': np.random.uniform(-33, 5, n_samples),  # Brasil
        'longitude': np.random.uniform(-73, -34, n_samples),
        'temperatura_media': np.random.uniform(15, 35, n_samples),  # °C
        'umidade_relativa': np.random.uniform(40, 95, n_samples),  # %
        'precipitacao': np.random.uniform(0, 300, n_samples),  # mm
        'altitude': np.random.uniform(0, 1000, n_samples),  # metros
        'mes': np.random.randint(1, 13, n_samples),
        'populacao_densidade': np.random.uniform(10, 10000, n_samples)  # hab/km²
    }

    df = pd.DataFrame(data)

    # Criar variável alvo baseada em condições favoráveis à dengue
    # Dengue é mais comum com: temperatura alta, umidade alta, chuva moderada
    df['risco_dengue'] = 'Baixo'

    condicao_alto = (
            (df['temperatura_media'] > 25) &
            (df['umidade_relativa'] > 70) &
            (df['precipitacao'] > 80) &
            (df['altitude'] < 500)
    )

    condicao_medio = (
            (df['temperatura_media'] > 22) &
            (df['umidade_relativa'] > 60) &
            (df['precipitacao'] > 40)
    )

    df.loc[condicao_alto, 'risco_dengue'] = 'Alto'
    df.loc[condicao_medio & ~condicao_alto, 'risco_dengue'] = 'Médio'

    return df


# =====================================================
# 2. PRÉ-PROCESSAMENTO
# =====================================================

def preprocessar_dados(df):
    """
    Prepara os dados para o modelo
    """
    # Criar features adicionais
    df['estacao'] = df['mes'].apply(lambda x:
                                    'Verão' if x in [12, 1, 2] else
                                    'Outono' if x in [3, 4, 5] else
                                    'Inverno' if x in [6, 7, 8] else 'Primavera'
                                    )

    # Codificar variável categórica
    le_estacao = LabelEncoder()
    df['estacao_encoded'] = le_estacao.fit_transform(df['estacao'])

    # Selecionar features
    features = [
        'latitude', 'longitude', 'temperatura_media',
        'umidade_relativa', 'precipitacao', 'altitude',
        'mes', 'populacao_densidade', 'estacao_encoded'
    ]

    X = df[features]

    # Codificar variável alvo
    le_risco = LabelEncoder()
    y = le_risco.fit_transform(df['risco_dengue'])

    return X, y, le_risco, features


# =====================================================
# 3. TREINAMENTO DO MODELO NAIVE BAYES
# =====================================================

def treinar_modelo(X, y):
    """
    Treina o modelo Naive Bayes Gaussiano
    """
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Normalizar dados (opcional para Naive Bayes, mas pode ajudar)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criar e treinar o modelo
    modelo = GaussianNB()
    modelo.fit(X_train_scaled, y_train)

    # Fazer predições
    y_pred = modelo.predict(X_test_scaled)

    return modelo, scaler, X_test_scaled, y_test, y_pred


# =====================================================
# 4. AVALIAÇÃO DO MODELO
# =====================================================

def avaliar_modelo(y_test, y_pred, le_risco):
    """
    Avalia o desempenho do modelo
    """
    print("=" * 60)
    print("MÉTRICAS DE AVALIAÇÃO DO MODELO")
    print("=" * 60)

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia: {accuracy:.2%}")

    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred,
                                target_names=le_risco.classes_))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_risco.classes_,
                yticklabels=le_risco.classes_)
    plt.title('Matriz de Confusão - Predição de Risco de Dengue')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
    print("\nMatriz de confusão salva como 'matriz_confusao.png'")


# =====================================================
# 5. PREDIÇÃO PARA NOVOS DADOS
# =====================================================

def prever_risco(modelo, scaler, le_risco, dados_novos):
    """
    Faz predição para novos pontos geográficos
    """
    # Normalizar dados
    dados_scaled = scaler.transform(dados_novos)

    # Predição
    predicao = modelo.predict(dados_scaled)
    probabilidades = modelo.predict_proba(dados_scaled)

    # Converter para labels
    risco_predicado = le_risco.inverse_transform(predicao)

    resultado = pd.DataFrame({
        'Risco_Predito': risco_predicado,
        'Prob_Alto': probabilidades[:, 0],
        'Prob_Baixo': probabilidades[:, 1],
        'Prob_Médio': probabilidades[:, 2]
    })

    return resultado


# =====================================================
# 6. VISUALIZAÇÃO DOS RESULTADOS
# =====================================================

def visualizar_importancia_features(df, features):
    """
    Visualiza correlação entre features e risco de dengue
    """
    # Codificar risco para correlação
    le = LabelEncoder()
    df['risco_encoded'] = le.fit_transform(df['risco_dengue'])

    # Calcular correlação
    correlacao = df[features + ['risco_encoded']].corr()['risco_encoded'].drop('risco_encoded')
    correlacao = correlacao.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    correlacao.plot(kind='barh', color='steelblue')
    plt.title('Correlação das Features com Risco de Dengue')
    plt.xlabel('Correlação')
    plt.tight_layout()
    plt.savefig('correlacao_features.png', dpi=300, bbox_inches='tight')
    print("Gráfico de correlação salvo como 'correlacao_features.png'")


# =====================================================
# 7. FUNÇÃO PRINCIPAL
# =====================================================

def main():
    print("🦟 SISTEMA DE PREDIÇÃO DE RISCO DE DENGUE 🦟")
    print("=" * 60)

    # 1. Carregar dados
    print("\n[1] Carregando dados...")
    df = carregar_dados()
    print(f"✓ {len(df)} registros carregados")
    print(f"\nDistribuição de Risco:")
    print(df['risco_dengue'].value_counts())

    # 2. Preprocessar
    print("\n[2] Preprocessando dados...")
    X, y, le_risco, features = preprocessar_dados(df)
    print(f"✓ {len(features)} features preparadas")

    # 3. Treinar modelo
    print("\n[3] Treinando modelo Naive Bayes...")
    modelo, scaler, X_test, y_test, y_pred = treinar_modelo(X, y)
    print("✓ Modelo treinado com sucesso")

    # 4. Avaliar
    print("\n[4] Avaliando modelo...")
    avaliar_modelo(y_test, y_pred, le_risco)

    # 5. Visualizar
    print("\n[5] Gerando visualizações...")
    visualizar_importancia_features(df, features)

    # 6. Exemplo de predição
    print("\n[6] Exemplo de predição para novos dados:")
    print("=" * 60)

    # Criar exemplo de dados novos
    dados_exemplo = pd.DataFrame({
        'latitude': [-23.5505],  # São Paulo
        'longitude': [-46.6333],
        'temperatura_media': [28.0],
        'umidade_relativa': [80.0],
        'precipitacao': [150.0],
        'altitude': [760.0],
        'mes': [2],  # Fevereiro (verão)
        'populacao_densidade': [7398.0],
        'estacao_encoded': [0]  # Verão
    })

    resultado = prever_risco(modelo, scaler, le_risco, dados_exemplo)
    print("\nLocal: São Paulo (exemplo)")
    print(f"Risco Predito: {resultado['Risco_Predito'].values[0]}")
    print(f"\nProbabilidades:")
    print(f"  - Alto: {resultado['Prob_Alto'].values[0]:.2%}")
    print(f"  - Médio: {resultado['Prob_Médio'].values[0]:.2%}")
    print(f"  - Baixo: {resultado['Prob_Baixo'].values[0]:.2%}")

    print("\n" + "=" * 60)
    print("✓ Processamento concluído!")

    return modelo, scaler, le_risco, features


# Executar
if __name__ == "__main__":
    modelo, scaler, le_risco, features = main()