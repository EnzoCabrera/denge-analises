# Estados do Brasil com coordenadas
ESTADOS_BRASIL = {
    'Acre': {'sigla': 'AC', 'regiao': 'Norte', 'lat': -9.0238, 'lon': -70.8120},
    'Alagoas': {'sigla': 'AL', 'regiao': 'Nordeste', 'lat': -9.5713, 'lon': -36.7820},
    'Amapá': {'sigla': 'AP', 'regiao': 'Norte', 'lat': 0.9020, 'lon': -52.0030},
    'Amazonas': {'sigla': 'AM', 'regiao': 'Norte', 'lat': -3.4168, 'lon': -65.8561},
    'Bahia': {'sigla': 'BA', 'regiao': 'Nordeste', 'lat': -12.5797, 'lon': -41.7007},
    'Ceará': {'sigla': 'CE', 'regiao': 'Nordeste', 'lat': -5.4984, 'lon': -39.3206},
    'Distrito Federal': {'sigla': 'DF', 'regiao': 'Centro-Oeste', 'lat': -15.7998, 'lon': -47.8645},
    'Espírito Santo': {'sigla': 'ES', 'regiao': 'Sudeste', 'lat': -19.1834, 'lon': -40.3089},
    'Goiás': {'sigla': 'GO', 'regiao': 'Centro-Oeste', 'lat': -15.8270, 'lon': -49.8362},
    'Maranhão': {'sigla': 'MA', 'regiao': 'Nordeste', 'lat': -4.9609, 'lon': -45.2744},
    'Mato Grosso': {'sigla': 'MT', 'regiao': 'Centro-Oeste', 'lat': -12.6819, 'lon': -56.9211},
    'Mato Grosso do Sul': {'sigla': 'MS', 'regiao': 'Centro-Oeste', 'lat': -20.7722, 'lon': -54.7852},
    'Minas Gerais': {'sigla': 'MG', 'regiao': 'Sudeste', 'lat': -18.5122, 'lon': -44.5550},
    'Pará': {'sigla': 'PA', 'regiao': 'Norte', 'lat': -1.9981, 'lon': -54.9306},
    'Paraíba': {'sigla': 'PB', 'regiao': 'Nordeste', 'lat': -7.2399, 'lon': -36.7819},
    'Paraná': {'sigla': 'PR', 'regiao': 'Sul', 'lat': -25.2521, 'lon': -52.0215},
    'Pernambuco': {'sigla': 'PE', 'regiao': 'Nordeste', 'lat': -8.8137, 'lon': -36.9541},
    'Piauí': {'sigla': 'PI', 'regiao': 'Nordeste', 'lat': -7.7183, 'lon': -42.7289},
    'Rio de Janeiro': {'sigla': 'RJ', 'regiao': 'Sudeste', 'lat': -22.9068, 'lon': -43.1729},
    'Rio Grande do Norte': {'sigla': 'RN', 'regiao': 'Nordeste', 'lat': -5.4026, 'lon': -36.9541},
    'Rio Grande do Sul': {'sigla': 'RS', 'regiao': 'Sul', 'lat': -30.0346, 'lon': -51.2177},
    'Rondônia': {'sigla': 'RO', 'regiao': 'Norte', 'lat': -11.5057, 'lon': -63.5806},
    'Roraima': {'sigla': 'RR', 'regiao': 'Norte', 'lat': 2.7376, 'lon': -62.0751},
    'Santa Catarina': {'sigla': 'SC', 'regiao': 'Sul', 'lat': -27.2423, 'lon': -50.2189},
    'São Paulo': {'sigla': 'SP', 'regiao': 'Sudeste', 'lat': -23.5505, 'lon': -46.6333},
    'Sergipe': {'sigla': 'SE', 'regiao': 'Nordeste', 'lat': -10.5741, 'lon': -37.3857},
    'Tocantins': {'sigla': 'TO', 'regiao': 'Norte', 'lat': -10.1753, 'lon': -48.2982}
}

# Parâmetros climáticos por região
PARAMETROS_CLIMA = {
    'Norte': {
        'temp_base': 27, 'temp_var': 3,
        'umidade_base': 80, 'umidade_var': 10,
        'precip_base': 200, 'precip_var': 100,
        'casos_base': 150
    },
    'Nordeste': {
        'temp_base': 28, 'temp_var': 4,
        'umidade_base': 70, 'umidade_var': 15,
        'precip_base': 100, 'precip_var': 80,
        'casos_base': 120
    },
    'Centro-Oeste': {
        'temp_base': 26, 'temp_var': 4,
        'umidade_base': 65, 'umidade_var': 15,
        'precip_base': 150, 'precip_var': 90,
        'casos_base': 100
    },
    'Sudeste': {
        'temp_base': 24, 'temp_var': 5,
        'umidade_base': 70, 'umidade_var': 12,
        'precip_base': 130, 'precip_var': 70,
        'casos_base': 200
    },
    'Sul': {
        'temp_base': 20, 'temp_var': 6,
        'umidade_base': 75, 'umidade_var': 10,
        'precip_base': 140, 'precip_var': 60,
        'casos_base': 60
    }
}

# Configurações de Machine Learning
ML_CONFIG = {
    'test_size': 0.3,
    'random_state': 42,
    'n_estimators': 100,
    'features_ml': ['temperatura_media', 'umidade_relativa', 'precipitacao',
                    'mes', 'temperatura_max', 'temperatura_min']
}

# Cores para visualizações
CORES_RISCO = {
    'Alto': '#e74c3c',
    'Médio': '#f39c12',
    'Baixo': '#27ae60'
}

CORES_RISCO_EMOJI = {
    'Alto': '🔴',
    'Médio': '🟡',
    'Baixo': '🟢'
}

# Nomes dos meses
MESES_NOMES = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
               'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

# Informações da aplicação
APP_INFO = {
    'title': 'Sistema de Predição de Risco de Dengue',
    'icon': '🦟',
    'author': 'Enzo Cabrera',
    'github': '@EnzoCabrera',
    'version': '2.0',
    'last_update': '2025-10-31'
}