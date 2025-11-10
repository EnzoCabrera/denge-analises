import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backend.config import CORES_RISCO, MESES_NOMES

def _agregar_por_mes(df: pd.DataFrame) -> pd.DataFrame:

    #Agrega dados por mês/ano (para visualização)

    colunas_agrupar = ['ano', 'mes', 'mes_nome', 'ano_mes']
    colunas_existentes = [col for col in colunas_agrupar if col in df.columns]

    # Adicionar colunas opcionais se existirem
    for col in ['estado', 'sigla', 'regiao']:
        if col in df.columns:
            colunas_existentes.append(col)

    agg_dict = {
        'casos_dengue': 'sum',
        'temperatura_media': 'mean',
        'umidade_relativa': 'mean',
        'precipitacao': 'mean',
        'risco_dengue': lambda x: x.mode()[0] if len(x) > 0 else 'Médio'
    }

    # Adicionar colunas opcionais de temperatura
    if 'temperatura_max' in df.columns:
        agg_dict['temperatura_max'] = 'mean'
    if 'temperatura_min' in df.columns:
        agg_dict['temperatura_min'] = 'mean'

    return df.groupby(colunas_existentes).agg(agg_dict).reset_index()

def criar_grafico_casos_temporal(df: pd.DataFrame, estado_nome: str) -> go.Figure:
    """Gráfico de casos de dengue ao longo do tempo"""

    # AGREGAR DADOS
    df_agg = _agregar_por_mes(df)

    fig = go.Figure()

    for ano in df_agg['ano'].unique():
        df_ano = df_agg[df_agg['ano'] == ano]
        fig.add_trace(go.Scatter(
            x=df_ano['mes_nome'],
            y=df_ano['casos_dengue'],
            mode='lines+markers',
            name=f'{ano}',
            line=dict(width=3),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title=f'📈 Casos de Dengue por Mês - {estado_nome}',
        xaxis_title='Mês',
        yaxis_title='Número de Casos',
        hovermode='x unified',
        height=400,
        template='plotly_white',
        font=dict(size=12)
    )

    return fig


def criar_grafico_tendencia_anual(df: pd.DataFrame, estado_nome: str) -> go.Figure:
    #Tendência de casos por ano

    # AGREGAR DADOS
    df_agg = _agregar_por_mes(df)

    casos_ano = df_agg.groupby('ano')['casos_dengue'].sum().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=casos_ano['ano'],
        y=casos_ano['casos_dengue'],
        marker_color='#3498db',
        text=casos_ano['casos_dengue'],
        textposition='outside',
        texttemplate='%{text:.0f}'
    ))

    # Linha de tendência (apenas se houver mais de 1 ano)
    if len(casos_ano) > 1:
        z = np.polyfit(casos_ano['ano'], casos_ano['casos_dengue'], 1)
        p = np.poly1d(z)

        fig.add_trace(go.Scatter(
            x=casos_ano['ano'],
            y=p(casos_ano['ano']),
            mode='lines',
            name='Tendência',
            line=dict(color='red', width=2, dash='dash')
        ))

    fig.update_layout(
        title=f'📊 Total de Casos por Ano - {estado_nome}',
        xaxis_title='Ano',
        yaxis_title='Total de Casos',
        height=400,
        template='plotly_white',
        showlegend=True if len(casos_ano) > 1 else False
    )

    return fig

def criar_grafico_clima(df: pd.DataFrame, estado_nome: str) -> go.Figure:
    """Gráfico com variáveis climáticas"""

    # AGREGAR DADOS
    df_agg = _agregar_por_mes(df)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperatura Média', 'Umidade Relativa',
                        'Precipitação', 'Casos de Dengue'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(x=df_agg['ano_mes'], y=df_agg['temperatura_media'],
                   name='Temperatura', line=dict(color='#e74c3c', width=2)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df_agg['ano_mes'], y=df_agg['umidade_relativa'],
                   name='Umidade', line=dict(color='#3498db', width=2)),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=df_agg['ano_mes'], y=df_agg['precipitacao'],
               name='Precipitação', marker_color='#9b59b6'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=df_agg['ano_mes'], y=df_agg['casos_dengue'],
                   name='Casos', line=dict(color='#e67e22', width=2),
                   fill='tozeroy'),
        row=2, col=2
    )

    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_yaxes(title_text="Casos", row=2, col=2)

    fig.update_layout(
        title_text=f'📊 Indicadores Climáticos e Casos de Dengue - {estado_nome}',
        showlegend=False,
        height=600,
        template='plotly_white'
    )

    return fig


def criar_grafico_correlacao(df: pd.DataFrame, estado_nome: str) -> go.Figure:
    #Correlação entre clima e casos de dengue

    # Usar dados agregados se houver muitas amostras
    if len(df) > 100:
        df_plot = _agregar_por_mes(df)
    else:
        df_plot = df

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot['temperatura_media'],
        y=df_plot['casos_dengue'],
        mode='markers',
        marker=dict(
            size=df_plot['precipitacao'] / 10,
            color=df_plot['umidade_relativa'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Umidade %"),
            line=dict(width=1, color='white')
        ),
        text=[f"Temp: {t:.1f}°C<br>Casos: {c}<br>Precip: {p:.1f}mm<br>Umidade: {u:.1f}%"
              for t, c, p, u in zip(df_plot['temperatura_media'],
                                    df_plot['casos_dengue'],
                                    df_plot['precipitacao'],
                                    df_plot['umidade_relativa'])],
        hovertemplate='%{text}<extra></extra>',
        name=''
    ))

    fig.update_layout(
        title=f'🔬 Correlação: Temperatura × Casos (tamanho = precipitação) - {estado_nome}',
        xaxis_title='Temperatura Média (°C)',
        yaxis_title='Casos de Dengue',
        height=400,
        template='plotly_white'
    )

    return fig

def criar_grafico_risco_mensal(df: pd.DataFrame, estado_nome: str) -> go.Figure:
    #Heatmap de risco por mês/ano

    # AGREGAR DADOS
    df_agg = _agregar_por_mes(df)

    pivot = df_agg.pivot_table(
        values='casos_dengue',
        index='mes_nome',
        columns='ano',
        aggfunc='sum'
    )

    pivot = pivot.reindex(MESES_NOMES)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='YlOrRd',
        text=pivot.values,
        texttemplate='%{text:.0f}',
        textfont={"size": 12},
        colorbar=dict(title="Casos")
    ))

    fig.update_layout(
        title=f'🔥 Mapa de Calor - Casos de Dengue por Mês/Ano - {estado_nome}',
        xaxis_title='Ano',
        yaxis_title='Mês',
        height=400,
        template='plotly_white'
    )

    return fig


def criar_grafico_distribuicao_risco(df: pd.DataFrame, estado_nome: str) -> go.Figure:
    #Gráfico de pizza - distribuição de risco

    risco_counts = df['risco_dengue'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=risco_counts.index,
        values=risco_counts.values,
        hole=0.4,
        marker=dict(colors=[CORES_RISCO[r] for r in risco_counts.index]),
        textinfo='label+percent',
        textfont_size=14
    )])

    fig.update_layout(
        title=f'🎯 Distribuição de Risco - {estado_nome}',
        height=400,
        template='plotly_white'
    )

    return fig


def criar_grafico_modelos(df_resultados: pd.DataFrame) -> go.Figure:
    #Gráfico de comparação de modelos

    fig = go.Figure(data=[
        go.Bar(
            x=df_resultados['Modelo'],
            y=df_resultados['Acurácia'] * 100,
            text=[f"{v:.1f}%" for v in df_resultados['Acurácia'] * 100],
            textposition='outside',
            marker_color='#3498db'
        )
    ])

    fig.update_layout(
        title='Comparação de Modelos - Acurácia',
        xaxis_title='Modelo',
        yaxis_title='Acurácia (%)',
        height=400,
        template='plotly_white'
    )

    return fig

def criar_mapa_brasil(estados_df: pd.DataFrame) -> go.Figure:
    #Mapa do Brasil com estados

    fig = px.scatter_geo(
        estados_df,
        lat='lat',
        lon='lon',
        hover_name='estado',
        color='regiao',
        size=[10] * len(estados_df),
        projection='mercator',
        title='Estados Brasileiros - Clique na barra lateral para analisar'
    )

    fig.update_geos(
        center=dict(lat=-14, lon=-55),
        projection_scale=3.5,
        showcountries=True,
        showland=True,
        landcolor='lightgray'
    )

    return fig

def criar_grafico_predicao_mes_atual(predicao: dict, estado_nome: str) -> go.Figure:

    #Gráfico de predição para o mês atual

    from datetime import datetime
    mes_atual_nome = datetime.now().strftime('%B/%Y')

    # Cores baseadas no risco
    cores = {
        'Alto': '#e74c3c',
        'Médio': '#f39c12',
        'Baixo': '#27ae60'
    }
    cor = cores.get(predicao['risco_previsto'], '#3498db')

    fig = go.Figure()

    # Barra histórica
    fig.add_trace(go.Bar(
        x=['Histórico<br>Média'],
        y=[predicao['casos_historicos_media']],
        name='Histórico',
        marker_color='#95a5a6',
        text=[f"{predicao['casos_historicos_media']}"],
        textposition='outside'
    ))

    # Barra de predição com intervalo de confiança
    fig.add_trace(go.Bar(
        x=['Predição<br>Atual'],
        y=[predicao['casos_previstos']],
        name='Predição',
        marker_color=cor,
        text=[f"{predicao['casos_previstos']}"],
        textposition='outside',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[predicao['intervalo_superior'] - predicao['casos_previstos']],
            arrayminus=[predicao['casos_previstos'] - predicao['intervalo_inferior']],
            color='rgba(0,0,0,0.3)'
        )
    ))

    fig.update_layout(
        title=f'🔮 Predição de Casos de Dengue - {mes_atual_nome} - {estado_nome}',
        yaxis_title='Número de Casos',
        showlegend=True,
        height=400,
        template='plotly_white'
    )

    return fig


def criar_grafico_serie_temporal_com_predicao(df_historico: pd.DataFrame,
                                              predicao: dict,
                                              estado_nome: str) -> go.Figure:

    #Série temporal histórica + predição do mês atual


    from datetime import datetime

    # Agregar histórico por mês
    df_agg = df_historico.groupby(['ano', 'mes', 'ano_mes']).agg({
        'casos_dengue': 'sum'
    }).reset_index()

    # Últimos 12 meses
    df_ultimos_12 = df_agg.tail(12)

    fig = go.Figure()

    # Linha histórica
    fig.add_trace(go.Scatter(
        x=df_ultimos_12['ano_mes'],
        y=df_ultimos_12['casos_dengue'],
        mode='lines+markers',
        name='Histórico',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8)
    ))

    # Ponto de predição
    mes_atual = datetime.now().strftime('%Y-%m')

    fig.add_trace(go.Scatter(
        x=[mes_atual],
        y=[predicao['casos_previstos']],
        mode='markers',
        name='Predição',
        marker=dict(
            size=15,
            color='#e74c3c',
            symbol='star',
            line=dict(color='white', width=2)
        ),
        error_y=dict(
            type='data',
            symmetric=False,
            array=[predicao['intervalo_superior'] - predicao['casos_previstos']],
            arrayminus=[predicao['casos_previstos'] - predicao['intervalo_inferior']],
            color='rgba(231, 76, 60, 0.3)',
            thickness=2
        )
    ))

    fig.update_layout(
        title=f'📈 Série Temporal com Predição - {estado_nome}',
        xaxis_title='Período',
        yaxis_title='Casos de Dengue',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )

    return fig


def criar_grafico_comparacao_predicao_historico(predicao: dict,
                                                df_historico: pd.DataFrame) -> go.Figure:

    #Compara predição com mesmos meses históricos


    from datetime import datetime
    mes_atual = datetime.now().month

    # Filtrar histórico do mesmo mês
    df_mesmo_mes = df_historico[df_historico['mes'] == mes_atual]

    # Agrupar por ano
    casos_por_ano = df_mesmo_mes.groupby('ano')['casos_dengue'].sum().reset_index()

    fig = go.Figure()

    # Barras históricas
    fig.add_trace(go.Bar(
        x=casos_por_ano['ano'].astype(str),
        y=casos_por_ano['casos_dengue'],
        name='Histórico',
        marker_color='#95a5a6'
    ))

    # Barra de predição
    fig.add_trace(go.Bar(
        x=['2025 (Predição)'],
        y=[predicao['casos_previstos']],
        name='Predição 2025',
        marker_color='#e74c3c',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[predicao['intervalo_superior'] - predicao['casos_previstos']],
            arrayminus=[predicao['casos_previstos'] - predicao['intervalo_inferior']]
        )
    ))

    # Linha de média histórica
    if len(casos_por_ano) > 0:
        media_historica = casos_por_ano['casos_dengue'].mean()

        fig.add_hline(
            y=media_historica,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Média Histórica: {int(media_historica)}",
            annotation_position="top left"
        )

    fig.update_layout(
        title=f'📊 Comparação: Mesmo Mês em Anos Anteriores',
        xaxis_title='Ano',
        yaxis_title='Casos de Dengue',
        showlegend=True,
        height=400,
        template='plotly_white'
    )

    return fig