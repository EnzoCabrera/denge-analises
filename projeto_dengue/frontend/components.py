"""
Componentes reutilizáveis da interface
"""

import streamlit as st
from backend.config import ESTADOS_BRASIL, CORES_RISCO_EMOJI


def renderizar_header():
    """Renderiza o cabeçalho da aplicação"""
    st.markdown("""
        <h1 style='text-align: center; color: #2c3e50;'>
            🦟 Sistema de Predição de Risco de Dengue
        </h1>
        <p style='text-align: center; color: #7f8c8d; font-size: 18px;'>
            Análise preditiva de casos de dengue por estado brasileiro
        </p>
        <hr style='margin-bottom: 30px;'>
    """, unsafe_allow_html=True)


def renderizar_sidebar():
    """
    Renderiza a barra lateral com controles

    Returns:
        Tupla (estado_selecionado, n_anos, analisar, usar_dados_reais)
    """
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
        st.markdown("## 🎛️ Painel de Controle")

        # Seleção do estado
        estado_selecionado = st.selectbox(
            "📍 Selecione o Estado:",
            options=sorted(ESTADOS_BRASIL.keys()),
            index=list(sorted(ESTADOS_BRASIL.keys())).index('São Paulo')
        )

        # Seleção de período
        n_anos = st.slider("📅 Período de Análise (anos):", 1, 5, 3)

        st.markdown("---")

        # NOVO: Opção de usar dados reais
        usar_dados_reais = st.checkbox(
            "🌐 Usar dados REAIS do Open-Meteo",
            value=True,
            help="Se marcado, busca dados climáticos reais. Se desmarcar, usa simulação."
        )

        if usar_dados_reais:
            st.info("💡 Dados climáticos virão do Open-Meteo (global, confiável)")
        else:
            st.warning("⚠️ Modo simulação ativado")

        st.markdown("---")

        # Informações do estado
        info = ESTADOS_BRASIL[estado_selecionado]
        st.markdown(f"""
        ### 📋 Informações
        **Estado:** {estado_selecionado}  
        **Sigla:** {info['sigla']}  
        **Região:** {info['regiao']}  
        **Latitude:** {info['lat']:.4f}  
        **Longitude:** {info['lon']:.4f}
        """)

        st.markdown("---")

        # Botão de análise
        analisar = st.button("🚀 Executar Análise Completa", type="primary", use_container_width=True)

        # Informações adicionais
        st.markdown("---")
        st.markdown("""
        ### ℹ️ Sobre os Dados
        
        **Dados Reais (Open-Meteo):**
        - Temperatura
        - Umidade
        - Precipitação
        - Vento
        
        **Dados Simulados:**
        - Casos de dengue (baseados em clima)
        - Classificação de risco
        """)

    return estado_selecionado, n_anos, analisar, usar_dados_reais


def renderizar_kpis(stats: dict):
    """
    Renderiza os KPIs principais

    Args:
        stats: Dicionário com estatísticas
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total de Casos",
            value=f"{stats['total_casos']:,}",
            delta=f"{stats['casos_ultimo_mes']} no último mês"
        )

    with col2:
        st.metric(
            label="🌡️ Temp. Média",
            value=f"{stats['media_temp']:.1f}°C"
        )

    with col3:
        st.metric(
            label="💧 Umidade Média",
            value=f"{stats['media_umidade']:.1f}%"
        )

    with col4:
        risco = stats['risco_predominante']
        st.metric(
            label="⚠️ Risco Predominante",
            value=f"{CORES_RISCO_EMOJI[risco]} {risco}"
        )


def renderizar_estatisticas_risco(df):
    """
    Renderiza estatísticas por nível de risco

    Args:
        df: DataFrame com dados
    """
    st.markdown("### 📊 Estatísticas por Nível de Risco")

    for risco in ['Alto', 'Médio', 'Baixo']:
        df_risco = df[df['risco_dengue'] == risco]
        if len(df_risco) > 0:
            casos_risco = df_risco['casos_dengue'].sum()
            pct = (len(df_risco) / len(df)) * 100

            # Emoji por risco
            emoji = {'Alto': '🔴', 'Médio': '🟡', 'Baixo': '🟢'}

            st.markdown(f"""
            {emoji[risco]} **{risco}:**  
            - Ocorrências: {len(df_risco)} registros ({pct:.1f}%)  
            - Total de casos: {casos_risco:,}  
            - Média de casos: {df_risco['casos_dengue'].mean():.0f}
            """)


def renderizar_footer():
    """Renderiza o rodapé da aplicação"""
    st.markdown("---")
    st.markdown(f"""
        <p style='text-align: center; color: #7f8c8d;'>
            Desenvolvido por <b>Enzo Cabrera</b> (@EnzoCabrera) | 
            Dados Climáticos: API Open-Meteo | Dados de Dengue: Simulados | 
            Última atualização: 2025-10-31
        </p>
    """, unsafe_allow_html=True)


def renderizar_ranking_modelos(df_resultados):
    """
    Renderiza ranking de modelos

    Args:
        df_resultados: DataFrame com resultados dos modelos
    """
    st.markdown("### 🏆 Melhor Modelo")
    melhor = df_resultados.iloc[0]
    st.success(f"**{melhor['Modelo']}**")
    st.metric("Acurácia", f"{melhor['Acurácia']*100:.2f}%")

    # Mostrar F1-Score se disponível
    if 'F1-Score' in df_resultados.columns:
        st.metric("F1-Score", f"{melhor['F1-Score']:.3f}")

    st.markdown("### 📋 Ranking Completo")

    for idx, row in df_resultados.iterrows():
        # Emoji de medalha
        if idx == 0:
            medalha = "🥇"
        elif idx == 1:
            medalha = "🥈"
        elif idx == 2:
            medalha = "🥉"
        else:
            medalha = f"{idx+1}."

        acuracia_pct = row['Acurácia'] * 100

        if 'F1-Score' in row:
            st.write(f"{medalha} {row['Modelo']}: {acuracia_pct:.1f}% (F1: {row['F1-Score']:.3f})")
        else:
            st.write(f"{medalha} {row['Modelo']}: {acuracia_pct:.1f}%")


def renderizar_info_dados(usar_dados_reais: bool, total_registros: int):
    """
    Renderiza informações sobre a fonte dos dados

    Args:
        usar_dados_reais: Se está usando dados reais
        total_registros: Número total de registros
    """
    if usar_dados_reais:
        st.info(f"""
        🌐 **Dados Climáticos REAIS do Open-Meteo**  
        Total de registros: {total_registros:,}  
        Fonte: Instituto Nacional de Meteorologia  
        
        ⚠️ Casos de dengue são simulados baseados nas condições climáticas reais.
        """)
    else:
        st.warning(f"""
        🎲 **Dados Simulados**  
        Total de registros: {total_registros:,}  
        Dados gerados algoritmicamente para fins educacionais.
        """)