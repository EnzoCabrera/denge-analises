"""
Aplicação Principal - Dashboard de Predição de Dengue
Autor: Enzo Cabrera (@EnzoCabrera)
Data: 2025-10-31
Versão: 2.0 - Com integração INMET
"""

import streamlit as st
import pandas as pd

# Imports dos módulos backend
from backend.config import APP_INFO
from backend.data_generator import gerar_dados_estado, calcular_estatisticas
from backend.models import ModeloDengue

# Imports dos módulos frontend
from frontend.components import (
    renderizar_header, renderizar_sidebar, renderizar_kpis,
    renderizar_estatisticas_risco, renderizar_footer, renderizar_ranking_modelos,
    renderizar_info_dados
)
from frontend.charts import (
    criar_grafico_casos_temporal, criar_grafico_clima,
    criar_grafico_risco_mensal, criar_grafico_distribuicao_risco,
    criar_grafico_correlacao, criar_grafico_tendencia_anual,
    criar_grafico_modelos, criar_mapa_brasil
)
from frontend.styles import aplicar_estilos

# Imports dos utilitários
from utils.helpers import preparar_dados_mapa, exportar_csv


# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

st.set_page_config(
    page_title=APP_INFO['title'],
    page_icon=APP_INFO['icon'],
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# FUNÇÃO PRINCIPAL
# =====================================================

def main():
    """Função principal da aplicação"""

    # Aplicar estilos
    aplicar_estilos()

    # Renderizar header
    renderizar_header()

    # Renderizar sidebar e obter seleções
    estado_selecionado, n_anos, analisar, usar_dados_reais = renderizar_sidebar()

    # Área principal
    if analisar or 'dados_carregados' in st.session_state:

        # Carregar dados (REAIS ou SIMULADOS)
        with st.spinner(f'⏳ Carregando dados de {estado_selecionado}...'):
            try:
                df = gerar_dados_estado(estado_selecionado, n_anos, usar_dados_reais)

                if df is None or len(df) == 0:
                    st.error("❌ Erro ao carregar dados. Tente novamente ou use dados simulados.")
                    return

                stats = calcular_estatisticas(df)
                st.session_state['dados_carregados'] = True
                st.session_state['dados_reais'] = usar_dados_reais
                st.session_state['total_registros'] = len(df)

            except Exception as e:
                st.error(f"❌ Erro ao processar dados: {str(e)}")
                st.exception(e)
                return

        # Informação sobre fonte de dados
        renderizar_info_dados(
            st.session_state.get('dados_reais', False),
            st.session_state.get('total_registros', 0)
        )

        st.markdown("---")

        # Renderizar KPIs
        renderizar_kpis(stats)

        st.markdown("---")

        # Tabs com análises
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Análise Temporal",
            "🌡️ Indicadores Climáticos",
            "🎯 Análise de Risco",
            "🤖 Modelo Preditivo"
        ])

        # TAB 1: Análise Temporal
        with tab1:
            st.markdown("### 📈 Evolução Temporal dos Casos")

            try:
                st.plotly_chart(
                    criar_grafico_casos_temporal(df, estado_selecionado),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Erro ao criar gráfico temporal: {str(e)}")

            col1, col2 = st.columns(2)

            with col1:
                try:
                    st.plotly_chart(
                        criar_grafico_tendencia_anual(df, estado_selecionado),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro ao criar gráfico de tendência: {str(e)}")

            with col2:
                try:
                    st.plotly_chart(
                        criar_grafico_risco_mensal(df, estado_selecionado),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro ao criar mapa de calor: {str(e)}")

        # TAB 2: Indicadores Climáticos
        with tab2:
            st.markdown("### 🌡️ Dados Climáticos")

            try:
                st.plotly_chart(
                    criar_grafico_clima(df, estado_selecionado),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Erro ao criar gráfico climático: {str(e)}")

            try:
                st.plotly_chart(
                    criar_grafico_correlacao(df, estado_selecionado),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Erro ao criar gráfico de correlação: {str(e)}")

        # TAB 3: Análise de Risco
        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                try:
                    st.plotly_chart(
                        criar_grafico_distribuicao_risco(df, estado_selecionado),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro ao criar gráfico de distribuição: {str(e)}")

            with col2:
                try:
                    renderizar_estatisticas_risco(df)
                except Exception as e:
                    st.error(f"Erro ao calcular estatísticas de risco: {str(e)}")

        # TAB 4: Modelo Preditivo
        with tab4:
            st.markdown("### 🤖 Treinamento do Modelo Preditivo")

            with st.spinner("Treinando modelos de Machine Learning..."):
                try:
                    modelo = ModeloDengue()
                    df_resultados = modelo.treinar_modelos(df)

                    if df_resultados is None or len(df_resultados) == 0:
                        st.error("❌ Erro ao treinar modelos.")
                        return

                except Exception as e:
                    st.error(f"❌ Erro no treinamento: {str(e)}")
                    st.exception(e)
                    return

            col1, col2 = st.columns([2, 1])

            with col1:
                try:
                    st.plotly_chart(
                        criar_grafico_modelos(df_resultados),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro ao criar gráfico de modelos: {str(e)}")

            with col2:
                try:
                    renderizar_ranking_modelos(df_resultados)
                except Exception as e:
                    st.error(f"Erro ao renderizar ranking: {str(e)}")

            # Métricas adicionais
            st.markdown("---")
            st.markdown("### 📊 Métricas Detalhadas")

            try:
                st.dataframe(
                    df_resultados.style.format({
                        'Acurácia': '{:.2%}',
                        'F1-Score': '{:.3f}',
                        'CV Acurácia': '{:.2%}'
                    }),
                    use_container_width=True
                )
            except Exception as e:
                st.dataframe(df_resultados, use_container_width=True)

        # Dados brutos (expansível)
        with st.expander("📋 Ver Dados Brutos"):
            try:
                st.markdown(f"**Total de registros:** {len(df):,}")
                st.dataframe(df, use_container_width=True)

                # Botão de download
                csv = exportar_csv(df, estado_selecionado)
                st.download_button(
                    label="📥 Baixar dados em CSV",
                    data=csv,
                    file_name=f'dados_dengue_{estado_selecionado.lower().replace(" ", "_")}_{n_anos}anos.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Erro ao exibir dados brutos: {str(e)}")

    else:
        # Tela inicial (quando não foi executada análise)
        st.info("👈 Selecione um estado na barra lateral e clique em '🚀 Executar Análise Completa'")

        st.markdown("### 🗺️ Estados Disponíveis para Análise")

        try:
            estados_df = preparar_dados_mapa()
            st.plotly_chart(
                criar_mapa_brasil(estados_df),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Erro ao criar mapa: {str(e)}")

        # Informações adicionais
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🌐 Dados do INMET
            
            O sistema utiliza dados climáticos **reais** da API do INMET:
            - ✅ Temperatura
            - ✅ Umidade
            - ✅ Precipitação
            - ✅ Vento
            """)

        with col2:
            st.markdown("""
            ### 🤖 Machine Learning
            
            Modelos disponíveis:
            - 📊 Naive Bayes
            - 🌳 Random Forest
            - 📈 Gradient Boosting
            - 🚀 XGBoost (se disponível)
            """)

        with col3:
            st.markdown("""
            ### 📊 Visualizações
            
            Gráficos interativos:
            - 📈 Série temporal
            - 🗺️ Mapa de calor
            - 🎯 Distribuição de risco
            - 🔬 Correlações
            """)

    # Footer
    renderizar_footer()


# =====================================================
# PONTO DE ENTRADA
# =====================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ Erro crítico na aplicação!")
        st.exception(e)
        st.info("💡 Tente recarregar a página (F5) ou limpar o cache (Settings > Clear cache)")