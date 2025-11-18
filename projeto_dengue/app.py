import streamlit as st
import pandas as pd
from datetime import datetime

# Imports dos módulos backend
from backend.config import APP_INFO, CORES_RISCO_EMOJI
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
    criar_grafico_modelos, criar_mapa_brasil,
    criar_grafico_predicao_mes_atual,
    criar_grafico_serie_temporal_com_predicao,
    criar_grafico_comparacao_predicao_historico
)
from frontend.styles import aplicar_estilos

# Imports dos utilitários
from utils.helpers import preparar_dados_mapa, exportar_csv

# Imports de predição
from backend.predicao import PredicaoDengue, obter_clima_atual_estimado

st.set_page_config(
    page_title=APP_INFO['title'],
    page_icon=APP_INFO['icon'],
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            "🤖 Modelo Preditivo",
            "🔮 Predição Mês Atual"
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
                st.error(f"❌ Erro ao criar gráfico temporal: {str(e)}")

            st.markdown("---")

            try:
                st.plotly_chart(
                    criar_grafico_tendencia_anual(df, estado_selecionado),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Erro ao criar gráfico de tendência: {str(e)}")

            st.markdown("---")

            st.markdown("#### 📊 Resumo Estatístico")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total de Casos", f"{df['casos_dengue'].sum():,}")

            with col2:
                st.metric("Média Mensal", f"{df['casos_dengue'].mean():,.0f}")

            with col3:
                st.metric("Maior Surto", f"{df['casos_dengue'].max():,}")

            with col4:
                st.metric("Menor Registro", f"{df['casos_dengue'].min():,}")

        # TAB 2: Análise Climática
        with tab2:
            st.markdown("### 🌤️ Análise de Fatores Climáticos")

            st.markdown("#### 📊 Resumo Climático do Período")

            col1, col2, col3 = st.columns(3)

            with col1:
                temp_media = df['temperatura_media'].mean()
                temp_std = df['temperatura_media'].std()
                st.metric("🌡️ Temperatura Média", f"{temp_media:.1f}°C", f"± {temp_std:.1f}°C")

            with col2:
                umid_media = df['umidade_relativa'].mean()
                umid_std = df['umidade_relativa'].std()
                st.metric("💧 Umidade Relativa Média", f"{umid_media:.1f}%", f"± {umid_std:.1f}%")

            with col3:
                precip_media = df['precipitacao'].mean()
                precip_std = df['precipitacao'].std()
                st.metric("☔ Precipitação Média", f"{precip_media:.1f}mm/mês", f"± {precip_std:.1f}mm")

            st.markdown("---")

            try:
                st.plotly_chart(
                    criar_grafico_clima(df, estado_selecionado),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Erro ao criar gráfico climático: {str(e)}")

            st.markdown("---")

            st.markdown("#### 📋 Estatísticas Detalhadas")

            try:
                resumo = df[['temperatura_media', 'temperatura_max', 'temperatura_min',
                             'umidade_relativa', 'precipitacao']].describe()
                resumo = resumo.round(2)
                resumo.columns = ['Temp. Média (°C)', 'Temp. Máx (°C)', 'Temp. Mín (°C)',
                                  'Umidade (%)', 'Precipitação (mm)']

                st.dataframe(resumo, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erro ao gerar estatísticas: {str(e)}")

        # TAB 3: Modelo Preditivo
        with tab3:
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

            if modelo.tipo_modelo == 'regressao':
                st.info("""
                ℹ️ **Modelo de Regressão Ativado**

                Como os dados apresentam correlação baixa entre features e target,
                o sistema está usando **modelos de regressão** para prever o **número de casos**
                em vez da classificação de risco.
                """)

            col1, col2 = st.columns([2, 1])

            with col1:
                try:
                    st.plotly_chart(
                        criar_grafico_modelos(df_resultados),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Erro ao criar gráfico de modelos: {str(e)}")

            with col2:
                try:
                    renderizar_ranking_modelos(df_resultados)
                except Exception as e:
                    st.error(f"❌ Erro ao renderizar ranking: {str(e)}")

            st.markdown("---")
            st.markdown("### 📊 Métricas Detalhadas")

            try:
                if modelo.tipo_modelo == 'regressao':
                    st.dataframe(
                        df_resultados.style.format({
                            'Acurácia': '{:.2%}',
                            'R²': '{:.3f}',
                            'MAE': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                else:
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

        # TAB 4: Predição do Mês Atual
        with tab4:
            st.markdown("### 🔮 Predição de Casos para o Mês Atual")

            try:
                with st.spinner("🤖 Gerando predição..."):
                    modelo_predicao = PredicaoDengue()
                    resultado_treino = modelo_predicao.treinar_modelo(df)
                    clima_atual = obter_clima_atual_estimado(estado_selecionado)
                    predicao = modelo_predicao.prever_mes_atual(df, clima_atual)

                # Métricas principais
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Casos Previstos",
                        f"{int(predicao['casos_previstos']):,}",
                        f"Intervalo: {int(predicao['intervalo_inferior']):,} - {int(predicao['intervalo_superior']):,}"
                    )

                with col2:
                    st.metric(
                        "Modelo Usado",
                        predicao['modelo_usado'],
                        f"R²: {predicao['confianca']:.3f}"
                    )

                # Gráficos
                st.markdown("---")
                st.markdown("### 📊 Visualizações")

                col1, col2 = st.columns(2)

                with col1:
                    try:
                        st.plotly_chart(
                            criar_grafico_predicao_mes_atual(predicao, estado_selecionado),
                            use_container_width=True
                        )
                    except:
                        pass

                with col2:
                    try:
                        st.plotly_chart(
                            criar_grafico_comparacao_predicao_historico(predicao, df),
                            use_container_width=True
                        )
                    except:
                        pass

                try:
                    st.plotly_chart(
                        criar_grafico_serie_temporal_com_predicao(df, predicao, estado_selecionado),
                        use_container_width=True
                    )
                except:
                    pass

                # Métricas do modelo
                st.markdown("---")
                st.markdown("### 📈 Métricas do Modelo")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Modelo", predicao['modelo_usado'])

                with col2:
                    st.metric("R² Score", f"{predicao['confianca']:.3f}")

                with col3:
                    st.metric("Erro Médio (MAE)", f"{resultado_treino['mae']:.0f} casos")

                # Detalhes
                with st.expander("📊 Detalhes da Predição"):
                    st.json(predicao)

                with st.expander("📈 Resultados do Treino"):
                    st.dataframe(resultado_treino['resultados'])

            except Exception as e:
                st.error(f"❌ Erro na predição: {str(e)}")

        # Dados brutos (expansível)
        with st.expander("📋 Ver Dados Brutos"):
            try:
                st.markdown(f"**Total de registros:** {len(df):,}")
                st.dataframe(df, use_container_width=True)

                csv = exportar_csv(df, estado_selecionado)
                st.download_button(
                    label="📥 Baixar dados em CSV",
                    data=csv,
                    file_name=f'dados_dengue_{estado_selecionado.lower().replace(" ", "_")}_{n_anos}anos.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"❌ Erro ao exibir dados brutos: {str(e)}")

    else:
        # Tela inicial
        st.info("👈 Selecione um estado na barra lateral e clique em '🚀 Executar Análise Completa'")

        st.markdown("### 🗺️ Estados Disponíveis para Análise")

        try:
            estados_df = preparar_dados_mapa()
            st.plotly_chart(
                criar_mapa_brasil(estados_df),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"❌ Erro ao criar mapa: {str(e)}")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🌐 Dados Climáticos

            Fonte: **Open-Meteo API**
            - ✅ Temperatura
            - ✅ Umidade
            - ✅ Precipitação
            - ✅ Dados históricos reais
            """)

        with col2:
            st.markdown("""
            ### 🤖 Machine Learning

            Modelos disponíveis:
            - 📊 Ridge & Lasso
            - 🌳 Random Forest
            - 📈 Gradient Boosting
            - 🚀 XGBoost
            """)

        with col3:
            st.markdown("""
            ### 🔮 Predição

            Sistema preditivo:
            - 📈 Série temporal
            - 🎯 Predição mês atual
            - 📊 Intervalo de confiança
            - ⚠️ Alertas automáticos
            """)

    # Footer
    renderizar_footer()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ Erro crítico na aplicação!")
        st.exception(e)