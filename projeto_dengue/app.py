import streamlit as st
import pandas as pd
from datetime import datetime

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
    #Função principal da aplicação

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

        # Informação sobre fonte de dados (antiga)
        renderizar_info_dados(
            st.session_state.get('dados_reais', False),
            st.session_state.get('total_registros', 0)
        )

        st.markdown("---")

        from frontend.components import renderizar_fonte_dados
        renderizar_fonte_dados(df)

        st.markdown("---")

        # Renderizar KPIs
        renderizar_kpis(stats)

        st.markdown("---")

        # Tabs com análises (COM TAB DE PREDIÇÃO)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Análise Temporal",
            "🌡️ Indicadores Climáticos",
            "🎯 Análise de Risco",
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

            # =====================================================
            # EXIBIR TIPO DE MODELO USADO
            # =====================================================

            if modelo.tipo_modelo == 'regressao':
                st.info("""
                ℹ️ **Modelo de Regressão Ativado**

                Como os dados apresentam apenas uma classe de risco ou poucos dados,
                o sistema está usando **modelos de regressão** para prever o **número de casos**
                em vez da classificação de risco.
                """)

            # =====================================================

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
                # Formatar baseado no tipo de modelo
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

        # TAB 5: Predição do Mês Atual
        with tab5:
            st.markdown("### 🔮 Predição de Casos para o Mês Atual")

            with st.spinner("🤖 Treinando modelo preditivo..."):
                try:
                    # Criar modelo de predição
                    modelo_predicao = PredicaoDengue()

                    # Treinar com dados históricos
                    resultado_treino = modelo_predicao.treinar_modelo(df)

                    # Obter clima atual (estimado)
                    clima_atual = obter_clima_atual_estimado(estado_selecionado)

                    # Fazer predição
                    predicao = modelo_predicao.prever_mes_atual(df, clima_atual)

                except Exception as e:
                    st.error(f"❌ Erro na predição: {str(e)}")
                    st.exception(e)

            # Exibir resultados
            # Exibir resultados
            if 'predicao' in locals():

                # Total de amostras por mês no dataset
                df_mes_atual = df[df['mes'] == datetime.now().month]
                n_amostras_mes = len(df_mes_atual)
                n_anos_dados = len(df['ano'].unique())

                # Casos agregados (soma de todas as localidades simuladas)
                casos_agregados_mes = df_mes_atual.groupby('ano')['casos_dengue'].sum().mean()

                # Escalar predição para ser comparável
                if n_anos_dados > 0:
                    amostras_por_mes = n_amostras_mes / n_anos_dados
                else:
                    amostras_por_mes = 1

                casos_previstos_total = int(predicao['casos_previstos'] * amostras_por_mes)
                intervalo_inf_total = int(predicao['intervalo_inferior'] * amostras_por_mes)
                intervalo_sup_total = int(predicao['intervalo_superior'] * amostras_por_mes)

                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 30px; border-radius: 15px; color: white; margin-bottom: 30px;'>
                    <h2 style='margin: 0; color: white;'>🔮 Predição para {datetime.now().strftime('%B/%Y')}</h2>
                    <hr style='border-color: rgba(255,255,255,0.3); margin: 20px 0;'>
                    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;'>
                        <div>
                            <p style='margin: 0; opacity: 0.9; font-size: 14px;'>CASOS PREVISTOS (TOTAL)</p>
                            <h1 style='margin: 10px 0; font-size: 48px;'>{casos_previstos_total:,}</h1>
                            <p style='margin: 0; opacity: 0.8; font-size: 12px;'>
                                Intervalo: {intervalo_inf_total:,} - {intervalo_sup_total:,}
                            </p>
                        </div>
                        <div>
                            <p style='margin: 0; opacity: 0.9; font-size: 14px;'>RISCO PREVISTO</p>
                            <h1 style='margin: 10px 0; font-size: 48px;'>{predicao['risco_previsto']}</h1>
                            <p style='margin: 0; opacity: 0.8; font-size: 12px;'>
                                Modelo: {predicao['modelo_usado']}
                            </p>
                        </div>
                        <div>
                            <p style='margin: 0; opacity: 0.9; font-size: 14px;'>VARIAÇÃO vs HISTÓRICO</p>
                            <h1 style='margin: 10px 0; font-size: 48px;'>
                                {((casos_previstos_total - casos_agregados_mes) / casos_agregados_mes * 100):+.1f}%
                            </h1>
                            <p style='margin: 0; opacity: 0.8; font-size: 12px;'>
                                Confiança (R²): {predicao['confianca']:.2%}
                            </p>
                        </div>
                    </div>
                    <div style='margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); 
                                border-radius: 10px; font-size: 16px;'>
                        {predicao['alerta']}
                    </div>
                    <div style='margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.05); 
                                border-radius: 10px; font-size: 14px; opacity: 0.9;'>
                        💡 <b>Metodologia:</b> Predição baseada em {int(amostras_por_mes)} localidades simuladas | 
                        Média histórica: <b>{int(casos_agregados_mes):,}</b> casos/mês
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Gráficos de predição
                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(
                        criar_grafico_predicao_mes_atual(predicao, estado_selecionado),
                        use_container_width=True
                    )

                with col2:
                    st.plotly_chart(
                        criar_grafico_comparacao_predicao_historico(predicao, df),
                        use_container_width=True
                    )

                # Série temporal com predição
                st.plotly_chart(
                    criar_grafico_serie_temporal_com_predicao(df, predicao, estado_selecionado),
                    use_container_width=True
                )

                # Métricas do modelo
                st.markdown("---")
                st.markdown("### 📊 Métricas do Modelo Preditivo")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Modelo Usado", predicao['modelo_usado'])

                with col2:
                    st.metric("R² Score", f"{predicao['confianca']:.3f}")

                with col3:
                    mae = resultado_treino['mae']
                    st.metric("Erro Médio (MAE)", f"{mae:.0f} casos")

                # Tabela com resultados de treino
                with st.expander("📈 Ver Desempenho de Todos os Modelos"):
                    st.dataframe(
                        resultado_treino['resultados'].style.format({
                            'MAE': '{:.2f}',
                            'R²': '{:.3f}',
                            'RMSE': '{:.2f}'
                        }),
                        use_container_width=True
                    )

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
            st.error(f"Erro ao criar mapa: {str(e)}")

        # Informações adicionais
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
            - 📊 Naive Bayes
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
        st.info("💡 Tente recarregar a página (F5) ou limpar o cache (Settings > Clear cache)")