"""
Estilos CSS da aplicação
"""

CSS_CUSTOM = """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
"""


def aplicar_estilos():
    """Aplica estilos CSS customizados"""
    import streamlit as st
    st.markdown(CSS_CUSTOM, unsafe_allow_html=True)