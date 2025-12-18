# core/ui.py
import streamlit as st

def set_global_style():
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stSlider [data-baseweb="slider"] { padding: 0; }
            .metric-card {
                border-radius: 10px;
                padding: 15px;
                background-color: #f0f2f6;
                margin-bottom: 15px;
            }
            .analysis-card {
                border-radius: 10px;
                padding: 20px;
                background-color: #ffffff;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #2ecc71;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
