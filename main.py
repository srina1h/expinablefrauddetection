import time
import torch
import openai
import streamlit as st

if __name__ == "__main__":
    st.set_page_config(
        page_title="Explainable Fraud Detection",
    )
    query = st.text_input("Enter your Query")