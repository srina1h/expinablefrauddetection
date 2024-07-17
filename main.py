import time
import torch
import openai
import streamlit as st

def option_choice():
    genre = st.radio(
        "Do you want to select a sample text file OR upload your own?",
        ('Select Sample text file', 'Upload my own')
    )
    if genre == "Select Sample text file": flag=False
    else: flag = True
    return flag

if __name__ == "__main__":
    st.set_page_config(
        page_title="Explainable Fraud Detection",
    )
    query = st.text_input("Enter your Query")