# Explainable Fraud detection using LLMs
An XGBoost model predicts fraud based on a credit card fraud simulation dataset. If fraud is detected, the most important features in the prediction are understood through SHAP analysis and the rules in the XGBoost decision tree is extracted using te2rules.

All of this information along with the user's last 100 transactions are fed to a LLM to explain why the model has detected the transaction as fraud.

## Requirements

Install requirements using requirements.txt

## Assets

Running requires the download of the assets folder which can be found at - [Assets](https://drive.google.com/file/d/1G7EBI3_qHfydQkKV1QSgCgf2GS_n2K02/view?usp=sharing)

This needs to be placed in the working directory

## Langsmith integration

Create a Langsmith account and get your API key.

## .env setup

Paste openAI key and Langsmith API key in the .env file

## Running the app

In the working directory - streamlit run main.py
