# Explainable Fraud detection using LLMs
An XGBoost model predicts fraud based on a credit card fraud simulation dataset. If fraud is detected, the most important features in the prediction are understood through SHAP analysis and the rules in the XGBoost decision tree is extracted using te2rules.

All of this information along with the user's last 100 transactions are fed to a LLM to explain why the model has detected the transaction as fraud.

## Requirements

Install requirements using requirements.txt

This notebook uses AI explainability libraries [SHAP](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html) & [TE2Rules](https://github.com/linkedin/TE2Rules)

Tested on Python 3.11.3 + (May not work in previous versions)

## Assets

Dataset can be found at - [Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

The base model training and the preprocessing can be found here - [Colab notebook link](https://colab.research.google.com/drive/1vuu-INf_eJUDAUXqZK2JuDKx5gDxIuDy?usp=sharing)

Running requires the download of the assets folder which can be found at - [Assets](https://drive.google.com/file/d/1G7EBI3_qHfydQkKV1QSgCgf2GS_n2K02/view?usp=sharing)

This needs to be placed in the working directory

## Langsmith integration

Create a Langsmith account and get your API key.

## .env setup

Paste openAI key and Langsmith API key in the .env file

## Running the app

In the working directory - streamlit run main.py
