import streamlit as st
from dotenv import find_dotenv, load_dotenv
import os
import nest_asyncio
import pandas as pd
from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain import PromptTemplate, LLMChain
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import to_graphviz
import faiss
import numpy as np
import faiss
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import faiss.contrib.torch_utils
import torch
import matplotlib.pyplot as plt
import shap

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "fraud-detection"

client = Client()

model = XGBClassifier()
model.load_model("assets/models/xgboost_model.model")

def createVectorStore():
    x_train = pd.read_csv("assets/data/x_train.csv")
    y_train = pd.read_csv("assets/data/y_train.csv")
    # Concatenate x_train and y_train
    X_Y_data = pd.concat([x_train, y_train], axis=1)
    # Convert x_train and y_train to numpy arrays

    features = X_Y_data[['category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'age', 'is_fraud']]

    features = features.astype(np.float32)
    print(features.shape[1])
    
    # Create a Faiss index
    index = faiss.IndexFlatL2(features.shape[1])

    array = torch.from_numpy(features.to_numpy()).contiguous()

    # Add vectors to the index
    index.add(array)
    faiss.write_index(index, 'assets/data/train_x_y.index')

def loadVectorStore():
    if os.path.exists("assets/data/train_x_y.idx"):
        index = faiss.read_index("assets/data/train_x_y.index")
    else:
        createVectorStore()
        index = faiss.read_index("assets/data/train_x_y.index")
    return index

def format_transactions(transactions):
    formatted = []
    for i, txn in enumerate(transactions):
        txn_str = f"Transaction {i+1}: Cateogry: {txn['category']}, Amount: {txn['amt']}, City: {txn['city']}, State: {txn['state']}, Latitude: {txn['lat']}, Longitude: {txn['long']}, City Population: {txn['city_pop']}, Age: {txn['age']}, Is Fraud: {txn['is_fraud']}"
        formatted.append(txn_str)
    return "\n".join(formatted)

def rescale_transaction(transaction):
    numeric_columns = ['amt', 'lat', 'long', 'city_pop', 'age']
    categorical_columns = ['category', 'city', 'state']
    all_columns = ['category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'age']

    numeric_scaler = joblib.load(open('assets/models/scalers/scaler.pkl', 'rb')) 
    categorical_encoders = joblib.load(open('assets/models/scalers/encoders.pkl', 'rb'))

    transaction = pd.DataFrame(transaction).transpose()
    transaction.columns = all_columns
    transaction = pd.DataFrame(numeric_scaler.inverse_transform(transaction))
    transaction.columns = all_columns
    for col in categorical_columns:
        transaction[col] = categorical_encoders[col].inverse_transform(transaction[col].astype(int))
    return transaction

def format_important_features(sorted_features):
    full_form_features = {
        'category': 'Category',
        'amt': 'Amount',
        'city': 'City',
        'state': 'State',
        'lat': 'Latitude',
        'long': 'Longitude',
        'city_pop': 'City Population',
        'age': 'Age',
    }
    formatted_features = []
    for feature in sorted_features:
        feature_name = full_form_features.get(feature, feature)
        formatted_features.append(feature_name)
    return formatted_features

def get_response_from_query(index, transaction, shap_importance_features, k=5, llm = "GPT-3.5", temperature=0.2):
    # loading similar transactions to given transaction
    copy_transaction = transaction.copy()
    dataset = pd.read_csv("assets/data/training_data.csv")
    transaction['is_fraud'] = 1

    vector = torch.from_numpy(transaction.astype(np.float32).to_numpy()).contiguous()

    _, indices = index.search(vector, k=k)
    similar_transactions = dataset.iloc[indices[0]].to_dict(orient='records')

    formatted_transactions = format_transactions(similar_transactions)

    # loading language model
    if llm == "GPT-3.5":
        chat = ChatOpenAI(model_name="gpt-3.5", temperature = temperature)
    elif llm == "GPT-3.5 Turbo":
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = temperature)
    elif llm == "GPT-4":
        chat = ChatOpenAI(model_name="gpt-4", temperature= temperature)
    elif llm == "GPT-4 Turbo":
        chat = ChatOpenAI(model_name="gpt-4-turbo", temperature= temperature)
    
    prompt_template = """
        You are a helpful assistant that serves a customer of a large bank. The customer is asking why the following transaction has been flagged
        as fraud by the system: 
        
        Category: {category}, Amount: {amt}, City: {city}, State: {state}, Latitude: {lat}, Longitude: {long}, City Population: {city_pop}, Age: {age}
        
        The bank uses an XGBoost model to detect fraud. Running the prediction through SHAP, the most important features that influenced the model's decision are:
        {important_features}

        Use this info to explain why the transaction was flagged as fraud while correlating the important features with the transaction details.
        """
    
    prompt = PromptTemplate(
        input_variables=["category", "amt", "city", "state", "lat", "long", "city_pop", "age", "important_features"],
        template=prompt_template
    )

    chain = prompt | chat

    re_scaled_transaction = rescale_transaction(copy_transaction.transpose())
    re_scaled_transaction['is_fraud'] = 1

    new_transaction_data = {
    'category': re_scaled_transaction['category'],
    'amt': re_scaled_transaction['amt'],
    'city': re_scaled_transaction['city'],
    'state': re_scaled_transaction['state'],
    'lat': re_scaled_transaction['lat'],
    'long': re_scaled_transaction['long'],
    'city_pop': re_scaled_transaction['city_pop'],
    'age': re_scaled_transaction['age'],
    # 'is_fraud': re_scaled_transaction['is_fraud'],
    # 'similar_transactions': formatted_transactions
    'important_features': shap_importance_features
    }

    explanation = chain.invoke(new_transaction_data)
    print(explanation)
    return explanation

def loadTestingdata():
    x_test = pd.read_csv("assets/data/x_test.csv")
    x_test = x_test.iloc[:, 1:]
    y_test = pd.read_csv("assets/data/y_test.csv")
    return x_test, y_test

def predictXGB(data):
    # all_columns = ['category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'age']
    prediction = model.predict(data)
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    return prediction, shap_values

def plot_tree_wrapper(xgb_model, filename, rankdir='UT'):
    """
    Plot the tree in high resolution
    :param xgb_model: xgboost trained model
    :param filename: the pdf file where this is saved
    :param rankdir: direction of the tree: default Top-Down (UT), accepts:'LR' for left-to-right tree
    :return:
    """
    gvz = to_graphviz(xgb_model, num_trees = 0, rankdir=rankdir)
    _, file_extension = os.path.splitext(filename)
    format = file_extension.strip('.').lower()
    data = gvz.pipe(format=format)
    full_filename = filename

    with open(full_filename, 'wb') as f:
        f.write(data)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Explainable Fraud Detection",
    )
    st.write("# Explainable Fraud Detection")
    st.write("#### An XGBoost model classifies fraudulent transactions. If a transaction is flagged as fraud, an AI assistant will explain why.")
    test_setx, test_sety = loadTestingdata()
    st.write(f"Currently there are **{test_sety.shape[0]}** transactions in the test dataset. Which sample would you like to test?")
    sample = st.slider("Select a sample", 0, test_sety.shape[0])
    llm = st.radio("Select a language model", ["GPT-3.5", "GPT-3.5 Turbo", "GPT-4", "GPT-4 Turbo"])
    temperature = st.slider("Select temperature", 0.1, 1.5)
    st.write(f"Transaction {sample} is:")
    # st.write(test_setx.iloc[sample])
    st.write(rescale_transaction(test_setx.iloc[sample]))
    data_sample = pd.DataFrame(test_setx.iloc[sample]).transpose()
    all_columns = ['category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'age']
    data_sample.columns = all_columns
    st.write(f"Transaction {sample} scaled values is:")
    st.write(data_sample)
    # plot_tree_wrapper(model, 'xgb_tree.pdf', rankdir='UT')
    # st.pyplot(plt.gcf())
    prediction, shap_values = predictXGB(data_sample)
    shap.plots.waterfall(shap_values[0])
    shap.plots.beeswarm(shap_values)
    st.pyplot(plt.gcf())
    if prediction[0] == 1:
        st.write("### This transaction is flagged as **FRAUD**")
        st.write("#### Let's try to explain why this transaction is flagged as fraud.")
        shap_importance = pd.DataFrame(shap_values[0].values, all_columns).abs().sort_values(by=0, ascending=False).T
        st.write(shap_importance.columns.tolist())
        explanation = get_response_from_query(loadVectorStore(), data_sample, format_important_features(shap_importance), 5, llm, temperature)
        st.write(explanation.content)
    else:
        st.write("### This transaction is **NOT** flagged as fraud.")

# Fraud transactions: 2479, 2238, 2457