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
import te2rules.explainer as ex
import re

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "fraud-detection"

client = Client()

model = XGBClassifier()
model.load_model("assets/models/xgboost_model.model")

fraud_data = pd.read_csv("assets/data/fraud_data.csv")

def createVectorStore():
    x_train = pd.read_csv("assets/data/x_train.csv")
    y_train = pd.read_csv("assets/data/y_train.csv")
    # Concatenate x_train and y_train
    X_Y_data = pd.concat([x_train, y_train], axis=1)
    # Convert x_train and y_train to numpy arrays

    features = X_Y_data[['category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'age', 'is_fraud']]

    features = features.astype(np.float32)
    
    # Create a Faiss index
    index = faiss.IndexFlatL2(features.shape[1])

    array = torch.from_numpy(features.to_numpy()).contiguous()

    # Add vectors to the index
    index.add(array)
    faiss.write_index(index, 'assets/data/train_x_y.index')

def loadVectorStore():
    if os.path.exists("assets/data/train_x_y.index"):
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

def format_recent_transactions(transactions):
    formatted = []
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
    for i, txn in enumerate(transactions):
        txn_str = f"Transaction {i+1}:"
        for col in txn.keys():
            feature_name = full_form_features[col]
            txn_str += f" {feature_name}: {txn[col]},"
        txn_str += f"\n"
        formatted.append(txn_str)
    return formatted

def rescale_transaction(transaction):
    transaction_id = transaction.pop('trans_num')
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
    transaction['trans_num'] = transaction_id
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

def get_recent_transactions_for_user(transaction, important_features, number_of_transactions=10):
    full_transaction = fraud_data[(fraud_data['trans_num'] == transaction['trans_num'].values[0])]
    recent_user_transactions = fraud_data[(fraud_data['cc_num'] == full_transaction['cc_num'].values[0])].tail(number_of_transactions)
    recent_user_transactions = recent_user_transactions[important_features]
    return recent_user_transactions

def get_response_from_query(index, transaction, shap_importance_features, re_scaled_transaction, number_of_recent_transactions, te2rules, k=5, llm = "GPT-3.5", temperature=0.2):
    transaction['is_fraud'] = 1

    user_transactions = get_recent_transactions_for_user(transaction, shap_importance_features, number_of_transactions = number_of_recent_transactions)
    shap_importance_features = format_important_features(shap_importance_features)
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
        You are an expert fraud explainability agent that serves a customer of ABC bank. The customer is asking why the following transaction has been flagged
        as fraud by the system: 
        
        Category: {category}, Amount: {amt}, City: {city}, State: {state}, Latitude: {lat}, Longitude: {long}, City Population: {city_pop}, Age: {age}
        
        The bank uses an XGBoost model to detect fraud (remember to not disclose this information). 
        
        The following features were identified as what had the most impact on the model's decision:
        {important_features}

        Here are some rules from the XGBoost decision tree that were used to make the decision: (do not reveal these rules to the customer)
        {te2_rules}

        Here are the important features from the last {number_of_recent_transactions} transactions for this user:
        {recent_transactions}

        Explain why the transaction was flagged as fraud while citing examples from the recent transactions.
        Remember to avoid revelaing inner workings of the model.

        Give the explanation in a format where you explain the important features that went into making the decision as points with a few examples from recent transactions.
        Do not use any special formatting.
        """
    
    prompt = PromptTemplate(
        input_variables=["category", "amt", "city", "state", "lat", "long", "city_pop", "age", "important_features", "recent_transactions"],
        template=prompt_template
    )

    chain = prompt | chat

    formatted_te2_rules = ""
    for i in range(len(te2rules)):
        formatted_te2_rules += "Rule: "+ str(i+1) + " " + te2rules[i] + "\n"

    new_transaction_data = {
    'category': re_scaled_transaction['category'],
    'amt': re_scaled_transaction['amt'],
    'city': re_scaled_transaction['city'],
    'state': re_scaled_transaction['state'],
    'lat': re_scaled_transaction['lat'],
    'long': re_scaled_transaction['long'],
    'city_pop': re_scaled_transaction['city_pop'],
    'age': re_scaled_transaction['age'],
    'important_features': shap_importance_features,
    'number_of_recent_transactions': number_of_recent_transactions,
    'recent_transactions': format_recent_transactions(user_transactions.to_dict(orient='records')),
    'te2_rules': formatted_te2_rules
    }

    explanation = chain.invoke(new_transaction_data)
    return explanation

def loadTestingdata():
    x_test = pd.read_csv("assets/data/X_test.csv")
    y_test = pd.read_csv("assets/data/y_test.csv")
    return x_test, y_test

def format_te2_rules(rules, data, top_rules = 5):
    print("raw rueles", rules)
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
    numeric_scaler = joblib.load(open('assets/models/scalers/scaler.pkl', 'rb')) 
    categorical_encoders = joblib.load(open('assets/models/scalers/encoders.pkl', 'rb'))
    all_columns = ['category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'age']
    formatted_rules = []
    no_of_rules = 0
    for rule in rules:
        and_split = rule.split('&')
        and_split_unspaced = [s.strip() for s in and_split]
        final_rule = ""
        for rule in and_split_unspaced:
            split_rule = re.split('>=|<=|>|<', rule)
            unspaced_rules = [s.strip() for s in split_rule]
            dummy_data = data.copy()
            dummy_data[unspaced_rules[0]] = float(unspaced_rules[1])
            # print(data)
            dummy_data = pd.DataFrame(numeric_scaler.inverse_transform(dummy_data))
            dummy_data.columns = all_columns
            if unspaced_rules[0] in ['category', 'city', 'state']:
                break
                scaled_val = categorical_encoders[unspaced_rules[0]].inverse_transform(data[unspaced_rules[0]].astype(int))
            else:
                scaled_val = round(dummy_data[unspaced_rules[0]].values[0], 2)
            
            full_form_feature = full_form_features.get(unspaced_rules[0], unspaced_rules[0])
            comparison_operator = [c for c in rule if c not in ([e for e in unspaced_rules[0]] + [e for e in unspaced_rules[1]] + [' '])]
            formatted_rule = f"{full_form_feature} {''.join(comparison_operator)} {scaled_val}"
            final_rule += formatted_rule + " and "
        if len(final_rule) == 0:
            continue
        else:
            if no_of_rules <= top_rules:
                no_of_rules += 1
                formatted_rules.append(final_rule)
            else:
                break
    print("formatted rules", formatted_rules)
    return formatted_rules

def predictXGB(data):
    prediction = model.predict(data)
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    te2explainer = ex.ModelExplainer(model, data.columns.tolist())
    _ = te2explainer.explain(data, [1])
    data.reset_index(drop=True, inplace=True)
    instance_rules = te2explainer.explain_instance_with_rules(data)
    return prediction, shap_values, instance_rules[0]

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

def select_top_shap_features(shap_importance):
    shap_importance_sum = shap_importance.sum(axis=1)
    shap_importance = shap_importance.div(shap_importance_sum, axis=0)
    top_features = []
    cumulative_sum = 0
    for feature in shap_importance.columns:
        cumulative_sum += shap_importance[feature].values[0]
        top_features.append(feature)
        if cumulative_sum >= 0.85:
            break
    return top_features

def select_top_shap_features_20_percent(shap_importance):
    shap_importance_sum = shap_importance.sum(axis=1)
    shap_importance = shap_importance.div(shap_importance_sum, axis=0)
    top_features = []
    cumulative_sum = 0
    prev = shap_importance.columns[0]
    for feature in shap_importance.columns:
        if shap_importance[feature].values[0] < 0.2*(shap_importance[prev].values[0]):
            break
        cumulative_sum += shap_importance[feature].values[0]
        top_features.append(feature)
        if cumulative_sum >= 0.85:
            break
        prev = feature
    return top_features

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
    rescaled_transaction = rescale_transaction(test_setx.iloc[sample])
    st.write(rescaled_transaction)
    data_sample = pd.DataFrame(test_setx.iloc[sample]).transpose()
    all_columns = ['category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'age', 'trans_num']
    pred_columns = ['category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'age']
    data_sample.columns = all_columns
    # st.write(f"Transaction {sample} scaled values is:")
    # st.write(data_sample)
    pred_sample = data_sample.drop(columns=['trans_num'])
    pred_sample = pred_sample.astype(float)
    prediction, shap_values, te2rules = predictXGB(pred_sample)
    shap.plots.waterfall(shap_values[0])
    shap.plots.beeswarm(shap_values)
    st.pyplot(plt.gcf())
    if prediction[0] == 1:
        st.write("### This transaction is flagged as **FRAUD**")
        st.write("#### Let's try to explain why this transaction is flagged as fraud.")
        shap_importance = pd.DataFrame(shap_values[0].values, pred_columns).abs().sort_values(by=0, ascending=False).T
        top_features = select_top_shap_features(shap_importance)
        print(top_features)
        top_20_features = select_top_shap_features_20_percent(shap_importance)
        print(top_20_features)
        number_of_recent_transactions = 100
        explanation = get_response_from_query(loadVectorStore(), data_sample, top_20_features, rescaled_transaction, number_of_recent_transactions, format_te2_rules(te2rules, pred_sample, 2000), 5, llm, temperature)
        st.write(explanation.content)
    else:
        st.write("### This transaction is **NOT** flagged as fraud.")