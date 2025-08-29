import streamlit as st
import xgboost as xgb 
import numpy as np
import pandas as pd
import pickle

with open("xgb_model (3).pkl","rb") as f:
    model=pickle.load(f)

with open("encoders (4).pkl","rb") as f:
    encoders = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")

person_age = st.number_input("Enter your age", min_value=18, max_value=100, step=1)
person_income = st.number_input("Enter your monthly income", min_value=0, step=1000)
person_home_ownership = st.selectbox("Select your home ownership status", encoders['person_home_ownership'].classes_)
person_emp_length = st.number_input("Enter years of employment", min_value=0, max_value=50, step=1)
loan_intent = st.selectbox("Select the purpose of the loan",encoders['loan_intent'].classes_ )
loan_grade = st.selectbox("Select the loan grade", encoders['loan_grade'].classes_)
loan_amnt = st.number_input("Enter the loan amount", min_value=0, step=500)
loan_percent_income = st.number_input("Enter loan amount as % of income", min_value=0.0, max_value=1.0, step=0.01)
cb_person_default_on_file = st.selectbox("Any previous loan default? (Yes/No)", encoders['cb_person_default_on_file'].classes_)
cb_person_cred_hist_length = st.number_input("Enter credit history length (in years)", min_value=0, max_value=50, step=1)


input_data = {
    "person_age": [person_age],
    "person_income": [person_income],
    "person_home_ownership": [person_home_ownership],
    "person_emp_length": [person_emp_length],
    "loan_intent": [loan_intent],
    "loan_grade": [loan_grade],
    "loan_amnt": [loan_amnt],
    "loan_percent_income": [loan_percent_income],
    "cb_person_default_on_file": [cb_person_default_on_file],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length],
}

df = pd.DataFrame(input_data)

for i in encoders:
    df[i]=df[i].astype(str).fillna("UNKNOWN")
for i in encoders:
    df[i]=encoders[i].transform(df[i]) 

df = df[model.feature_names_in_]


if st.button("Predict"):
    prediction = model.predict(df)

    if prediction[0] == 1:
        st.success("✅Congratulations, Your Loan is Approved")
    else:
        st.error("❌Sorry, Your Loan is Not Approved")
        