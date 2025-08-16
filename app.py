import streamlit as st
import pandas as pd
import pickle
# Load trained models
knn_model = pickle.load(open("knn_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))
rfc_model = pickle.load(open("rfc_model.pkl", "rb"))

st.title("🏦 Loan Approval Prediction")

# Sidebar for model selection
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Choose a Model", ["KNN", "SVM", "Random Forest"])

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.selectbox("Loan Amount Term (in months)",[360, 180, 120, 84, 60, 36, 12])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode categorical values
def encode_input():
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)
    education_val = 0 if education == "Graduate" else 1
    self_emp_val = 1 if self_employed == "Yes" else 0
    property_val = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    return [[gender_val, married_val, dependents_val, education_val, self_emp_val,
             applicant_income, coapplicant_income, loan_amount, loan_amount_term,
             credit_history, property_val]]

# Predict
if st.button("Check Loan Approval"):
    user_data = encode_input()
    if model_choice == "KNN":
        prediction = knn_model.predict(user_data)
    elif model_choice == "SVM":
        prediction = svm_model.predict(user_data)
    else:
        prediction = rfc_model.predict(user_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
