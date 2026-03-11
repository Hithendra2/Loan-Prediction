import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_model.pkl")

st.title("Loan Approval Prediction System")

st.write("Enter applicant details below")

# User inputs
income = st.number_input("Applicant Income", min_value=0)

loan_amount = st.number_input("Loan Amount", min_value=0)

credit_history = st.selectbox(
"Credit History",
[1,0]
)

education = st.selectbox(
"Education",
["Graduate","Not Graduate"]
)

married = st.selectbox(
"Married",
["Yes","No"]
)

# Convert inputs
education = 1 if education == "Graduate" else 0
married = 1 if married == "Yes" else 0

loan_income_ratio = loan_amount / income if income != 0 else 0

# Prediction
if st.button("Predict Loan Status"):

    user_data = pd.DataFrame([[
        income,
        loan_amount,
        credit_history,
        education,
        married,
        loan_income_ratio
    ]], columns=[
        "ApplicantIncome",
        "LoanAmount",
        "Credit_History",
        "Education",
        "Married",
        "LoanIncomeRatio"
    ])

    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
