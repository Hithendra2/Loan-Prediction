import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("loan_data.csv")

# Feature Engineering
data["LoanIncomeRatio"] = data["LoanAmount"] / data["ApplicantIncome"]
data["LoanIncomeRatio"].replace([np.inf, -np.inf], np.nan, inplace=True)
data["LoanIncomeRatio"].fillna(0, inplace=True)

# Encode categorical variables
data["Education"] = data["Education"].map({"Graduate":1, "Not Graduate":0})
data["Married"] = data["Married"].map({"Yes":1, "No":0})
data["Loan_Status"] = data["Loan_Status"].map({"Y":1, "N":0})

# Features
X = data[
[
"ApplicantIncome",
"LoanAmount",
"Credit_History",
"Education",
"Married",
"LoanIncomeRatio"
]]

# Target
y = data["Loan_Status"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")

print("Model trained and saved successfully!")
