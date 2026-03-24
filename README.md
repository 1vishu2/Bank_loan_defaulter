# 🏦 Bank Loan Defaulter Prediction

🔗 **Live App:** https://bankloandefaulter-x3vnvhyfdzktv4ns3z5w7x.streamlit.app/

## 📌 Overview
This project predicts whether a customer is likely to **default on a loan** using Machine Learning 

## 🚀 Features
- Real-time prediction using Streamlit  
- Trained ML model (CatBoost)  
- Simple and interactive UI  

## 📂 Project Structure
Bank_loan_defaulter/
│
├── app/
│ ├── credit_risk.py
│ └── catboost_pipeline.pkl
|
├── data/
│ ├── credit_risk_cleaned.csv
│ └── credit_risk_raw.csv
|
├── models/
│ ├── catboost_pipeline.pkl
|
├── notebooks/
│ ├── bank_loan_eda.ipynb
| ├── bank_model_gitready.ipynb
|
├── requirements.txt
└── README.md

## 📊 Key Insights (Risk Factors)

Factors that **increase default risk**:
- High **loan_percent_income** (loan burden)
- Previous **loan defaults**
- Lower **income levels**
- Poor **loan grade**
- Short **employment length**

Factors that **reduce default risk**:
- Stable income
- Good loan grade (A/B)
- No past defaults
- Balanced loan-to-income ratio



