import streamlit as st
import pandas as pd
import pickle 

class CatBoostWithThreshold:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

### use credit risk env
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open(r"catboost_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()




# Input fields

st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

st.divider()

# ----------------------------------
# Applicant Details
# ----------------------------------
st.subheader("👤 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    person_age = st.slider("Age", 18, 100, 30)
    person_emp_length = st.slider(
        "Employment Length (years)", 0.0, 50.0, 5.0, step=0.5
    )
    person_home_ownership = st.selectbox(
        "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

with col2:
    person_income = st.slider(
        "Annual Income", 10000, 2000000, 500000, step=10000
    )
    cred_hist_length = st.slider(
        "Credit History Length (years)", 0.0, 50.0, 10.0, step=0.5
    )
    past_default_status = st.selectbox(
        "Past Default", ["N", "Y"]
    )

st.divider()

# ----------------------------------
# Loan Details
# ----------------------------------
st.subheader("💰 Loan Details")

col3, col4 = st.columns(2)

with col3:
    loan_amnt = st.slider(
        "Loan Amount", 1000, 1000000, 100000, step=5000
    )
    loan_int_rate = st.slider(
        "Interest Rate (%)", 1.0, 40.0, 12.5, step=0.1
    )
    loan_grade = st.selectbox(
        "Loan Grade", ["A", "B", "C", "D", "E", "F", "G"]
    )

with col4:
    loan_percent_income = st.slider(
        "Loan % of Income", 0.0, 1.0, 0.2, step=0.01
    )
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "PERSONAL",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        ],
    )

st.divider()

# ----------------------------------
# Create input dataframe
# ----------------------------------
input_data = pd.DataFrame({
    "person_age": [person_age],
    "person_income": [person_income],
    "person_home_ownership": [person_home_ownership],
    "person_emp_length": [person_emp_length],
    "loan_intent": [loan_intent],
    "loan_grade": [loan_grade],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "past_default_status": [past_default_status],
    "cred_hist_length": [cred_hist_length]
})


prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]


# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔍 Predict Default Risk", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default\n\nProbability: **{probability:.2%}**")
    else:
        st.success(f"✅ Low Risk of Default\n\nProbability: **{probability:.2%}**")
    

