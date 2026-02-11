import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="TrustBank Loan System",
    page_icon="🏦",
    layout="wide"
)

# =====================================
# Aesthetic Light Gradient Styling
# =====================================
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #f8fbff, #e6f0ff);
}

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    background: linear-gradient(to right, #0066ff, #00c6ff);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.stNumberInput, .stSelectbox {
    background-color: white;
}

h1 {
    color: #003366;
    text-align: center;
}

h2, h3 {
    color: #004080;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# Header
# =====================================
st.markdown("<h1>🏦 TrustBank Financial Services</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Smart AI Loan Approval Dashboard</h3>", unsafe_allow_html=True)
st.write("---")

# =====================================
# Load Model
# =====================================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =====================================
# Input Section
# =====================================
st.header("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Dependents", 0, 10, 0)
    income_annum = st.number_input("Annual Income (₹)", min_value=0)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
    loan_term = st.number_input("Loan Term (Months)", min_value=1)
    cibil_score = st.number_input("CIBIL Score", 300, 900, 650)

with col2:
    residential_assets_value = st.number_input("Residential Assets (₹)", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets (₹)", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets (₹)", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

st.write("---")

# =====================================
# Prediction
# =====================================
if st.button("🔍 Evaluate Loan Application"):

    input_data = {
        "no_of_dependents": no_of_dependents,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "education": education,
        "self_employed": self_employed
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    approved_prob = probability[1] * 100
    rejected_prob = probability[0] * 100

    st.subheader("📊 Decision Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.progress(int(approved_prob))
        st.write(f"Approval Probability: {approved_prob:.2f}%")
        st.write(f"Rejection Probability: {rejected_prob:.2f}%")
    else:
        st.error("❌ Loan Rejected")
        st.progress(int(rejected_prob))
        st.write(f"Rejection Probability: {rejected_prob:.2f}%")
        st.write(f"Approval Probability: {approved_prob:.2f}%")

    st.write("---")

    # =====================================
    # Feature Importance
    # =====================================
    st.subheader("📈 Key Risk Factors")

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(8)

    st.bar_chart(
        importance_df.set_index("Feature")
    )

st.write("---")
st.caption("© 2026 TrustBank | Intelligent Loan Risk Assessment System")
