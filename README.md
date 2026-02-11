# 🏦 TrustBank Loan Approval Prediction System

An AI-powered web application that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial details.

Built using Machine Learning and deployed with Streamlit.

---

## 🚀 Live Demo

🔗 [Click Here to Use the App](PASTE_YOUR_STREAMLIT_LINK_HERE)

---

## 📌 Project Overview

This project implements a supervised machine learning pipeline to predict loan approval status using financial and credit-related features.

The system:
- Performs data preprocessing
- Handles categorical encoding
- Applies feature scaling
- Trains multiple classification models
- Selects the best performing model
- Deploys the model using Streamlit

---

## 🧠 Problem Statement

Financial institutions must evaluate loan applications based on multiple risk factors such as income, credit score, and asset value.

This project builds a predictive model to automate the loan approval decision process.

---

## 📊 Dataset Features

- Number of Dependents
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value
- Education Level
- Self Employed Status

Target Variable:
- Loan Status (Approved / Rejected)

---

## ⚙️ Machine Learning Pipeline

1. Data Cleaning
2. Outlier Detection (IQR method – no aggressive removal)
3. Target Encoding (Approved = 1, Rejected = 0)
4. One-Hot Encoding for categorical features
5. Stratified Train-Test Split
6. Feature Scaling (StandardScaler)
7. Model Training

Models Used:
- Logistic Regression
- Random Forest Classifier (Final Model)

---

## 📈 Model Performance

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | ~92%     |
| Random Forest        | ~98%     |

Random Forest was selected as the final model due to superior performance and robustness.

---

## 🖥️ Deployment

The trained model was deployed using **Streamlit**.

Saved components:
- loan_model.pkl
- scaler.pkl
- feature_columns.pkl

The web application:
- Accepts user input
- Applies preprocessing
- Predicts loan approval
- Displays probability scores
- Shows feature importance chart

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## 📂 Project Structure

├── app.py
├── loan_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md

---

## 🎯 Key Learnings

- Importance of correct preprocessing order
- Avoiding data leakage
- Handling class imbalance using stratified sampling
- Model comparison and evaluation
- Deploying ML models with Streamlit

---

## 👨‍💻 Author
~YASH MORE
Developed as a Machine Learning internship project.

---

[Click Here to Use the App]([PASTE_YOUR_STREAMLIT_LINK_HERE](https://loan-approval-dashboard-pjspzknbte7fvgjjwqd9jq.streamlit.app/))



