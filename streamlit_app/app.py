import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Banking Fraud Detection App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join("..", "models", "final_logreg_pipeline.joblib")
model = joblib.load(MODEL_PATH)

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

h1, h2, h3 {
    color: #0f172a;
    font-family: "Segoe UI", sans-serif;
}

.hero-box {
    background: linear-gradient(135deg, #0f172a, #1d4ed8);
    padding: 1.5rem 2rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
}

.metric-card {
    background: white;
    padding: 1.2rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-top: 1rem;
    margin-bottom: 1rem;
    border: 1px solid #e2e8f0;
}

.info-card {
    background: #ffffff;
    padding: 1.1rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

.small-note {
    font-size: 0.92rem;
    color: #475569;
}

.result-fraud {
    color: #b91c1c;
    font-weight: 700;
}

.result-safe {
    color: #166534;
    font-weight: 700;
}

div.stButton > button {
    background-color: #1d4ed8;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1e40af;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <h1>🛡️ Banking Fraud Detection System</h1>
    <p style="font-size:16px; margin-bottom:0;">
        This application predicts the likelihood that a banking transaction is fraudulent
        using a trained machine learning pipeline. It is designed as a decision-support
        tool for fraud analysis and educational demonstration.
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Prediction mode")
mode = st.sidebar.radio(
    "Choose an option",
    ["Single transaction", "Batch (CSV upload)"]
)

def get_single_input():
    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        ["Online", "POS", "ATM", "Transfer"]
    )

    transaction_location = st.text_input(
        "Transaction Location",
        value="London"
    )

    transaction_time = st.text_input(
        "Transaction Time (e.g., 2025-01-15 13:45:00)",
        value="2025-01-15 13:45:00"
    )

    device_used = st.selectbox(
        "Device Used",
        ["Mobile", "Web", "ATM", "POS"]
    )

    account_age = st.number_input(
        "Account Age (years)",
        min_value=0,
        value=3,
        step=1
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=10
    )

    previous_fraud = st.number_input(
        "Previous Fraud Count",
        min_value=0,
        value=0,
        step=1
    )

    input_data = {
        "Transaction_Amount": transaction_amount,
        "Transaction_Type": transaction_type,
        "Transaction_Location": transaction_location,
        "Transaction_Time": transaction_time,
        "Device_Used": device_used,
        "Account_Age": account_age,
        "Credit_Score": credit_score,
        "Previous_Fraud": previous_fraud,
    }

    return pd.DataFrame([input_data])

if mode == "Single transaction":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Single transaction prediction")
        input_df = get_single_input()
        predict_btn = st.button("Predict Transaction")

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>How to use</h3>
            <p class="small-note">
                Enter the transaction details and click <strong>Predict Transaction</strong>.
                The model will return a predicted class and a fraud probability score.
            </p>
        </div>
        <div class="info-card">
            <h3>Interpretation</h3>
            <p class="small-note">
                A higher probability indicates a higher likelihood of fraud.
                This output should support human review rather than replace it.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        proba = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        label = "Fraud" if pred == 1 else "Non-Fraud"
        label_class = "result-fraud" if pred == 1 else "result-safe"

        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-bottom:0.5rem;">Prediction Result</h3>
            <p><strong>Predicted Class:</strong> <span class="{label_class}">{label}</span></p>
            <p><strong>Fraud Probability:</strong> {proba:.3f}</p>
            <p class="small-note">1 = Fraud, 0 = Non-Fraud</p>
        </div>
        """, unsafe_allow_html=True)

elif mode == "Batch (CSV upload)":
    st.subheader("Batch prediction from CSV")

    st.markdown("""
    <div class="info-card">
        <p class="small-note">
            Upload a CSV containing the same input fields used during model training:
            <strong>Transaction_Amount, Transaction_Type, Transaction_Location,
            Transaction_Time, Device_Used, Account_Age, Credit_Score, Previous_Fraud</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if st.button("Run batch prediction"):
            probs = model.predict_proba(batch_df)[:, 1]
            preds = model.predict(batch_df)

            results = batch_df.copy()
            results["Fraud_Probability"] = probs
            results["Predicted_Is_Fraud"] = preds

            st.subheader("Prediction results")
            st.dataframe(results.head(10))

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

st.markdown("""
<hr>
<p class="small-note">
This tool is for demonstration and educational purposes only.
It should not be used as a sole decision-making system in real banking environments.
</p>
""", unsafe_allow_html=True)