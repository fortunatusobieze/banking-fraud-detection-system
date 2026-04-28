# Banking Fraud Detection System 🛡️

End‑to‑end machine learning system for predicting fraudulent banking transactions.
The project includes a full ML pipeline (data preprocessing, model training, evaluation) and an interactive **Streamlit** web application for single and batch transaction scoring.

> This repository is for educational and demonstration purposes only. It must not be used as a production fraud detection tool in real banking environments.

---

## 🚀 Live Demo

Deployed on **Streamlit Community Cloud**:

> `https://banking-fraud-detection-system.streamlit.app/`

The app provides:
- **Single transaction prediction** with an interactive form.
- **Batch CSV upload** for scoring multiple transactions at once.
- A clear display of **predicted class** (Fraud / Non‑Fraud) and **fraud probability**.

---

## 📊 Project Overview

The goal of this project is to design, evaluate, and deploy a machine learning system that predicts whether a banking transaction is likely to be fraudulent.

Key elements:

- Uses a **2025 synthetic banking transactions dataset** for fraud detection.
- Frames the problem as a **binary classification** task: legitimate (0) vs fraudulent (1).
- Trains several models (Logistic Regression, Random Forest, XGBoost) and selects a final model based on fraud‑class performance.
- Wraps the final model in a **Streamlit app** for accessible, interactive use.

---

## 🧠 Machine Learning Pipeline

The ML workflow is implemented in a Jupyter notebook (see the `notebooks/` folder) and includes:

1. **Data exploration**
   - Dataset inspection, class imbalance analysis.
   - Basic visualisations (class distribution, feature distributions).

2. **Preprocessing and feature handling**
   - Train/test split with stratification on the fraud label.
   - Separate handling of numerical and categorical features using `ColumnTransformer`.
   - Encoding of categorical features; scaling/transforming numerical features where appropriate.

3. **Models implemented**
   - **Logistic Regression** (baseline and final selected model).
   - **Random Forest** classifier.
   - **XGBoost** classifier.

4. **Evaluation**
   - Focus on fraud‑class metrics: precision, recall, F1‑score.
   - ROC‑AUC and confusion matrix analysis.
   - Model comparison table summarising performance across models.

5. **Model selection**
   - Final choice: **Logistic Regression** pipeline with class‑imbalance handling, based on best fraud‑class F1 and competitive ROC‑AUC.
   - Saved as a reusable pipeline with `joblib` and loaded by the Streamlit app.

---

## 🌐 Streamlit Application

The Streamlit app (`streamlit_app/app.py`) exposes the trained model via a simple web interface:

### Features

- **Single Transaction Mode**
  - Form fields for transaction amount, type, location, time, device, account age, credit score, and previous fraud count.
  - On submit, the app displays:
    - Predicted class: **Fraud** or **Non‑Fraud**.
    - Fraud probability (between 0 and 1).
  - Result is shown in a styled “metric card” with explanatory text.

- **Batch CSV Mode**
  - Upload a CSV file with the same feature columns as used in training.
  - The app returns:
    - Fraud probability for each transaction.
    - Predicted fraud label.
  - Results can be downloaded as a new CSV file.

- **Design**
  - Clean, dashboard‑style layout with hero banner and explanatory side cards.
  - Simple colour scheme suitable for analytics / banking use cases.

---

## 📁 Repository Structure

```text
.
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── data/                    # (Optional) Sample data / raw data
├── models/
│   └── final_logreg_pipeline.joblib   # Saved Logistic Regression pipeline
├── notebooks/
│   └── fraud_detection_notebook.ipynb # Full ML workflow and experiments
├── reports/                 # Figures, evaluation outputs (e.g., ROC curves)
├── src/                     # (Optional) Reusable source code modules
├── streamlit_app/
│   └── app.py               # Streamlit web application
├── requirements.txt         # Minimal dependencies for deployment
└── README.md
```

> The exact structure may vary slightly from this diagram, but the core components remain the same.

---

## 🛠️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/fortunatusobieze/banking-fraud-detection-system.git
cd banking-fraud-detection-system
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app locally

```bash
cd streamlit_app
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`) in your browser.

---

## 📈 Usage

### Single transaction prediction

1. Choose **“Single transaction”** in the sidebar.
2. Fill in the transaction details in the form.
3. Click **“Predict Transaction”**.
4. Review the predicted class and fraud probability.

### Batch prediction with CSV

1. Choose **“Batch (CSV upload)”** in the sidebar.
2. Upload a CSV file containing the expected columns.
3. Click **“Run batch prediction”**.
4. View the preview of results and download the full predictions as CSV.

---

## 📚 Model and Evaluation Summary

High‑level notes (see notebook for full details):

- **Final model:** Logistic Regression pipeline (with preprocessing and class‑imbalance handling).
- **Target:** Fraud label (0 = non‑fraud, 1 = fraud).
- **Key metrics (fraud class):**
  - Precision, recall, F1 and ROC‑AUC reported on held‑out test data.
- **Baseline comparisons:**
  - Random Forest and XGBoost models were trained and evaluated.
  - Random Forest showed poor fraud recall.
  - XGBoost was competitive, but Logistic Regression offered the best balance of recall, F1, and simplicity.

---

## ⚠️ Disclaimer

- The dataset used is **synthetic** and designed for educational experiments, not for production deployment.
- The model has not been validated on real banking customer data.
- The system is intended as a **teaching and portfolio project**, not as a live fraud prevention solution.

---

## 🙋‍♂️ Acknowledgements

- Module: **Advanced Machine Learning**, Wrexham University, Wales, United Kingdom.
- Streamlit Community Cloud for free hosting of the demo app.
- Open‑source Python libraries: Streamlit, scikit‑learn, pandas, XGBoost, joblib.