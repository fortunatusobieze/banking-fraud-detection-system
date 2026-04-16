#### Baseline model: Logistic Regression

- **Model**: Logistic Regression with `class_weight='balanced'` to account for the higher number of non‑fraud cases.  
- **Preprocessing**: Numeric features were standardised and categorical features were one‑hot encoded within a single sklearn `Pipeline`.  
- **Fraud-class performance (Is_Fraud = 1)**: precision ≈ 0.463, recall ≈ 0.639, F1‑score ≈ 0.537 on the held‑out test set.  
- **Observation**: The model is able to identify a reasonable proportion of fraud cases, but it still misses some frauds and raises a fair number of false alarms, so there is clear room for improvement with more expressive models and further tuning.