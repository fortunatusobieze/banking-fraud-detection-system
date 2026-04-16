#### Model 3: XGBoost

- **Model**: XGBoost classifier wrapped in the same preprocessing pipeline, with `scale_pos_weight` set to the class ratio to give more emphasis to fraud cases.  
- **Fraud-class performance (Is_Fraud = 1)**: precision ≈ 0.435, recall ≈ 0.616, F1‑score ≈ 0.510 on the test set.  
- **Observation**: XGBoost provides a balanced trade‑off between detecting fraudulent transactions and controlling false positives. It clearly outperforms the Random Forest in terms of fraud recall and offers similar performance to Logistic Regression, while being more flexible and expressive. This makes it a strong candidate for the final deployed model.