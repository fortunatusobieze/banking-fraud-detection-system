#### Model comparison

| Model                | Precision (fraud) | Recall (fraud) | F1‑score (fraud) | ROC‑AUC |
|----------------------|------------------:|---------------:|-----------------:|--------:|
| Logistic Regression  | 0.463            | 0.639          | 0.537            | 0.737   |
| Random Forest        | 0.550            | 0.232          | 0.327            | 0.738   |
| XGBoost              | 0.435            | 0.616          | 0.510            | 0.711   |

The logistic regression model provides the best balance between detecting fraudulent transactions and controlling false positives, as reflected in the highest F1‑score for the fraud class and a strong ROC‑AUC. Random Forest achieves slightly higher ROC‑AUC but misses most fraud cases due to very low recall, while XGBoost offers competitive performance but does not surpass the logistic model overall.