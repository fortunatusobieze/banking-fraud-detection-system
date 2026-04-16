#### Model 2: Random Forest

- **Model**: Random Forest classifier with 200 trees and `class_weight='balanced'` in the same preprocessing pipeline.  
- **Fraud-class performance (Is_Fraud = 1)**: precision ≈ 0.55, recall ≈ 0.23, F1‑score ≈ 0.33.  
- **Observation**: Although the Random Forest improves performance on the majority (non‑fraud) class and slightly increases overall accuracy, it performs poorly at detecting fraud cases, missing most of them. This makes it unsuitable as the final model for a fraud detection system where catching fraud is a priority.