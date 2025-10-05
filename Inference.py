"""
Inference script for the trained XGBoost model.
Make sure you have the trained model saved as:
    models/xgboost_model.joblib
and the new data CSV ready for testing (e.g., "test_data.csv").
"""

import pandas as pd
import joblib
import os

# === 1. Load the trained model ===
model_path = "models/xgboost_model.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")
model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# === 2. Load and preprocess the new data ===
# Replace 'test_data.csv' with your test file
test_data = pd.read_csv("test_data.csv")

# Apply the same preprocessing as during training
# (Drop CLASS, shift labels if needed)
if "CLASS" in test_data.columns:
    test_data = test_data.drop(columns=["CLASS"])
if "Y" in test_data.columns:
    test_data["Y"] = test_data["Y"] - 1  # only if applicable

# === 3. Make predictions ===
pred_probs = model.predict_proba(test_data)
pred_classes = model.predict(test_data)

# === 4. Combine and save results ===
output = test_data.copy()
output["Predicted_Class"] = pred_classes

# (Optional) add predicted probabilities for each class
for i in range(pred_probs.shape[1]):
    output[f"Prob_Class_{i}"] = pred_probs[:, i]

# Save predictions
os.makedirs("outputs", exist_ok=True)
output.to_csv("outputs/inference_results.csv", index=False)

print("✅ Inference complete!")
print("Predictions saved to outputs/inference_results.csv")

