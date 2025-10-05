import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

def preprocess_data_for_xgboost(df):
    """
    Applies the specific preprocessing steps found in the XGBoost.ipynb notebook.
    This MUST be identical to the function in the training script.
    """
    print("Starting data preprocessing for XGBoost inference...")
    df = df.drop_duplicates()
    if 'CLASS' in df.columns:
        df = df.drop(columns=['CLASS'])
    cols_to_drop = ['b', 'e', 'DR']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df = df.dropna()
    print("Preprocessing complete.")
    return df

if __name__ == "__main__":
    # --- 1. Load Trained Artifacts ---
    print("Loading trained XGBoost model and scaler from 'models/' directory...")
    model = joblib.load('models/xgboost_model.joblib')
    scaler = joblib.load('models/scaler_xgb.joblib')

    # --- 2. Load New Data for Inference ---
    # We'll use the first 5 rows of the original data as "new" data.
    new_data = pd.read_csv('data/processed/final_processed_data.csv').head(5)
    
    # Ensure the target column 'NSP' is dropped if it exists
    if 'NSP' in new_data.columns:
        new_data_features = new_data.drop(columns=['NSP'])
    else:
        new_data_features = new_data

    # --- 3. Preprocess New Data ---
    new_data_processed = preprocess_data_for_xgboost(new_data_features.copy())

    # --- 4. Apply Scaling ---
    new_data_scaled = scaler.transform(new_data_processed)
    
    # --- 5. Make Predictions ---
    print("\nMaking predictions on new data...")
    predictions = model.predict(new_data_scaled)
    
    # Map numeric predictions (0, 1, 2) back to readable labels
    label_map = {0: "Normal", 1: "Suspect", 2: "Pathologic"}
    readable_predictions = [label_map[p] for p in predictions]

    print("\n--- Predictions ---")
    for i, prediction in enumerate(readable_predictions):
        print(f"Sample {i+1}: {prediction}")
