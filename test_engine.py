import pandas as pd
from engine import process_mpesa_statement

print("Loading data...")
df = pd.read_csv("feature_engineered.csv")

print("Running engine process...")
result = process_mpesa_statement(df, user_id="test_user")

print("Engine complete.")
print(f"ML Models saved status: {result.get('ml_models_saved')}")
if 'ml_model_metadata' in result:
    print(f"Selected model: {result['ml_model_metadata'].get('selected_model')}")
    print(f"Test R2: {result['ml_model_metadata'].get('metrics', {}).get('ridge_tuned', {}).get('test_r2')}")
