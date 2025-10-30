import json
import pandas as pd
import os

print("=" * 70)
print("🎵 SPOTIFY ML PIPELINE - RESULTS DASHBOARD 🎵")
print("=" * 70)

v1 = json.load(open("models/metrics_v1.json"))
v2 = json.load(open("models/metrics_v2.json"))

df = pd.DataFrame({
    "Model": ["V1", "V2"],
    "RMSE": [v1["rmse"], v2["rmse"]],
    "MAE": [v1["mae"], v2["mae"]],
    "R²": [v1["r2"], v2["r2"]],
    "Train Samples": [v1["n_train"], v2["n_train"]],
    "Test Samples": [v1["n_test"], v2["n_test"]]
})

print("\nModel Performance:\n")
print(df.to_string(index=False))

improve_r2 = ((v2["r2"] - v1["r2"]) / v1["r2"]) * 100
print(f"\n✓ Model V2 improved R² by {improve_r2:.2f}%")
print(f"✓ Average error reduced by {(v1['mae'] - v2['mae']):.2f} popularity points")

print("\nAll models and datasets are tracked with DVC on Google Cloud Storage.")
print("=" * 70)
