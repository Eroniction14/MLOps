import json
import os

print("=" * 60)
print("MODEL COMPARISON: V1 vs V2")
print("=" * 60)

v1_path = "models/metrics_v1.json"
v2_path = "models/metrics_v2.json"

if not (os.path.exists(v1_path) and os.path.exists(v2_path)):
    raise FileNotFoundError("❌ Both metrics_v1.json and metrics_v2.json must exist.")

with open(v1_path, "r") as f:
    v1 = json.load(f)
with open(v2_path, "r") as f:
    v2 = json.load(f)

rmse_improve = ((v1["rmse"] - v2["rmse"]) / v1["rmse"]) * 100
mae_improve = ((v1["mae"] - v2["mae"]) / v1["mae"]) * 100
r2_improve = ((v2["r2"] - v1["r2"]) / v1["r2"]) * 100

summary = (
    f"Model Comparison:\n"
    f"RMSE: {v1['rmse']:.2f} → {v2['rmse']:.2f} ({rmse_improve:+.2f}%)\n"
    f"MAE: {v1['mae']:.2f} → {v2['mae']:.2f} ({mae_improve:+.2f}%)\n"
    f"R²: {v1['r2']:.4f} → {v2['r2']:.4f} ({r2_improve:+.2f}%)\n"
)

print(summary)

with open("models/comparison.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("✓ Comparison summary saved to models/comparison.txt")
print("=" * 60)
