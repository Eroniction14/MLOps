import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
from datetime import datetime
import os

print("=" * 50)
print("TRAIN MODEL V2 - Improved Random Forest with Scaling")
print("=" * 50)

df = pd.read_csv("data/spotify_featured.csv")

feature_cols = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo_normalized",
    "energy_ratio", "mood_score", "duration_min", "key", "mode"
]
target_col = "popularity"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel V2 Performance → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

os.makedirs("models", exist_ok=True)

with open("models/spotify_model_v2.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)

metrics = {
    "version": "v2",
    "timestamp": datetime.now().isoformat(),
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "n_features": len(feature_cols),
    "features": feature_cols,
    "n_train": len(X_train),
    "n_test": len(X_test)
}

with open("models/metrics_v2.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Model and metrics saved.")
print("=" * 50)
