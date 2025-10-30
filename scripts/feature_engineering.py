import pandas as pd
import numpy as np
import os

input_path = "data/spotify_cleaned.csv"
output_path = "data/spotify_featured.csv"

print("=" * 50)
print("FEATURE ENGINEERING STAGE - Spotify Dataset")
print("=" * 50)

if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ Input file not found: {input_path}")

df = pd.read_csv(input_path)
print(f"Loaded cleaned data: {df.shape}")

# --- Feature Engineering ---
df["energy_ratio"] = df["energy"] / (df["acousticness"] + 0.01)
df["mood_score"] = df["valence"] * df["danceability"]
df["tempo_normalized"] = (df["tempo"] - df["tempo"].min()) / (df["tempo"].max() - df["tempo"].min())
df["duration_min"] = df["duration_ms"] / 60000
df["popularity_category"] = pd.cut(df["popularity"],
                                   bins=[0, 30, 60, 100],
                                   labels=["Low", "Medium", "High"])


print("\n✓ New features created:")
print("energy_ratio, mood_score, tempo_normalized, duration_min, popularity_category")

df.to_csv(output_path, index=False)
print(f"\n✓ Featured data saved to: {output_path}")
print(f"Final shape: {df.shape}")
print("=" * 50)
