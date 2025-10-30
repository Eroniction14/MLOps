import pandas as pd
import numpy as np
import os

# Paths
input_path = "data/spotify_songs.csv"
output_path = "data/spotify_cleaned.csv"

print("=" * 50)
print("DATA CLEANING STAGE - Spotify Dataset")
print("=" * 50)

# Load raw data
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ Input file not found: {input_path}")

df = pd.read_csv(input_path)
print(f"Original dataset shape: {df.shape}")

# --- Data Cleaning ---
print("\nRemoving duplicates and missing values...")
initial_rows = len(df)

df = df.drop_duplicates()
df = df.dropna()

removed = initial_rows - len(df)
print(f"✓ Removed {removed} rows (duplicates or nulls)")
print(f"✓ Final shape: {df.shape}")

# Save cleaned data
os.makedirs("data", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\n✓ Cleaned dataset saved to: {output_path}")
print("=" * 50)
