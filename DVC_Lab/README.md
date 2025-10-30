# Spotify Tracks Popularity Prediction using DVC

## Project Overview

This project builds a reproducible machine learning pipeline to predict the popularity of Spotify tracks using Data Version Control (DVC).  
The pipeline automates data preparation, feature engineering, model training, evaluation, and versioning — with all data and models stored remotely on Google Cloud Storage (GCS).

The dataset contains audio and metadata features of Spotify tracks, including danceability, energy, loudness, valence, tempo, and acousticness, along with a popularity score (0–100).  
Using DVC ensures traceable experiments, version-controlled data/models, and reproducible results across different environments.

---

## Dataset

- **Source:** Spotify Tracks Dataset (`spotify_songs.csv`)
- **Size (after cleaning):** 113,999 tracks
- **Features:** 23 audio and metadata characteristics
- **Target:** `popularity` (integer range 0–100)

---

## Pipeline Architecture (6 Stages — DVC-Tracked)

### Stage 1: Data Cleaning
- **Script:** `scripts/data_cleaning.py`
- **Input:** `data/spotify_songs.csv`
- **Process:** Remove duplicates, handle missing values
- **Output:** `data/spotify_cleaned.csv`
- **Result:** 114,000 → 113,999 rows

### Stage 2: Feature Engineering
- **Script:** `scripts/feature_engineering.py`
- **Input:** `data/spotify_cleaned.csv`
- **Process:** Create new analytical features:
  - `energy_ratio = energy / (acousticness + 0.01)`
  - `mood_score = valence × danceability`
  - `tempo_normalized` (scaled between 0–1)
  - `duration_min = duration_ms / 60000`
  - `popularity_category` (Low/Medium/High)
- **Output:** `data/spotify_featured.csv` (26 columns)

### Stage 3: Model Training V1
- **Script:** `scripts/train_model_v1.py`
- **Algorithm:** Random Forest Regressor
- **Parameters:** `n_estimators=100`, `max_depth=15`
- **Features Used (12):**  
  `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo_normalized`, `energy_ratio`, `mood_score`, `duration_min`
- **Output:**
  - `models/spotify_model_v1.pkl`
  - `models/metrics_v1.json`

### Stage 4: Model Training V2
- **Script:** `scripts/train_model_v2.py`
- **Algorithm:** Random Forest Regressor (improved version)
- **Enhancements:**
  - Added normalization using `StandardScaler`
  - Included `key` and `mode` as new features (14 total)
  - Tuned hyperparameters (`n_estimators=150`, `max_depth=20`, `min_samples_split=5`)
- **Output:**
  - `models/spotify_model_v2.pkl`
  - `models/scaler_v2.pkl`
  - `models/metrics_v2.json`

### Stage 5: Model Comparison
- **Script:** `scripts/compare_models.py`
- **Purpose:** Compare V1 vs V2 using RMSE, MAE, and R² metrics
- **Output:** `models/comparison.txt`

### Stage 6: Results Dashboard
- **Script:** `scripts/view_results.py`
- **Purpose:** Display summary metrics and improvement statistics in the console

---

## 🧠 Results

| Metric | Model V1 | Model V2 | Improvement |
|--------|----------|----------|-------------|
| RMSE   | 18.32    | 16.41    | +10.41%     |
| MAE    | 14.64    | 12.67    | +13.48%     |
| R²     | 0.3233   | 0.4569   | +41.32%     |

**Key Takeaways:**
- Model V2 outperforms V1 with a ~41% improvement in R².
- Feature scaling and hyperparameter tuning significantly reduced error.
- The final model explains nearly 46% of variance in track popularity.

---

## 🗂 Project Structure
```
DVC_Lab/
├── data/
│   ├── spotify_songs.csv          # Raw data [DVC tracked]
│   ├── spotify_cleaned.csv        # Cleaned data [DVC tracked]
│   └── spotify_featured.csv       # Engineered features [DVC tracked]
├── models/
│   ├── spotify_model_v1.pkl       # Model V1 [DVC tracked]
│   ├── spotify_model_v2.pkl       # Model V2 [DVC tracked]
│   ├── scaler_v2.pkl              # Scaler for Model V2 [DVC tracked]
│   ├── metrics_v1.json            # V1 metrics
│   ├── metrics_v2.json            # V2 metrics
│   └── comparison.txt             # Model comparison summary
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_model_v1.py
│   ├── train_model_v2.py
│   ├── compare_models.py
│   └── view_results.py
├── .dvc/                          # DVC configuration
├── dvc.yaml                       # DVC pipeline (6 stages)
├── dvc.lock                       # Locked hashes for reproducibility
├── .git/                          # Git repository
└── README.md
```

---

## 🧰 Technologies Used

- **Python 3.x**
- **DVC** (Data Version Control)
- **Google Cloud Storage (GCS)** – Remote data/model storage
- **pandas, numpy** – Data processing
- **scikit-learn** – Model training and evaluation

---

## ⚙️ Setup & Installation

### Install dependencies
```bash
pip install "dvc[gs]" pandas numpy scikit-learn
```

### DVC Configuration (Your Environment)
```bash
# Initialize DVC
dvc init

# Add your Google Cloud remote
dvc remote add -d lab2 gs://mlops_lab

# Link your credentials JSON
dvc remote modify lab2 credentialpath mlops-476615-b41c3ac01bf8.json
```

Your `.dvc/config` will look like:
```ini
[core]
    remote = lab2
['remote "lab2"']
    url = gs://mlops_lab
    credentialpath = mlops-476615-b41c3ac01bf8.json
```

---

## ▶️ Running the Pipeline

### Run All Stages
```bash
dvc repro
```

DVC will automatically:
- Detect which stages need re-execution
- Run them in order
- Cache outputs for future runs

### Run a Specific Stage
```bash
dvc repro train_model_v2
```

---

## ☁️ DVC Remote Workflow

### Track data/models
```bash
dvc add data/spotify_songs.csv
git add data/spotify_songs.csv.dvc
git commit -m "Track raw Spotify dataset with DVC"
```

### Push data/models to GCS
```bash
dvc push
```

### Pull existing data/models from GCS
```bash
dvc pull
```

### Check sync status
```bash
dvc status -c
```

**Output:** `Cache and remote 'lab2' are in sync.`

---

## 🔁 Reproducibility

To reproduce your exact results:

1. Clone this repository
2. Install dependencies (`pip install "dvc[gs]" pandas numpy scikit-learn`)
3. Configure your Google Cloud remote and credentials
4. Run:
```bash
   dvc pull
   dvc repro
```

You'll obtain the same metrics and model outputs (`metrics_v1.json`, `metrics_v2.json`, `comparison.txt`).

---

## 🧩 Key DVC Commands Reference

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize a DVC repository |
| `dvc add <file>` | Track data/model files with DVC |
| `dvc repro` | Run the entire DVC pipeline |
| `dvc push` | Upload tracked files to remote storage |
| `dvc pull` | Download tracked files from remote |
| `dvc status -c` | Check sync between cache and remote |
| `dvc dag` | Visualize pipeline dependencies |

## 👩‍💻 Author

**Eroniction Presley**  
MLOps Lab – DVC Implementation (Spotify Track Popularity Prediction)

---

