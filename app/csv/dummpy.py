import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

# =====================================================
# INPUT / OUTPUT FILES
# =====================================================
INPUT_FILE = r"artifacts\training\raw\volley\Player05\5.csv"
OUTPUT_DIR = r"app\csv"
OUTPUT_FILENAME = "subject_volley_synthetic.csv"
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# =====================================================
# Read first 5 header rows separately
# =====================================================
with open(INPUT_FILE, "r") as f:
    header_lines = [next(f).rstrip("\n") for _ in range(5)]

# =====================================================
# Read numerical data
# =====================================================
df = pd.read_csv(INPUT_FILE, skiprows=5, header=None)

# =====================================================
# Detect EMG VALUE columns automatically
# Time columns have increasing values
# EMG columns have positive & negative values
# =====================================================
emg_columns = []

for col in df.columns:
    numeric = pd.to_numeric(df[col], errors="coerce")

    if numeric.notna().sum() < len(df) * 0.9:
        continue

    if numeric.min() < 0 and numeric.max() > 0:
        emg_columns.append(col)

print("Detected EMG columns:", emg_columns)

# =====================================================
# Extract EMG matrix
# =====================================================
emg = df[emg_columns].astype(float).values

# =====================================================
# Train KNN
# =====================================================
k = 8
knn = NearestNeighbors(
    n_neighbors=k,
    algorithm="auto"
)
knn.fit(emg)

# =====================================================
# Generate synthetic EMG
# =====================================================
synthetic = np.zeros_like(emg)
noise_level = 0.02
std = np.std(emg, axis=0)
for i in range(len(emg)):
    # Pick a random real sample
    seed = np.random.randint(len(emg))
    # Find neighbours
    _, idx = knn.kneighbors(emg[seed].reshape(1, -1))
    neighbours = idx[0]
    # Pick two neighbours
    n1 = emg[np.random.choice(neighbours)]
    n2 = emg[np.random.choice(neighbours)]
    # Random interpolation
    alpha = np.random.rand()
    sample = alpha * n1 + (1 - alpha) * n2
    # Add small Gaussian noise
    sample += np.random.normal(
        0,
        std * noise_level
    )
    synthetic[i] = sample

# =====================================================
# Replace only EMG columns
# =====================================================
new_df = df.copy()
for i, c in enumerate(emg_columns):
    new_df[c] = synthetic[:, i]
# =====================================================
# Write output while preserving header
# =====================================================
with open(OUTPUT_FILE, "w", newline="") as f:
    for line in header_lines:
        f.write(line + "\n")
    new_df.to_csv(
        f,
        index=False,
        header=False
    )
print("\nDone!")
print("Synthetic file saved as:", OUTPUT_FILE)