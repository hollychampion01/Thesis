# Author: Holly Champion
# Year: 2025
# Title: Isolation Forest Outlier
# Description: Detects anomalies via Isolation Forest (unsupervised tree ensemble).

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os

# Load data
base = input("Enter person name: ").strip()
compress_50 = pd.read_csv(f"OriginalData/{base}_50_features.csv").values
compress_80 = pd.read_csv(f"OriginalData/{base}_80_features.csv").values
normal      = pd.read_csv(f"OriginalData/{base}_normal_features.csv").values

# Split data into training (80%) and testing (20%)
normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=42)

# Standardise so each feature has zero mean and unit variance to ensure features hold equal wait
scale = StandardScaler().fit(normal_train)
scaled_train = scale.transform(normal_train)        # Learn mean and std from training data
scaled_test  = scale.transform(normal_test)         # Apply that scale to test data
scaled_50    = scale.transform(compress_50)         # Apply that scale to compress_50 data
scaled_80    = scale.transform(compress_80)         # Apply that scale to compress_80 data

# Fit Isolation Forest on normal stabilized training data
IF = IsolationForest(
    n_estimators=100,                               # Number of trees
    contamination=0.05,                             # Expected outlier fraction in normal data
    random_state=42                                 # Fixed seed for reproducibility
)
IF.fit(scaled_train)                                # Learn from scaled normal data

# Anomaly scores
s_train = -IF.score_samples(scaled_train)
s_test  = -IF.score_samples(scaled_test)
s_50    = -IF.score_samples(scaled_50)
s_80    = -IF.score_samples(scaled_80)

# Normal threshold so that there is 5% of false positives
threshold = np.percentile(s_train, 95)

# Ratio of samples above threshold (anomaly rate)
ratio_norm = (s_train > threshold).mean()
ratio_val  = (s_test  > threshold).mean()
ratio_50   = (s_50    > threshold).mean()
ratio_80   = (s_80    > threshold).mean()

# Print results
print("\nIsolation Forest:")
print(f"Normal (train)   > thr: {ratio_norm:.3f}")
print(f"Normal (val)     > thr: {ratio_val:.3f}")
print(f"Compression 50%  > thr: {ratio_50:.3f}")
print(f"Compression 80%  > thr: {ratio_80:.3f}")

print(f"\nNumber of trees: {IF.n_estimators}")

# Save scores for evaluation
os.makedirs("Results", exist_ok=True)
np.save(f"Results/{base}_isoforest_norm.npy", s_test)
np.save(f"Results/{base}_isoforest_50.npy",  s_50)
np.save(f"Results/{base}_isoforest_80.npy",  s_80)