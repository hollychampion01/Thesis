# Author: Holly Champion
# Year: 2025
# Title: Isolation Forest Outlier
# Description: Detects anomalies via Isolation Forest (unsupervised tree ensemble).

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

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
    n_estimators=200,                               # Number of trees
    contamination=0.05,                             # Expected outlier fraction in normal data
    random_state=42                                 # Fixed seed for reproducibility
)
IF.fit(scaled_train)                                # Learn from scaled normal data

# Predict anomaly labels: -1 = anomaly, +1 = normal
pred_train = IF.predict(scaled_train)
pred_test  = IF.predict(scaled_test)
pred_50    = IF.predict(scaled_50)
pred_80    = IF.predict(scaled_80)

# Ratio of samples flagged as anomalies (anomaly rate)
ratio_norm = (pred_train == -1).mean()
ratio_val  = (pred_test  == -1).mean()
ratio_50   = (pred_50    == -1).mean()
ratio_80   = (pred_80    == -1).mean()

# print results
print("\nIsolation Forest Outlier Detection: Amount of detected abnormalities")
print(f"Normal (train)   > thr: {ratio_norm:.3f}")
print(f"Normal (val)     > thr: {ratio_val:.3f}")
print(f"Compression 50%  > thr: {ratio_50:.3f}")
print(f"Compression 80%  > thr: {ratio_80:.3f}")

# IF Information
print(f"\nNumber of trees: {IF.n_estimators}")