# Author: Holly Champion
# Year: 2025
# Title: Feature Significance
# Description: Rank individual features by z-score detection of anomalies

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

# Load data
base = input("Enter person name: ").strip()

compress_50 = pd.read_csv(f"OriginalData/{base}_50_features.csv").values
compress_80 = pd.read_csv(f"OriginalData/{base}_80_features.csv").values
normal      = pd.read_csv(f"OriginalData/{base}_normal_features.csv")

features = list(normal.columns) # Get names of features
normal = normal.values.astype(float)

rows = []

for j, name in enumerate(features):

    # Normal baselines
    mean = normal[:, j].mean()
    std = normal[:, j].std() + 1e-8 

    # Z-scores
    z_norm = (normal[:, j] - mean) / std
    z_50  = (compress_50[:, j] - mean) / std
    z_80  = (compress_80[:, j] - mean) / std

    # True labels
    compress_50_true = np.r_[np.zeros_like(z_norm), np.ones_like(z_50)]
    compress_80_true = np.r_[np.zeros_like(z_norm), np.ones_like(z_80)]

    # Anomaly scores 
    compress_50_score = np.r_[np.abs(z_norm), np.abs(z_50)]
    compress_80_score = np.r_[np.abs(z_norm), np.abs(z_80)]

    # Compute Area Under the Receiving Operating Characteristic Curve
    auc_50 = roc_auc_score(compress_50_true, compress_50_score)
    auc_80 = roc_auc_score(compress_80_true, compress_80_score)

    rows.append({
        "feature": name,
        "AUROC_vs_50": auc_50,
        "AUROC_vs_80": auc_80,
    })

res = pd.DataFrame(rows)

# Save and display
# output = f"{base}_zscore_feature_screening.csv"
# res.to_csv(output, index=False)
print("\n")
print(res)