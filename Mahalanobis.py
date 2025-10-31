# # Author: Holly Champion
# # Year: 2025
# # Title: Mahalanobis Significance
# # Description: Detects multivariate anomalies using Mahalanobis distance.
# # Learns from NORMAL only; threshold from Normal-val (~5% FPR).

import numpy as np
import pandas as pd
import warnings

# Ignore warnings (harmless run-time warnings from near-singular covariance)
np.seterr(over='ignore', divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*numpy\.linalg.*")

# Load data
base = input("Enter person name: ").strip()
compress_50 = pd.read_csv(f"OriginalData/{base}_50_features.csv").values
compress_80 = pd.read_csv(f"OriginalData/{base}_80_features.csv").values
normal      = pd.read_csv(f"OriginalData/{base}_normal_features.csv").values

# Mean and covariance of normal data
mean = normal.mean(axis = 0)                                # Calculate mean for each feature
cov  = np.cov(normal, rowvar = False)                       # Covariance matrix
inv_cov = np.linalg.pinv(cov)                               # Inverse of covariance matrix

# Mahalobis distance function
def calculate_mahalanobis(X):
    dist = X - mean                                         # How far is each point from centre
    scaled_dist = dist @ inv_cov                            # Matrix multiplication: scale features by correlation
    return np.sqrt(np.sum(scaled_dist * dist, axis = 1))    # Combined scaled and original distance

# Calculate mahalanobis distance for each data set
d_norm = calculate_mahalanobis(normal)
d_50   = calculate_mahalanobis(compress_50)
d_80   = calculate_mahalanobis(compress_80)

# Normal threshold so that there is 5% of false positives
threshold = np.percentile(d_norm, 95.0)

# Ratio of samples above threshold (anomaly rate)
ratio_norm = (d_norm > threshold).mean()
ratio_50 = (d_50 > threshold).mean()
ratio_80 = (d_80 > threshold).mean()

# Print results
print("\nMahalanobis Results: Amount of detected abnormalities")
print(f"Normal          > thr: {ratio_norm:.3f}")
print(f"Compression 50% > thr: {ratio_50:.3f}")
print(f"Compression 80% > thr: {ratio_80:.3f}")