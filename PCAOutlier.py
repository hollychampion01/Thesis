# Author: Holly Champion
# Year: 2025
# Title: PCA Outlier
# Description: Detects anomalies via PCA (Principal Component Analysis) reconstruction error.

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings

# Ignore warnings (harmless run-time warnings from near-singular covariance)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load data
base = input("Enter person name: ").strip()
compress_50 = pd.read_csv(f"OriginalData/{base}_50_features.csv").values
compress_80 = pd.read_csv(f"OriginalData/{base}_80_features.csv").values
normal      = pd.read_csv(f"OriginalData/{base}_normal_features.csv").values

# Split data into training (80%) and testing (20%)
normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=42)

# Standardise so each feature has zero mean and unit variance to ensure features hold equal wait
scale = StandardScaler()
scaled_train = scale.fit_transform(normal_train)      # Learn mean and std from training data
scaled_test  = scale.transform(normal_test)           # Apply that scale to test data
scaled_50    = scale.transform(compress_50)           # Apply that scale to compress_50 data
scaled_80    = scale.transform(compress_80)           # Apply that scale to compress_80 data

# Fit PCA on normal stabilized training data
pca = PCA(n_components=0.99, svd_solver="full")       # Keep 99% of variance
pca.fit(scaled_train)                                 # Learn from scaled normal data

# Calculates reconstruction error
def recon_error(X):
    X_proj = pca.transform(X)                         # Project data into PCA space
    X_recon = pca.inverse_transform(X_proj)           # Reconstruct data back into original feature space
    return np.mean((X - X_recon) ** 2, axis=1)        # Compute mean squared error between original and reconstructed samples

# Calculate errors
err_train = recon_error(scaled_train)
err_test  = recon_error(scaled_test)
err_50    = recon_error(scaled_50)
err_80    = recon_error(scaled_80)

# Normal threshold so that there is 5% false positives
threshold = np.percentile(err_test, 95)

# Ratio of samples above threshold (anomaly rate)
ratio_norm = (err_train > threshold).mean()
ratio_test  = (err_test  > threshold).mean()
ratio_50   = (err_50    > threshold).mean()
ratio_80   = (err_80    > threshold).mean()

# print results
print("\nPCA Outlier Detection: Amount of detected abnormalities")
print(f"Normal (train)   > thr: {ratio_norm:.3f}")
print(f"Normal (test)    > thr: {ratio_test:.3f}")
print(f"Compression 50%  > thr: {ratio_50:.3f}")
print(f"Compression 80%  > thr: {ratio_80:.3f}")

# PCA Information
print(f"\nExplained variance ratio sum: {pca.explained_variance_ratio_.sum():.3f}")
print(f"Number of PCA components: {pca.n_components_}")