# Author: Holly Champion
# Year: 2025
# Title: GMM Outlier
# Description: Detects anomalies via Gaussian Mixture Model (negative log-likelihood).

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

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
scale = StandardScaler().fit(normal_train)
scaled_train = scale.transform(normal_train)        # Learn mean and std from training data
scaled_test  = scale.transform(normal_test)         # Apply that scale to test data
scaled_50    = scale.transform(compress_50)         # Apply that scale to compress_50 data
scaled_80    = scale.transform(compress_80)         # Apply that scale to compress_80 data

# Fit GMM on normal stabilized training data
GMM = GaussianMixture(
    n_components=4,                                 # Number of Guassian components
    covariance_type="full",                         # Each component has its own full covariance matrix
    random_state=42                                 # Fixed seed for reproducibility
)
GMM.fit(scaled_train)                               # Learn from scaled normal data

# Calculate anomaly scores (negative log-likelihood, higher = more anomalous)
s_train = -GMM.score_samples(scaled_train)
s_test  = -GMM.score_samples(scaled_test)
s_50    = -GMM.score_samples(scaled_50)
s_80    = -GMM.score_samples(scaled_80)

# Normal threshold so that there is 5% false positives
threshold = np.percentile(s_train, 95)

# Ratio of samples above threshold (anomaly rate)
ratio_norm = (s_train > threshold).mean()
ratio_test = (s_test  > threshold).mean()
ratio_50   = (s_50    > threshold).mean()
ratio_80   = (s_80    > threshold).mean()

# Print results
print("\GMM Clustering Outlier Detection: Amount of detected abnormalities")
print(f"Normal (train)   > thr: {ratio_norm:.3f}")
print(f"Normal (val)     > thr: {ratio_test:.3f}")
print(f"Compression 50%  > thr: {ratio_50:.3f}")
print(f"Compression 80%  > thr: {ratio_80:.3f}")

# GMM Information
print(f"\nNumber of GMM components: {GMM.n_components}")