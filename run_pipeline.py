#!/usr/bin/env python3
# Author: Holly Champion
# Year: 2025
# Title: Run Pipeline
# Description: Run file Preprocessing then Feature Extraction then Feature Significance

import subprocess

name = input("Enter person name: ").strip()
type = ["normal", "50", "80"]

for t in type:
    base = f"{name}_{t}"
    base_path = f"OriginalData/{base}"

    print(f"\nPreprocessing {base}")
    subprocess.run(["python", "Preprocessing.py"], input=(base_path + "\n").encode("utf-8"), check=True)

    print(f"\nFeature Extraction {base}")
    subprocess.run(["python", "FeatureExtraction.py"], input=(base_path + "\n").encode("utf-8"), check=True)

print(f"\nFeature Significance {name}")
subprocess.run(["python", "FeatureSignificance.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nMahalanobis {name}")
subprocess.run(["python", "Mahalanobis.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nPCA Outlier {name}")
subprocess.run(["python", "PCAOutlier.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nGMM Clustering {name}")
subprocess.run(["python", "GMM.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nIsolation Forest{name}")
subprocess.run(["python", "IsolationForest.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nCNN Autoencoder {name}")
subprocess.run(["python", "CNNAE.py"], input=(name + "\n").encode("utf-8"), check=True)

print("\nPipeline completed")