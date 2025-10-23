#!/usr/bin/env python3
# Author: Holly Champion
# Year: 2025
# Title: Run Pipeline
# Description: Run file Preprocessing then Feature Extraction then Feature Significance

# Have files all be created in a new directory and delete the directory upon completion

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

print("\nPipeline completed")