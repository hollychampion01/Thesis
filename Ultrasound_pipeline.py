# Author: Holly Champion
# Year: 2025
# Title: Signal Preprocessing
# Description: # Description: Runs the full ultrasound processing pipeline including preprocessing, augmenting data,
# extracting features and executing all anomaly detection models

import os
import glob
import subprocess
import numpy as np

ULTRA_DIR = "CSV Ultrasound"

# Map classes to match thesis labels through dictionary
GROUPS = {
    "normal": "Normal",
    "50": "LightStenosis",
    "80": "FirmStenosis",
}

# Loop through ultrasound subfiles and preprocess each
for label, prefix in GROUPS.items():
    pattern = os.path.join(ULTRA_DIR, f"{prefix}*.csv")
    for csv_path in sorted(glob.glob(pattern)):
        base_path = os.path.splitext(csv_path)[0]
        print(f"\nUltrasound Preprocessing {base_path}")
        subprocess.run(["python", "UltrasoundPreprocessing.py"], input=(base_path + "\n").encode("utf-8"), check=True)

# Combine datasets into their own groups (normal, light stenosis and firm stenosis)
def concat_npy(pattern_list, out_path, allow_pickle=False):
    arrays = []

    # Load any npy file that matches the pattern
    for pattern in pattern_list:
        for npy_path in sorted(glob.glob(pattern)):
            arr = np.load(npy_path, allow_pickle=allow_pickle)
            arrays.append(arr)

    # Stack all waveforms into one big array
    if allow_pickle:
        # Variable length waveforms
        items = []
        for arr in arrays:
            items.extend(arr)
        merged = np.array(items, dtype=object)
    else:
        # Fixed length waveforms
        merged = np.concatenate(arrays, axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, merged)
    print(f"Saved {out_path}")

OUT_DIR = "OriginalData"
name = base = "Ultrasound"

# Variable-length waveform features
concat_npy([os.path.join(ULTRA_DIR, "Normal*_normalised.npy")], os.path.join(OUT_DIR, f"{base}_normal_normalised.npy"), allow_pickle=True)
concat_npy([os.path.join(ULTRA_DIR, "LightStenosis*_normalised.npy")], os.path.join(OUT_DIR, f"{base}_50_normalised.npy"), allow_pickle=True)
concat_npy([os.path.join(ULTRA_DIR, "FirmStenosis*_normalised.npy")], os.path.join(OUT_DIR, f"{base}_80_normalised.npy"), allow_pickle=True)

# Fixed-length waveforms
concat_npy([os.path.join(ULTRA_DIR, "Normal*_autoencoder_ready.npy")], os.path.join(OUT_DIR, f"{base}_normal_autoencoder_ready.npy"))
concat_npy([os.path.join(ULTRA_DIR, "LightStenosis*_autoencoder_ready.npy")], os.path.join(OUT_DIR, f"{base}_50_autoencoder_ready.npy"))
concat_npy([os.path.join(ULTRA_DIR, "FirmStenosis*_autoencoder_ready.npy")], os.path.join(OUT_DIR, f"{base}_80_autoencoder_ready.npy"))

print("\nUltrasound preprocessing complete")

#################### Print the number of waveforms ####################

print("\nWaveform counts:")

for label in ["normal", "50", "80"]:
    norm_path = f"OriginalData/Ultrasound_{label}_normalised.npy"
    arr = np.load(norm_path, allow_pickle=True)

    print(f"{label}: {len(arr)} waveforms")


#################### Run the rest of the pipeline ####################

type = ["normal", "50", "80"]

for t in type:
    base = f"{name}_{t}"
    base_path = f"OriginalData/{base}"

    print(f"\nFeature Extraction {base}")
    subprocess.run(["python", "FeatureExtraction.py"], input=(base_path + "\n").encode("utf-8"), check=True)

print(f"\nFeature Significance {name}")
subprocess.run(["python", "FeatureSignificance.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nMahalanobis {name}")
subprocess.run(["python", "Mahalanobis.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nEvaluating Mahalanobis {name}")
subprocess.run(["python", "Evaluation.py"], input=("mahalanobis\n" + name + "\n").encode("utf-8"), check=True)

print(f"\nPCA Outlier {name}")
subprocess.run(["python", "PCAOutlier.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nEvaluating PCA {name}")
subprocess.run(["python", "Evaluation.py"], input=("pca\n" + name + "\n").encode("utf-8"), check=True)

print(f"\nGMM Clustering {name}")
subprocess.run(["python", "GMM.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nEvaluating GMM {name}")
subprocess.run(["python", "Evaluation.py"], input=("gmm\n" + name + "\n").encode("utf-8"), check=True)

print(f"\nIsolation Forest {name}")
subprocess.run(["python", "IsolationForest.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nEvaluating Isolation Forest {name}")
subprocess.run(["python", "Evaluation.py"], input=("isoforest\n" + name + "\n").encode("utf-8"), check=True)

print(f"\nCNN Autoencoder {name}")
subprocess.run(["python", "CNNAE.py"], input=(name + "\n").encode("utf-8"), check=True)

print(f"\nEvaluating CNN Autoencoder {name}")
subprocess.run(["python", "Evaluation.py"], input=("cnn\n" + name + "\n").encode("utf-8"), check=True)

print("\nPipeline completed")