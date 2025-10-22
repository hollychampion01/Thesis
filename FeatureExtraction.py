# Author: Holly Champion
# Year: 2025
# Title: Feature Extraction
# Description: Extract feature vectors from normalised waveforms which can be fed into ML models

import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy

# Load data file
name = input("Enter filename (without extension): ").strip() 
filename = name + "_normalised.npy"
waveforms = np.load(filename, allow_pickle=True)

# Dictionary for feature storage
features_list = []

# Loop through all waveforms
for i, waveform in enumerate(waveforms):

    # Helper variables
    d_wave = np.diff(waveform)                                                  # First derivative
    dd_wave = np.diff(d_wave)                                                   # Second derivartive
    peak_idx = 0                                                                # Index of peak
    trough_idx = np.argmin(waveform)                                            # Index of trough
    fft = np.fft.rfft(waveform)                                                 # Fast fourier transform
    fft_magnitude = np.abs(fft)                                                 # Absolute energy at each frequency

    # Basic Statistics: Describe shape
    auc = np.trapezoid(waveform)                                                # Area under curve
    mean = np.mean(waveform)                                                    # Mean amplitude
    std = np.std(waveform)                                                      # Amplitude variability
    skewness = skew(waveform)                                                   # Skewness
    kurt = kurtosis(waveform)                                                   # Kurtosis (how peaked/flat the wave is)
    entropy_val = entropy(np.histogram(waveform, bins=25, density=True)[0])     # Amplitude randomness (Shannon entropy)

    # Temporal Features: When important events happen within a waveform
    time_to_peak = peak_idx / len(waveform)                                     # Normalised time to reach peak
    rise_fraction = (peak_idx - trough_idx) / len(waveform)                     # Fraction of wave spent rising
    decay_fraction = 1 - rise_fraction                                          # Fraction of wave spent decaying

    # Derivative-based features: Slope/curve information
    max_d = np.max(d_wave)                                                      # Steepest upward slope
    min_d = np.min(d_wave)                                                      # Steepest downward slope
    mean_d = np.mean(d_wave)                                                    # Average slope
    std_d= np.std(d_wave)                                                       # Slope variability
    max_dd = np.max(dd_wave)                                                    # Maximum curvature
    min_dd = np.min(dd_wave)                                                    # Minimum curvature
    sign_changes = np.sum(np.diff(np.sign(dd_wave)) != 0)                       # Number of sign changes

    # Frequency-domain features
    spectral_centroid = (np.sum(np.arange(len(fft_magnitude)) * fft_magnitude)  # Centre of mass of centroid
        / np.sum(fft_magnitude))
    spectral_entropy = entropy(fft_magnitude / np.sum(fft_magnitude))           # Randomness of normalised spectrum

    # Output all features as a dictionary
    features = {
        "auc": auc,
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": entropy_val,
        "time_to_peak": time_to_peak,
        "rise_fraction": rise_fraction,
        "decay_fraction": decay_fraction,
        "max_derivative": max_d,
        "min_derivative": min_d,
        "mean_derivative": mean_d,
        "std_derivative": std_d,
        "max_dd": max_dd,
        "min_dd": min_dd,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "sign_changes": sign_changes
    }

    features_list.append(features)

# Convert feature list into a data frame
features_df = pd.DataFrame(features_list)

# Export to a new csv file
output_file = name.replace("normalised", "") + "_features.csv"
features_df.to_csv(output_file, index=False)
print(f"Saved extracted features → {output_file}")