# Author: Holly Champion
# Year: 2025
# Title: Signal Preprocessing
# Description:
# (1) Filter signal
# (2) Segment into cycles (peak-to-peak)
# (3) Augment Data
# (4) Export cycles for ML models

import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, correlate
from Augmentation import apply_random_augmentations

#################### Constants ####################

PROM_FRAC = 0.4                # Peak prominance
DIST_FRAC = 0.55               # Minimum distance between peaks as a fraction of estimated waveform duration
BPM_MIN = 40                   # Assumption on lowest heart rate
BPM_MAX = 220                  # Assumption on highest heart rate
SAMPLING_RATE = 100            # Samples per second in the raw signal

AUG_COPIES_PER_WAVE = 10       # Augmented samples per original cycle
TARGET_LEN = 512               # Length for autoencoder input

#################### Helper Functions ####################

# Estimate typical waveform period using autocorrelation
def estimate_waveform_duration(signal, SAMPLING_RATE):

    # Centre the signal
    sig_centred = signal - np.median(signal)
    n = len(sig_centred)

    # Autocorrelation and normalisation
    autocorrelation = correlate(sig_centred, sig_centred, mode="full", method="fft")[n-1:]
    autocorrelation = autocorrelation / (autocorrelation[0] + 1e-12)

    # Convert BPM to frequency and find sample range
    fmin = BPM_MIN / 60
    fmax = BPM_MAX / 60
    shift_min = int(SAMPLING_RATE / fmax) # Shortest possible cycle
    shift_max = int(SAMPLING_RATE / fmin) # Longest possible cycle

    # Find strongest repeating pattern within that range
    search_band = autocorrelation[shift_min: shift_max + 1]
    cycle_samples = shift_min + int(np.argmax(search_band))

    return cycle_samples / SAMPLING_RATE

#################### Load Signal ####################

filename = input("Enter CSV filename (without .csv): ").strip() + ".csv"
df = pd.read_csv(filename, header=None, engine="python")

# Always true for ultrasound CSV: time + amplitude
t = df.iloc[:, 0].astype(float).values
signal = df.iloc[:, 1].astype(float).values

duration = max(t[-1] - t[0], 1e-9)
fs = (len(t) - 1) / duration

#################### Segmentation ####################

# Calculate range using robust percentiles (ignores extreme outliers)
high, low = np.percentile(signal, [97.5, 2.5])
robust_range = high - low
prom = PROM_FRAC * robust_range

# Find peaks using estimated waveform duration and minimum distance 
waveform_duration = estimate_waveform_duration(signal, fs)
min_distance = max(1, waveform_duration * fs * DIST_FRAC)

peaks, _ = find_peaks(signal, distance=min_distance, prominence=prom)

#################### Build Cycles ####################

raw_waves = []

for i in range(len(peaks) - 1):
    start = peaks[i]
    end   = peaks[i+1]

    w = signal[start:end].copy().astype(float)
    raw_waves.append(w)

#################### Augmentation ####################

all_waves = list(raw_waves)

if AUG_COPIES_PER_WAVE > 0:
    augmented_waves = []

    for w in raw_waves:
        for _ in range(AUG_COPIES_PER_WAVE):
            aug = apply_random_augmentations(w)
            aug = np.asarray(aug, dtype=float)
            if aug.size > 0 and np.isfinite(aug).all():
                augmented_waves.append(aug)

    print(f"Original cycles:  {len(raw_waves)}")
    print(f"Augmented cycles: {len(augmented_waves)}")

    # Use original + augmented for the rest of the pipeline
    all_waves.extend(augmented_waves)

#################### Normalise cycles ####################

normalised_waves = []

for w in all_waves:   # use original + augmented cycles
    w = np.asarray(w, dtype=float)

    # Z score standardisation
    wave_mean = np.mean(w)
    wave_std  = np.std(w)
    epsilon = 1e-12 

    # Standardize: Z-score = (X - mean) / std
    w_norm = (w - wave_mean) / (wave_std + epsilon)
    normalised_waves.append(w_norm)

#################### Save variable length waves ####################

normalised_path = os.path.splitext(filename)[0] + "_normalised.npy"
np.save(normalised_path, np.array(normalised_waves, dtype=object), allow_pickle=True)

#################### Save fixed length waves ####################

resampled_waves = []

for w in normalised_waves:
    original_x = np.linspace(0, 1, len(w))       # Original positions
    new_x      = np.linspace(0, 1, TARGET_LEN)   # Target positions
    resampled  = np.interp(new_x, original_x, w) # Linear interpolation
    resampled_waves.append(resampled)

resampled_waves = np.array(resampled_waves, dtype=float)

waveform_path = os.path.splitext(filename)[0] + "_autoencoder_ready.npy"
np.save(waveform_path, resampled_waves)
