# Author: Holly Champion
# Year: 2025
# Title: Signal Preprocessing
# Description: This file should do the following
# (1) Apply a filter to the signal data
# (2) Segment the data (peak to peak) 
# (3) Get rid of outlier/erratic data
# (4) Output the data ready for ML models

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, sosfiltfilt, correlate

#################### Constants ####################

SAMPLING_RATE = 1000.0          # Samples per second in the raw signal
OUTLIER_TIME_WINDOW = 60.0      # Time window to compare data and find outliers
PROM_FRAC = 0.4                 # Peak prominance
DIST_FRAC = 0.55                # Minimum distance between peaks as a fraction of estimated waveform duration
MAD_SIGMA = 1.4826              # Conversion scale from MAD to STD
K_OUTLIER = 3.0                 # Outlier minimum range
BPM_MIN = 40                    # Assumption on lowest heart rate
BPM_MAX = 220                   # Assumption on highest eart rate
F_LOW = 0.15                    # Hz: Removes baseline drift.
F_HIGH = 40.0                   # Hz: This may need to be adjusted. Increase for more detail (could be noise) or reduce for less noise (could loose detail)
F_ORDER = 2                     # Butterworth filter order

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

# Median Absolute Deviation
def MAD(signal):
    signal = np.asarray(signal, float)
    return np.median(np.abs(signal - np.median(signal)))

# Bandpass filter using butterworth
def filter(x, fs, f_low = F_LOW, f_high = F_HIGH, order = F_ORDER):
    nyq = fs / 2.0
    sos = butter(order, [f_low/nyq, f_high/nyq], btype = "bandpass", output = "sos")
    return sosfiltfilt(sos, x)

#################### Load Data ####################

filename = input("Enter CSV filename (without .csv): ").strip() + ".csv"
# signal = pd.read_csv(filename, header = None).iloc[:,0].astype(float).values
signal = pd.read_csv(filename, header=None, engine="python").iloc[:, 0].astype(float).values
signal = filter(signal, SAMPLING_RATE)
t = np.arange(len(signal)) / SAMPLING_RATE

#################### Adaptive Segmentation ####################

# Calculate Range - robust method that ignores extreme outliers
high, low = np.percentile(signal, [97.5, 2.5])
robust_range = high - low
prom = PROM_FRAC * robust_range

# Find peaks using estimated waveform duration and minimum distance 
waveform_duration = estimate_waveform_duration(signal, SAMPLING_RATE)
min_distance = waveform_duration * SAMPLING_RATE * DIST_FRAC
peaks, _ = find_peaks(signal, distance = min_distance, prominence = prom)

#################### Build Waveforms (peak to peak) ####################

waveforms = []
peak_vals, trough_vals, amplitudes, durations = [], [], [], []
troughs = []

for i in range(len(peaks) - 1):
    start = peaks[i]
    end = peaks[i + 1]
    segment = signal[start:end]

    peak_local = end
    peak_val = signal[end]

    trough_local = start + np.argmin(segment)
    trough_val = signal[trough_local]

    amplitude = peak_val - trough_val
    duration = (end - start) / SAMPLING_RATE

    # Store waveforms and features
    waveforms.append((start, end, peak_local))
    peak_vals.append(peak_val)
    trough_vals.append(trough_val)
    amplitudes.append(amplitude)
    durations.append(duration)
    troughs.append(trough_local)

# Convert lists to arrays
peak_vals  = np.asarray(peak_vals)
trough_vals = np.asarray(trough_vals)
amplitudes = np.asarray(amplitudes)
durations = np.asarray(durations)
troughs = np.asarray(troughs)

#################### Ditching Bad Data ####################

# Calculate outliers using a window to account for drift overtime or changes in monitor location
window_id = (peaks[1:]/SAMPLING_RATE // OUTLIER_TIME_WINDOW).astype(int)
is_outlier = np.zeros(len(waveforms), dtype=bool)

for w in np.unique(window_id):
    
    # Identify which window the wave belongs to
    idxs = np.where(window_id == w)[0]

    # Calculate local variability of each feature using MAD scaled to STD units
    sigma_peak = MAD_SIGMA * MAD(peak_vals[idxs])
    sigma_trough = MAD_SIGMA * MAD(trough_vals[idxs])
    sigma_amp = MAD_SIGMA * MAD(amplitudes[idxs])
    sigma_duration = MAD_SIGMA * MAD(durations[idxs])

    # Find medians
    med_peak = np.median(peak_vals[idxs])
    med_trough = np.median(trough_vals[idxs])
    med_amp = np.median(amplitudes[idxs])
    med_duration = np.median(durations[idxs])

    # Mark waveforms too far away from local median as an outlier
    peak_outlier = np.abs(peak_vals[idxs] - med_peak) > K_OUTLIER * sigma_peak
    trough_outlier = np.abs(trough_vals[idxs] - med_trough) > K_OUTLIER * sigma_trough
    amplitude_outlier = np.abs(amplitudes[idxs] - med_amp) > K_OUTLIER * sigma_amp
    duration_outlier  = np.abs(durations[idxs] - med_duration) > K_OUTLIER * sigma_duration

    is_outlier[idxs] = peak_outlier | trough_outlier | amplitude_outlier | duration_outlier

#################### Visualise Data Ditching ####################

# good_data = np.sum(~is_outlier)
# bad_data = np.sum(is_outlier)

# print(f"Total Waveforms: {len(waveforms)} | Kept Waveforms: {good_data} | Ditched Waveforms: {bad_data}")

# plt.figure()
# plt.plot(t, signal, label="Signal")
# plt.scatter(t[peaks], signal[peaks])

# # Shade waves as green (kept) or red (ditched)
# for i, (start, end, _) in enumerate(waveforms):
#     if is_outlier[i]:
#         color = 'red'
#     else:
#         color = 'green'
#     plt.axvspan(t[start], t[end], color=color, alpha=0.2)

# plt.title("Waveform Plots and Kept Data")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

#################### Export Normalized Cycle Data ####################

## Normalised waveforms (lengths vary) ##
normalised_waveforms = []

for i, (start, end, _) in enumerate(waveforms):
    if is_outlier[i]:
        continue
    wave = signal[start:end].copy()
    
    # Z score standardisation
    wave_mean = np.mean(wave)
    wave_std = np.std(wave)
    epsilon = 1e-12 
    
    # Standardize: Z-score = (X - mean) / std
    wave = (wave - wave_mean) / (wave_std + epsilon)
    normalised_waveforms.append(wave)

# Save as .npy for feature extraction later
normalised_path = os.path.splitext(filename)[0] + "_normalised.npy"
np.save(normalised_path, np.array(normalised_waveforms, dtype=object), allow_pickle=True)

## Normalised waveforms (resampled to be the same length for autoencoder) ##
TARGET_LEN = 512
resampled_waves = []

for wave in normalised_waveforms:
    original_x = np.linspace(0, 1, len(wave))       # Original wave value positions
    new_x = np.linspace(0, 1, TARGET_LEN)           # Target wave value positions
    resampled = np.interp(new_x, original_x, wave)  # Linear interpolation
    resampled_waves.append(resampled)

resampled_waveforms = np.array(resampled_waves)

# Save as .npy for use in autoencoder
waveform_path = os.path.splitext(filename)[0] + "_autoencoder_ready.npy"
np.save(waveform_path, resampled_waveforms)