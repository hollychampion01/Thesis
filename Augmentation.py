# Author: Holly Champion
# Year: 2025
# Title: Augmentation
# Description: Augments the data randomly

import numpy as np
np.random.seed(42)

# Add small global guassian noise
def jitter(wave):
    return wave + np.random.normal(0.0, 0.01, size=wave.shape)

# Add mild local noise over a small window of cycle
def local_noise(wave, window_frac=0.04):
    L = len(wave)
    window_len = int(L * window_frac)
    copy = wave.copy()

    # Choose random start location for window
    start = np.random.randint(0, L - window_len)
    copy[start:start + window_len] += np.random.normal(0.0, 0.01, size=window_len)
    return copy

# Gentle elastic time warping
def time_warp(wave):
    L = len(wave)                                   # Length of wave
    orig = np.arange(L)                             # Original time positions
    random_warp = np.random.normal(1.0, 0.05, L)    # Tiny stretch/compress factors
    warped = np.cumsum(random_warp)                 # Compute the cumulative sum (increasing time warp values)
    warped = warped / warped[-1] * (L - 1)          # Scale warped axis to for on the original length
    return np.interp(orig, warped, wave)            # Redraw waveform on the warped timeline

# Apply random augmentations chosen at random
def apply_random_augmentations(wave):
    copy = wave.copy()
    functions = [time_warp, jitter, local_noise]    # List of all augmentation functions
    np.random.shuffle(functions)                    # Shuffle functions
    k = np.random.choice([1, 2, 3])                 # Choose how many augmentations to apply
    for func in functions[:k]:                      # Apply functions
        copy = func(copy)
    return copy