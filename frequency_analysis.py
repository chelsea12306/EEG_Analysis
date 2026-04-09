#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSVEP Classification Based on Canonical Correlation Analysis (CCA)
Pipeline: Data Loading -> PSD Visualization -> CCA Feature Extraction -> Frequency Classification
All plots are automatically saved to the 'run_frequency_analysis/' folder
"""

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.mlab import psd

# --------------------------
# Create output directory for saving plots
# --------------------------
if not os.path.exists('run_frequency_analysis'):
    os.makedirs('run_frequency_analysis')
print("'run_frequency_analysis/' folder created for saving plots")

# --------------------------
# 1. Load EEG Dataset and Basic Information
# --------------------------
print("\n[1/3] Loading EEG dataset...")
# Load MATLAB format EEG data
mat_data = scipy.io.loadmat('dataset/data_frequency_analysis.mat')
print("Dataset keys:", list(mat_data.keys()))

# Extract filtered EEG data for three conditions
murder_EEG = mat_data['Murder_filteredEEG']
weapon_EEG = mat_data['Weapon_filteredEEG']
room_EEG = mat_data['Room_filteredEEG']

# Print data shape information
print(f'Shape of murder_EEG: {murder_EEG.shape}')
print(f'Shape of weapon_EEG: {weapon_EEG.shape}')
print(f'Shape of room_EEG: {room_EEG.shape}')

# Get basic signal parameters
n_channels, n_samples = murder_EEG.shape
sample_rate = 128.0  # Sampling frequency in Hz
signal_duration = n_samples / sample_rate
print(f'Duration of recordings: {signal_duration:.2f} seconds')

# --------------------------
# 2. Plot Power Spectral Density (PSD) for Visual Cortex Channels
# --------------------------
print("\n[2/3] Plotting PSD for selected channels...")
# Channels corresponding to the visual cortex region
channels_of_interest = [0, 1, 2, 11, 12, 13]

# Initialize figure with specified size
plt.figure(figsize=(10, 4.5))

# Plot PSD for each selected channel
for i, ch in enumerate(channels_of_interest):
    # Create subplot grid (2 rows, 3 columns)
    plt.subplot(2, 3, i + 1)
    
    # Compute Power Spectral Density for murder condition EEG
    murder_PSD, freqs = psd(murder_EEG[ch, :], NFFT=n_samples, Fs=sample_rate)
    
    # Plot PSD curve
    plt.plot(freqs, murder_PSD)
    
    # Set uniform axis limits for comparison
    plt.xlim(5, 20)
    plt.ylim(0, 50)
    
    # Add plot decorations
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB)')
    plt.title(f'Channel {ch}')

# Adjust layout to prevent label overlapping
plt.tight_layout()
# Save figure to run_frequency_analysis folder
plt.savefig('run_frequency_analysis/ssvep_channel_psd.png', dpi=300, bbox_inches='tight')
plt.close()
print("PSD plot saved to run_frequency_analysis/ssvep_channel_psd.png")

# --------------------------
# 3. SSVEP Classifier Using Canonical Correlation Analysis (CCA)
# --------------------------
def ssvep_classifier(EEG, frequencies, sample_rate):
    '''
    SSVEP classifier based on Canonical Correlation Analysis (CCA).
    Given a list of frequencies, this function returns the frequency
    that is most strongly present in the input EEG signal.
    
    Parameters
    ----------
    EEG : 2D-array (channels x samples)
        Input EEG signal for classification
    frequencies : list
        List of target frequencies to check
    sample_rate : float
        Sampling rate of the EEG signal in Hz

    Returns
    -------
    int : Dominant SSVEP frequency
    '''
    nchannels, nsamples = EEG.shape
    
    # Transpose EEG: rows = time samples, columns = channels
    X = EEG.T

    # Store correlation scores for each frequency
    scores = []
    
    # Generate time axis in seconds
    time = np.arange(nsamples) / float(sample_rate)
    
    # Compute correlation for each target frequency
    for frequency in frequencies:
        # Generate reference signals (fundamental + 2 harmonics)
        y = 2 * np.pi * frequency * time
    
        # Construct reference matrix Y with sine and cosine components
        Y = np.vstack([
            np.sin(y), np.cos(y),
            np.sin(2*y), np.cos(2*y),
            np.sin(3*y), np.cos(3*y)
        ]).T
    
        # Mean-center the data matrices
        X -= X.mean(axis=0, keepdims=True)
        Y -= Y.mean(axis=0, keepdims=True)
    
        # QR decomposition (only retain Q matrix)
        QX, _ = np.linalg.qr(X)
        QY, _ = np.linalg.qr(Y)
    
        # Singular Value Decomposition for canonical correlation
        _, D, _ = np.linalg.svd(QX.T @ QY)
        
        # Use highest correlation coefficient as score
        scores.append(D[0])
        
    # Return frequency with maximum correlation score
    return frequencies[np.argmax(scores)]

# --------------------------
# 4. Classify Frequencies for All Conditions
# --------------------------
print("\n[3/3] Performing SSVEP frequency classification...")
# Target frequencies for classification
target_freqs = [8, 10, 12, 15]

# Classify and print results
murder_freq = ssvep_classifier(murder_EEG, target_freqs, sample_rate)
weapon_freq = ssvep_classifier(weapon_EEG, target_freqs, sample_rate)
room_freq = ssvep_classifier(room_EEG, target_freqs, sample_rate)

print(f'Murderer frequency: {murder_freq} Hz.')
print(f'Weapon frequency:   {weapon_freq} Hz.')
print(f'Room frequency:      {room_freq} Hz.')

print("\nAll processing completed successfully!")
print("Generated plot saved in 'run_frequency_analysis/' folder")