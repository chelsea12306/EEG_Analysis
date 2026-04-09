# Motor Imagery EEG Analysis
"""
BCI Competition IV Dataset 1 - Motor Imagery Classification
Pipeline: Data Loading -> Filtering -> CSP -> LDA Classification
All plots are saved to the 'run_MI_EEG/' folder automatically
"""

import os
import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib import mlab

# --------------------------
# Create output directory for saving plots
# --------------------------
if not os.path.exists('run_MI_EEG'):
    os.makedirs('run_MI_EEG')
print("'run_MI_EEG/' folder created for saving plots")

# --------------------------
# 1. Load Dataset and Basic Information
# --------------------------
print("\n[1/7] Loading EEG dataset...")
# Load MATLAB dataset file
mat_data = scipy.io.loadmat('dataset/data_MI_EEG/BCICIV_calib_ds1c.mat', struct_as_record=True)

# Extract basic parameters
sample_rate = mat_data['nfo']['fs'][0][0][0][0]
eeg_data = mat_data['cnt'].T  # EEG data shape: (channels, samples)
n_channels, n_samples = eeg_data.shape

# Channel names
channel_names = [s[0] for s in mat_data['nfo']['clab'][0][0][0]]

# Event markers (trial onsets and labels)
event_onsets = mat_data['mrk'][0][0][0]
event_codes = mat_data['mrk'][0][0][1]

# Class labels
class_labels = [s[0] for s in mat_data['nfo']['classes'][0][0][0]]
class1 = class_labels[0]
class2 = class_labels[1]
n_classes = len(class_labels)
n_events = len(event_onsets)

# Print dataset information
print(f'Shape of EEG: {eeg_data.shape}')
print(f'Sample rate: {sample_rate} Hz')
print(f'Number of channels: {n_channels}')
print(f'Channel names: {channel_names}')
print(f'Number of events: {n_events}')
print(f'Event codes: {np.unique(event_codes)}')
print(f'Class labels: {class_labels}')

# --------------------------
# 2. Extract Single Trials from Continuous EEG
# --------------------------
print("\n[2/7] Extracting EEG trials...")
# Time window for each trial: 0.5s - 2.5s after stimulus onset
time_window = np.arange(int(0.5 * sample_rate), int(2.5 * sample_rate))
window_length = len(time_window)

# Store trials for each class
trials = {}

# Extract trials for both classes
for label, code in zip(class_labels, np.unique(event_codes)):
    # Get onset indices for current class
    class_onsets = event_onsets[event_codes == code]
    n_trials = len(class_onsets)
    
    # Initialize array: (channels, time points, trials)
    trials[label] = np.zeros((n_channels, window_length, n_trials))
    
    # Slice EEG data for each trial
    for i, onset in enumerate(class_onsets):
        trials[label][:, :, i] = eeg_data[:, time_window + onset]

print(f'Shape of trials [{class1}]: {trials[class1].shape}')
print(f'Shape of trials [{class2}]: {trials[class2].shape}')

# --------------------------
# 3. Power Spectral Density (PSD) Calculation
# --------------------------
def calculate_psd(trials_data):
    """
    Calculate Power Spectral Density (PSD) for all trials and channels
    Args:
        trials_data: 3D array (channels x samples x trials)
    Returns:
        psd_results: PSD values (channels x freq bins x trials)
        frequencies: Frequency axis
    """
    n_trials = trials_data.shape[2]
    psd_results = np.zeros((n_channels, 101, n_trials))
    
    for trial_idx in range(n_trials):
        for ch in range(n_channels):
            psd, freqs = mlab.psd(
                trials_data[ch, :, trial_idx],
                NFFT=int(window_length),
                Fs=sample_rate
            )
            psd_results[ch, :, trial_idx] = psd.ravel()
    
    return psd_results, freqs

def plot_psd_results(psd_dict, freqs, channel_indices, channel_labels, max_y, save_name):
    """
    Plot average PSD for selected channels and save the figure
    """
    plt.figure(figsize=(12, 5))
    n_chans = len(channel_indices)
    n_rows = int(np.ceil(n_chans / 3))
    n_cols = min(3, n_chans)

    for i, ch in enumerate(channel_indices):
        plt.subplot(n_rows, n_cols, i + 1)
        for cls in psd_dict.keys():
            plt.plot(freqs, np.mean(psd_dict[cls][ch, :, :], axis=1), label=cls)
        
        plt.xlim(1, 30)
        plt.ylim(0, max_y)
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.title(channel_labels[i])
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'run_MI_EEG/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PSD plot saved to run_MI_EEG/{save_name}.png")

# Compute and plot raw PSD
psd_class1, freqs = calculate_psd(trials[class1])
psd_class2, _ = calculate_psd(trials[class2])
psd_raw = {class1: psd_class1, class2: psd_class2}

# Plot PSD for C3, Cz, C4 channels
plot_channels = [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']]
plot_psd_results(
    psd_raw, freqs, plot_channels,
    ['left', 'center', 'right'], 500, 'psd_raw'
)

# --------------------------
# 4. Bandpass Filter (8-15 Hz, Mu/Beta Rhythm)
# --------------------------
def bandpass_filter(trials_data, low_freq, high_freq, fs):
    """
    Apply 6th-order IIR bandpass filter to EEG trials
    """
    # Design filter (normalized by Nyquist frequency)
    nyq = fs / 2.0
    b, a = scipy.signal.iirfilter(6, [low_freq/nyq, high_freq/nyq])
    
    n_trials = trials_data.shape[2]
    filtered = np.zeros_like(trials_data)
    
    # Apply zero-phase filtering
    for i in range(n_trials):
        filtered[:, :, i] = scipy.signal.filtfilt(b, a, trials_data[:, :, i], axis=1)
    
    return filtered

print("\n[3/7] Applying 8-15 Hz bandpass filter...")
# Filter data for both classes
trials_filtered = {
    class1: bandpass_filter(trials[class1], 8, 15, sample_rate),
    class2: bandpass_filter(trials[class2], 8, 15, sample_rate)
}

# Plot filtered PSD
psd_filt1, _ = calculate_psd(trials_filtered[class1])
psd_filt2, _ = calculate_psd(trials_filtered[class2])
psd_filtered = {class1: psd_filt1, class2: psd_filt2}
plot_psd_results(
    psd_filtered, freqs, plot_channels,
    ['left', 'center', 'right'], 300, 'psd_filtered'
)

# --------------------------
# 5. Log-Variance Feature Extraction
# --------------------------
def log_variance(trials_data):
    """Calculate log-variance of each channel across time"""
    return np.log(np.var(trials_data, axis=1))

def plot_log_variance(logvar_dict, save_name):
    """Plot log-variance bar chart for all channels"""
    plt.figure(figsize=(12, 5))
    x1 = np.arange(n_channels)
    x2 = x1 + 0.4

    mean1 = np.mean(logvar_dict[class1], axis=1)
    mean2 = np.mean(logvar_dict[class2], axis=1)

    plt.bar(x1, mean1, width=0.5, color='blue', label=class1)
    plt.bar(x2, mean2, width=0.4, color='red', label=class2)

    plt.xlim(-0.5, n_channels + 0.5)
    plt.grid(axis='y')
    plt.title('Log-Variance of Each Channel')
    plt.xlabel('Channels')
    plt.ylabel('Log-Variance')
    plt.legend()
    plt.savefig(f'run_MI_EEG/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Log-variance plot saved to run_MI_EEG/{save_name}.png")

# Compute log-variance for filtered data
logvar_filtered = {
    class1: log_variance(trials_filtered[class1]),
    class2: log_variance(trials_filtered[class2])
}
plot_log_variance(logvar_filtered, 'logvar_filtered')

# --------------------------
# 6. Common Spatial Pattern (CSP)
# --------------------------
def covariance_matrix(trials_data):
    """Compute average covariance matrix across all trials"""
    n_trials = trials_data.shape[2]
    cov_list = [(trials_data[:, :, i] @ trials_data[:, :, i].T) / window_length for i in range(n_trials)]
    return np.mean(cov_list, axis=0)

def whitening_transform(cov_matrix):
    """Compute whitening matrix using SVD"""
    U, lam, _ = np.linalg.svd(cov_matrix)
    return U @ np.diag(lam ** -0.5)

def csp_projection(trials_c1, trials_c2):
    """Compute CSP projection matrix W"""
    cov1 = covariance_matrix(trials_c1)
    cov2 = covariance_matrix(trials_c2)
    white_mat = whitening_transform(cov1 + cov2)
    B, _, _ = np.linalg.svd(white_mat.T @ cov2 @ white_mat)
    return white_mat @ B

def apply_csp(proj_mat, trials_data):
    """Apply CSP projection to EEG trials"""
    n_trials = trials_data.shape[2]
    csp_data = np.zeros_like(trials_data)
    for i in range(n_trials):
        csp_data[:, :, i] = proj_mat.T @ trials_data[:, :, i]
    return csp_data

print("\n[4/7] Computing CSP projection...")
# Compute CSP and apply to filtered data
csp_matrix = csp_projection(trials_filtered[class1], trials_filtered[class2])
trials_csp = {
    class1: apply_csp(csp_matrix, trials_filtered[class1]),
    class2: apply_csp(csp_matrix, trials_filtered[class2])
}

# Plot CSP log-variance
logvar_csp = {
    class1: log_variance(trials_csp[class1]),
    class2: log_variance(trials_csp[class2])
}
plot_log_variance(logvar_csp, 'logvar_csp')

# Plot CSP PSD for key components
psd_csp1, _ = calculate_psd(trials_csp[class1])
psd_csp2, _ = calculate_psd(trials_csp[class2])
psd_csp = {class1: psd_csp1, class2: psd_csp2}
plot_psd_results(
    psd_csp, freqs, [0, 28, -1],
    ['first component', 'middle component', 'last component'],
    0.75, 'psd_csp'
)

# --------------------------
# 7. Train-Test Split (50% / 50%)
# --------------------------
print("\n[5/7] Splitting data into train/test sets...")
train_ratio = 0.5
n_train1 = int(trials_filtered[class1].shape[2] * train_ratio)
n_train2 = int(trials_filtered[class2].shape[2] * train_ratio)

# Split filtered data
train_data = {
    class1: trials_filtered[class1][:, :, :n_train1],
    class2: trials_filtered[class2][:, :, :n_train2]
}
test_data = {
    class1: trials_filtered[class1][:, :, n_train1:],
    class2: trials_filtered[class2][:, :, n_train2:]
}

# Train CSP ONLY on training set
csp_train = csp_projection(train_data[class1], train_data[class2])

# Apply CSP
train_data[class1] = apply_csp(csp_train, train_data[class1])
train_data[class2] = apply_csp(csp_train, train_data[class2])
test_data[class1] = apply_csp(csp_train, test_data[class1])
test_data[class2] = apply_csp(csp_train, test_data[class2])

# Select first and last CSP components (most discriminative)
selected_comps = np.array([0, -1])
train_data[class1] = train_data[class1][selected_comps, :, :]
train_data[class2] = train_data[class2][selected_comps, :, :]
test_data[class1] = test_data[class1][selected_comps, :, :]
test_data[class2] = test_data[class2][selected_comps, :, :]

# Compute log-variance features
train_data[class1] = log_variance(train_data[class1])
train_data[class2] = log_variance(train_data[class2])
test_data[class1] = log_variance(test_data[class1])
test_data[class2] = log_variance(test_data[class2])

# --------------------------
# 8. LDA Classifier
# --------------------------
def plot_csp_scatter(data1, data2, title, save_name, decision_boundary=None):
    """Scatter plot of CSP log-variance features"""
    plt.figure(figsize=(8, 6))
    plt.scatter(data1[0, :], data1[-1, :], color='blue', label=class1)
    plt.scatter(data2[0, :], data2[-1, :], color='red', label=class2)
    
    if decision_boundary is not None:
        W, b = decision_boundary
        x = np.arange(-5, 1, 0.1)
        y = (b - W[0] * x) / W[1]
        plt.plot(x, y, 'k--', linewidth=2, label='Decision Boundary')
    
    plt.xlabel('First CSP Component')
    plt.ylabel('Last CSP Component')
    plt.title(title)
    plt.legend()
    plt.xlim(-3, 1)
    plt.ylim(-2.5, 1)
    plt.savefig(f'run_MI_EEG/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to run_MI_EEG/{save_name}.png")

def train_lda(feature1, feature2):
    """Train Linear Discriminant Analysis (LDA) classifier"""
    n1, n2 = feature1.shape[0], feature2.shape[0]
    prior1, prior2 = n1/(n1+n2), n2/(n1+n2)
    
    mean1, mean2 = np.mean(feature1, axis=0), np.mean(feature2, axis=0)
    centered1, centered2 = feature1 - mean1, feature2 - mean2
    
    cov1 = (centered1.T @ centered1) / (n1 - 2)
    cov2 = (centered2.T @ centered2) / (n2 - 2)
    
    pooled_cov = prior1 * cov1 + prior2 * cov2
    W = (mean2 - mean1) @ np.linalg.pinv(pooled_cov)
    b = (prior1 * mean1 + prior2 * mean2) @ W
    return W, b

def predict_lda(features, W, b):
    """LDA prediction"""
    preds = []
    for i in range(features.shape[1]):
        score = W @ features[:, i] - b
        preds.append(1 if score <= 0 else 2)
    return np.array(preds)

print("\n[6/7] Training LDA classifier...")
# Train LDA
W_lda, b_lda = train_lda(train_data[class1].T, train_data[class2].T)
print(f'LDA Weight W: {W_lda}')
print(f'LDA Bias b: {b_lda}')

# Plot training data with decision boundary
plot_csp_scatter(
    train_data[class1], train_data[class2],
    'Training Data (CSP Features)', 'train_scatter',
    decision_boundary=(W_lda, b_lda)
)

# Plot test data
plot_csp_scatter(
    test_data[class1], test_data[class2],
    'Test Data (CSP Features)', 'test_scatter',
    decision_boundary=(W_lda, b_lda)
)

# --------------------------
# 9. Evaluate Classification Performance
# --------------------------
print("\n[7/7] Evaluating model accuracy...")
# Predict test set
pred1 = predict_lda(test_data[class1], W_lda, b_lda)
pred2 = predict_lda(test_data[class2], W_lda, b_lda)

# Confusion matrix
conf_matrix = np.array([
    [np.sum(pred1 == 1), np.sum(pred2 == 1)],
    [np.sum(pred1 == 2), np.sum(pred2 == 2)]
])

accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

print('\nConfusion Matrix:')
print(conf_matrix)
print(f'\nClassification Accuracy: {accuracy:.3f}')
print("\nAll processing completed! Plots are saved in 'run_MI_EEG/' folder")