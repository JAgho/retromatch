import numpy as np
import librosa
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

# === Load and Slice Audio ===
def load_audio(filename, start_time=None, end_time=None):
    """
    Load an audio file and trim to a specified time range.

    Parameters:
        filename (str): Path to the audio file.
        start_time (float, optional): Start time in seconds.
        end_time (float, optional): End time in seconds.

    Returns:
        tuple: Tuple (y, sr) where y is the audio time series and sr is the sampling rate.
    """
    y, sr = librosa.load(filename, sr=None)
    if start_time is not None and end_time is not None:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y = y[start_sample:end_sample]
    return y, sr



# === Feature extraction ===
def extract_features(y, sr, hop_length=512, sigma=4):
    """
    Extract smoothed spectral features from an audio signal.

    Parameters:
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        hop_length (int): Hop length for short-time analysis.
        sigma (int): Standard deviation for Gaussian smoothing.

    Returns:
        tuple:
            - X (np.ndarray): Combined feature matrix (RMS, centroid, bandwidth).
            - rms (np.ndarray): Root-mean-square energy.
            - centroid (np.ndarray): Spectral centroid.
            - bandwidth (np.ndarray): Spectral bandwidth.
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]

    # Smooth features to reduce noise effects
    rms = gaussian_filter1d(rms, sigma=sigma)
    centroid = gaussian_filter1d(centroid, sigma=sigma)
    bandwidth = gaussian_filter1d(bandwidth, sigma=sigma)

    # Stack features
    X = np.stack([rms, centroid, bandwidth], axis=1)
    return X, rms, centroid, bandwidth

# === K-Means clustering ===
def cluster_features(X, n_clusters=4):
    """
    Apply K-Means clustering to feature vectors.

    Parameters:
        X (np.ndarray): Feature matrix.
        n_clusters (int): Number of clusters to form.

    Returns:
        np.ndarray: Cluster labels assigned to each feature vector.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    return kmeans.labels_

# === Combine repeating time intervals  ===
def get_segments(labels, sr, hop_length, time_offset=0.0):
    """
    Convert frame-wise cluster labels into time segments with durations.

    Parameters:
        labels (np.ndarray): Array of integer labels per frame.
        sr (int): Sampling rate.
        hop_length (int): Hop length used in feature extraction.
        time_offset (float): Optional time offset for alignment.

    Returns:
        pd.DataFrame: Table of segments with start/end time, duration, and label.
    """
    times = librosa.frames_to_time(range(len(labels)), sr=sr, hop_length=hop_length)
    times += time_offset  # Adjust times with offset
    segments = []
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append({
                'start_time': times[start_idx],
                'end_time': times[i],
                'duration': times[i] - times[start_idx],
                'label': int(labels[i - 1])
            })
            start_idx = i
    return pd.DataFrame(segments)

# === Compute RMS energy per time interval ===
def compute_rms_energy(y, sr, start_time, end_time):
    """
    Compute the mean RMS energy for a given time interval.

    Parameters:
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        float: Mean RMS energy of the interval.
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    sub_segment = y[start_sample:end_sample]
    rms = librosa.feature.rms(y=sub_segment)[0]
    return np.mean(rms)

def assign_rms_energy(df_segments, y, sr):
    """
    Add RMS energy values to each time segment in a DataFrame.

    Parameters:
        df_segments (pd.DataFrame): DataFrame with 'start_time' and 'end_time' columns.
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.

    Returns:
        pd.DataFrame: Original DataFrame with added 'rms_energy' column.
    """
    df_segments["rms_energy"] = df_segments.apply(
        lambda row: compute_rms_energy(y, sr, row["start_time"], row["end_time"]),
        axis=1
    )
    return df_segments

# === Assign semantic labels ===
def assign_event_labels(df_segments):
    """
    Assign semantic event labels ('inactive', 'pre', 'scan', 'post') to each segment
    based on relative RMS energy within fixed-size cycles of 4 segments.

    Parameters:
        df_segments (pd.DataFrame): DataFrame with 'rms_energy' values.

    Returns:
        pd.DataFrame: DataFrame with new 'event_label' column assigned.
    """
    new_labels = []
    for i in range(0, len(df_segments), 4):
        group = df_segments.iloc[i:i + 4]
        if len(group) < 4:
            continue
        min_idx = group["rms_energy"].idxmin()
        group_indices = group.index.tolist()
        inactive_pos = group_indices.index(min_idx)
        label_order = ["inactive", "pre", "scan", "post"]
        for offset, name in enumerate(label_order):
            if inactive_pos + offset < 4:
                idx = group_indices[inactive_pos + offset]
                new_labels.append((idx, name))

    df_segments["event_label"] = None
    for idx, name in new_labels:
        df_segments.at[idx, "event_label"] = name

    return df_segments


def extract_clean_cycles(df, pattern=["inactive", "pre", "scan", "post"], min_cycles=2):
    clean_segments = []
    cycle_idx = 0
    current_cycle = []

    for _, row in df.iterrows():
        label = row["event_label"]
        expected = pattern[cycle_idx]

        if label == expected:
            current_cycle.append(row)
            cycle_idx += 1

            if cycle_idx == len(pattern):
                # Completed a full cycle
                clean_segments.extend(current_cycle)
                current_cycle = []
                cycle_idx = 0  # reset for next cycle

        else:
            # Reset everything if pattern broken
            current_cycle = []
            cycle_idx = 0

    if len(clean_segments) >= min_cycles * len(pattern):
        df_clean = pd.DataFrame(clean_segments)
        start_time = df_clean["start_time"].min()
        end_time = df_clean["end_time"].max()
        print(f"Clean cycles extracted: {len(df_clean) // len(pattern)}")
        print(f"Start: {start_time:.2f}s | End: {end_time:.2f}s | Duration: {end_time - start_time:.2f}s")
        return df_clean
    else:
        print("Not enough clean cycles found.")
        return None


import os
import pandas as pd

def save_dataframe_to_csv(df, filename, clean=False, output_dir=".", verbose=True):
    """
    Save a DataFrame to CSV with optional 'clean' suffix.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): Base filename without '.csv'.
        clean (bool): Whether this is a cleaned version.
        output_dir (str): Directory to save into.
        verbose (bool): Whether to print a confirmation.
    """
    if df is None or df.empty:
        if verbose:
            print(f"[!] Skipping save for {filename} (empty or None)")
        return

    suffix = "_segments_clean.csv" if clean else "_segments.csv"
    full_path = os.path.join(output_dir, f"{filename}{suffix}")
    df.to_csv(full_path, index=False)

    if verbose:
        print(f"[✓] Saved to: {full_path}")