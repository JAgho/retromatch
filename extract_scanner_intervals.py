import numpy as np
import librosa
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import csv
import pandas as pd
from scanner_cycle_detection import load_audio, find_high_rms_window, extract_pca_templates, match_template, threshold_correlation
from extract_audio_phase import load_audio, extract_features, cluster_features, get_segments, assign_rms_energy, assign_event_labels, save_dataframe_to_csv, extract_clean_cycles

file_path = "/Users/Ritvik/Desktop/Retromatch/retromatch/Hyperfine_1.m4a"

# ----------- find scanner periods -----------
def detect_scanner_blocks(file_path):
    y, sr = load_audio(file_path)
    start, end = find_high_rms_window(y, sr, window_sec=40.0, step_sec=1.0)
    print(f"High energy block: {start/sr:.2f}s to {end/sr:.2f}s")

    templates = extract_pca_templates(y, sr, start, end, num_templates=5, window_duration=8.0)
    max_len = max(len(t) for t in templates)
    padded = [np.pad(t, (0, max_len - len(t))) for t in templates]
    avg_template = np.mean(padded, axis=0)

    corr, times = match_template(y, sr, avg_template)
    scanner_blocks = threshold_correlation(corr, times)

    print("\nDetected scanner blocks:")
    for i, (s, e) in enumerate(scanner_blocks):
        print(f"Block {i+1}: {s:.2f}s to {e:.2f}s")

    return scanner_blocks, avg_template, corr, times


# === full pipeline ===
def run_audio_labeling(filename, start_time=None, end_time=None, hop_length=512, n_clusters=4):
    y, sr = load_audio(filename, start_time, end_time)

    # offset: needed to shift relative times to full-audio absolute times
    global_offset = start_time if start_time is not None else 0.0

    X, rms, centroid, bandwidth = extract_features(y, sr, hop_length)
    labels = cluster_features(X, n_clusters)

    # Pass offset into get_segments
    df_segments = get_segments(labels, sr, hop_length, time_offset=global_offset)

    df_segments = assign_rms_energy(df_segments, y, sr)
    df_segments = assign_event_labels(df_segments)
    return df_segments


# Full audio and sample rate
y, sr = librosa.load(file_path, sr=None)

# Store results
block_dfs = []
scanner_blocks, avg_template, corr, times = detect_scanner_blocks(file_path)


for i, (start, end) in enumerate(scanner_blocks):
    print(f"\nRunning labeling on scanner block {i+1}: {start:.2f}s to {end:.2f}s")
    
    df_block = run_audio_labeling(file_path, start_time=start, end_time=end)
    
    # Add block ID as a column
    df_block["scanner_block"] = i + 1
    
    block_dfs.append(df_block)


block_dfs = []

for i, (start, end) in enumerate(scanner_blocks):
    print(f"\nProcessing scanner block {i+1}: {start:.2f}s to {end:.2f}s")

    df_block = run_audio_labeling(file_path, start_time=start, end_time=end)
    df_block["scanner_block"] = i + 1
    block_dfs.append(df_block)

    # Save raw
    save_dataframe_to_csv(df_block, filename=f"scanner_block_{i+1}_segments", clean=False)

    # Save cleaned
    df_clean = extract_clean_cycles(df_block)
    if df_clean is not None:
        save_dataframe_to_csv(df_clean, filename=f"scanner_block_{i+1}_segments", clean=True)
    else:
        print(f"Block {i+1}: no clean cycles found.")






