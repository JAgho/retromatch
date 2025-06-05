import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import pandas as pd
from extract_audio_phase import load_audio, extract_features, cluster_features, get_segments, assign_rms_energy, assign_event_labels, save_dataframe_to_csv, extract_clean_cycles

# -------- PARAMETERS --------
file_path = "/Users/Ritvik/Desktop/Retromatch/retromatch/Hyperfine_1.m4a"
template_duration = 8.0  # seconds for PCA template extraction
window_duration = 30.0   # seconds for energy block detection
num_templates = 5
threshold = 0.07  # RMS threshold
min_peaks_per_window = 10

n_fft = 2048
hop_length = 512
hop_length_rms = 512

# -------- LOAD AUDIO --------
y, sr = librosa.load(file_path, sr=None)
print(f"Audio loaded, duration: {len(y)/sr:.2f} seconds, sample rate: {sr}")

# -------- COMPUTE RMS ENERGY --------
rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length_rms)[0]
times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length_rms)

# -------- PLOT RMS ENERGY --------
plt.figure(figsize=(12, 4))
plt.plot(times_rms, rms, label='RMS Energy')
plt.axhline(y=threshold, color='gray', linestyle='--', label=f'Threshold = {threshold}')
plt.xlabel('Time (s)')
plt.ylabel('RMS Energy')
plt.title('RMS Energy of Audio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- DETECT HIGH ENERGY WINDOWS --------
window_size_frames = int(window_duration * sr / hop_length_rms)
step_size_frames = window_size_frames

peak_groups = []
peaks, _ = find_peaks(rms, height=threshold)

for start in range(0, len(rms) - window_size_frames + 1, step_size_frames):
    end = start + window_size_frames
    window_peaks = peaks[(peaks >= start) & (peaks < end)]
    if len(window_peaks) >= min_peaks_per_window:
        peak_groups.append((start, end))

# -------- NEW SECOND-TIER FILTER: WINDOWS WITH ≥25 PEAKS --------
high_density_peak_groups = []
for start, end in peak_groups:
    window_peaks = peaks[(peaks >= start) & (peaks < end)]
    if len(window_peaks) >= 25:
        high_density_peak_groups.append((start, end))

if not high_density_peak_groups:
    raise ValueError("No windows found with at least 25 peaks for PCA template selection.")

# -------- SELECT UP TO 5 HIGHEST PEAKS --------
candidate_peak_indices = []
candidate_peak_heights = []

for start, end in high_density_peak_groups:
    group_peaks = peaks[(peaks >= start) & (peaks < end)]
    candidate_peak_indices.extend(group_peaks)
    candidate_peak_heights.extend(rms[group_peaks])

candidate_peak_indices = np.array(candidate_peak_indices)
candidate_peak_heights = np.array(candidate_peak_heights)

if len(candidate_peak_indices) == 0:
    raise ValueError("No candidate peaks found in high-density windows.")

sorted_indices = np.argsort(candidate_peak_heights)[::-1]
sorted_peak_indices = candidate_peak_indices[sorted_indices]

# -------- Enforce Minimum Spacing Between Peaks --------
final_peaks = []
last_added = -np.inf
min_distance_frames = int(template_duration * sr / hop_length_rms)

for idx in sorted_peak_indices:
    if idx - last_added >= min_distance_frames:
        final_peaks.append(idx)
        last_added = idx
    if len(final_peaks) >= num_templates:
        break

if not final_peaks:
    raise ValueError("No valid PCA template peaks found after enforcing spacing.")

final_peaks = np.array(final_peaks)
final_peaks.sort()

print(f"Selected peaks (RMS frames) for PCA templates: {final_peaks}")

# -------- PLOT FINAL SELECTED PEAKS ON RMS --------
plt.figure(figsize=(12, 4))
plt.plot(times_rms, rms, label='RMS Energy')
plt.plot(times_rms[final_peaks], rms[final_peaks], 'o', color='green', markersize=10, label='Final Selected Peaks')
plt.axhline(y=threshold, color='gray', linestyle='--', label=f'Threshold = {threshold}')
plt.xlabel('Time (s)')
plt.ylabel('RMS Energy')
plt.title('Final Selected Peaks on RMS Energy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- GENERATE PCA TEMPLATES --------
pca_templates = []

for peak_frame in final_peaks:
    peak_sample = int(librosa.frames_to_samples(peak_frame, hop_length=hop_length_rms))
    start_sample = peak_sample
    end_sample = start_sample + int(template_duration * sr)

    if end_sample > len(y):
        continue

    segment = y[start_sample:end_sample]
    D = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(S_db.T)
    pca_vector = pca_result.flatten()

    pca_vector -= np.mean(pca_vector)
    pca_vector /= np.std(pca_vector)

    max_idx = np.argmax(pca_vector)
    shift_amount = pca_vector[max_idx]
    pca_vector_shifted = pca_vector - shift_amount

    pca_templates.append(pca_vector_shifted)

# -------- AVERAGE PCA TEMPLATES --------
if not pca_templates:
    raise ValueError("No valid PCA templates were extracted.")

max_len = max(len(vec) for vec in pca_templates)
padded_templates = [np.pad(vec, (0, max_len - len(vec)), mode='constant') for vec in pca_templates]
avg_pca_template = np.mean(padded_templates, axis=0)

plt.figure(figsize=(12, 4))
plt.plot(avg_pca_template, color='darkgreen')
plt.title("Averaged PCA Template")
plt.xlabel("Frame Index")
plt.ylabel("Amplitude (shifted)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- APPLY PCA TO FULL AUDIO --------
D_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
S_db_full = librosa.amplitude_to_db(np.abs(D_full), ref=np.max)

pca_full = PCA(n_components=1)
pca_full_result = pca_full.fit_transform(S_db_full.T)
pca_full_vector = pca_full_result.flatten()

pca_full_vector -= np.mean(pca_full_vector)
pca_full_vector /= np.std(pca_full_vector)

avg_pca_template -= np.mean(avg_pca_template)
avg_pca_template /= np.std(avg_pca_template)

# -------- CROSS-CORRELATION --------
corr = np.correlate(pca_full_vector, avg_pca_template, mode='valid')
corr_norm = (corr - np.min(corr)) / (np.max(corr) - np.min(corr))
times_corr = librosa.frames_to_time(np.arange(len(corr_norm)), sr=sr, hop_length=hop_length)

# -------- BINARY WAVEFORM & MERGED BLOCKS --------
sim_threshold = 0.9
binary_waveform = (corr_norm > sim_threshold).astype(int)
min_gap_sec = 5.0
on_indices = np.where(binary_waveform == 1)[0]

if len(on_indices) == 0:
    print("No active binary regions detected.")
    merged_intervals = []
else:
    intervals = []
    start_idx = on_indices[0]
    prev_idx = on_indices[0]

    for idx in on_indices[1:]:
        if idx == prev_idx + 1:
            prev_idx = idx
        else:
            intervals.append((start_idx, prev_idx))
            start_idx = idx
            prev_idx = idx
    intervals.append((start_idx, prev_idx))

    merged_intervals = []
    cur_start, cur_end = intervals[0]

    for start, end in intervals[1:]:
        gap = times_corr[start] - times_corr[cur_end]
        if gap < min_gap_sec:
            cur_end = end
        else:
            merged_intervals.append((cur_start, cur_end))
            cur_start, cur_end = start,end
    merged_intervals.append((cur_start, cur_end))

    print("\nTop-level binary activity blocks (start, end, duration):")
    for i, (start_idx, end_idx) in enumerate(merged_intervals):
        start_time = (times_corr[start_idx])
        end_time = (times_corr[end_idx])
        duration = end_time - start_time
        print(f"Block {i+1}: Start = {start_time:.2f}s, End = {end_time:.2f}s, Duration = {duration:.2f}s")


scanner_blocks = [(times_corr[start], times_corr[end]) for start, end in merged_intervals]
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

import os

# Extract basename of file (e.g., "Hyperfine_1" from path)
audio_basename = os.path.splitext(os.path.basename(file_path))[0]

# Check if any scanner blocks were detected
if not scanner_blocks:
    print("No scanner blocks found. Exiting.")
    exit()

# Run audio labeling on each scanner block
block_dfs = []

for i, (start, end) in enumerate(scanner_blocks):
    print(f"\nProcessing scanner block {i+1}: {start:.2f}s to {end:.2f}s")

    # Label segments in the current scanner block
    df_block = run_audio_labeling(file_path, start_time=start+0.5*template_duration, end_time=end-0.5*template_duration)
    df_block["scanner_block"] = i + 1
    block_dfs.append(df_block)

    # Save raw labeled segments
    raw_filename = f"{audio_basename}_scan_period_{i+1}_label_raw"
    save_dataframe_to_csv(df_block, filename=raw_filename, clean=False)

    # Save clean cycles only
    df_clean = extract_clean_cycles(df_block)
    if df_clean is not None:
        clean_filename = f"{audio_basename}_scan_period_{i+1}_label_clean"
        save_dataframe_to_csv(df_clean, filename=clean_filename, clean=True)
    else:
        print(f"Block {i+1}: no clean cycles found.")






