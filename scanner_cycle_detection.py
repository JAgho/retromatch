# scanner_cycle_detection.py

import numpy as np
import librosa
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# ----------- STEP 1: Load and Preprocess Audio -----------
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    print(f"Loaded audio: {len(y)/sr:.2f}s @ {sr}Hz")
    return y, sr


def find_high_rms_window(y, sr, window_sec=40, step_sec=1):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    
    window_len = int(window_sec * sr / 512)
    step_len = int(step_sec * sr / 512)
    
    best_score = -np.inf
    best_start_idx = 0

    for i in range(0, len(rms) - window_len, step_len):
        window = rms[i:i+window_len]
        mean_energy = np.mean(window)
        if mean_energy > best_score:
            best_score = mean_energy
            best_start_idx = i

    start_time = times[best_start_idx]
    end_time = times[best_start_idx + window_len]
    print(f"Best window: {start_time:.2f}s to {end_time:.2f}s | Mean RMS: {best_score:.4f}")
    return start_time, end_time

'''
# ----------- STEP 2: Find a High-Energy Block for Template Extraction -----------
def find_high_energy_block(y, sr, window_duration=40.0, scan_duration=300, step_sec=1.0):
    window_len = int(window_duration * sr)
    step = int(step_sec * sr)
    scan_samples = min(len(y), int(scan_duration * sr))
    energy = np.abs(y[:scan_samples])
    smoothed = np.convolve(energy, np.ones(int(0.1 * sr))/int(0.1 * sr), mode='same')
    threshold = np.percentile(smoothed, 75)
    
    best_score, best_start = -np.inf, 0i
    for i in range(0, len(smoothed) - window_len, step):
        seg = smoothed[i:i + window_len]
        mean, std = np.mean(seg), np.std(seg)
        if mean < threshold:
            continue
        score = mean - std
        if score > best_score:
            best_score, best_start = score, i
    return best_start, best_start + window_len
'''

# ----------- STEP 3: Extract PCA Templates from High-Energy Region -----------
def extract_pca_templates(y, sr, start_sample, end_sample, num_templates, window_duration):
    segment = y[start_sample:end_sample]
    energy = np.abs(segment)
    smoothed = np.convolve(energy, np.ones(int(0.1 * sr))/int(0.1 * sr), mode='same')
    peaks, _ = find_peaks(smoothed, distance=int(sr * window_duration / num_templates))
    peaks = peaks[:num_templates]
    
    templates = []
    for peak in peaks:
        seg = y[start_sample + peak: start_sample + peak + int((window_duration/num_templates)*sr)]
        D = librosa.stft(seg, n_fft=2048, hop_length=512, window='hann')
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        pca = PCA(n_components=1).fit_transform(S_db.T).flatten()
        pca = (pca - np.mean(pca)) / np.std(pca)
        pca -= np.max(pca)
        templates.append(pca)
    return templates


# ----------- STEP 4: Cross-Correlate Average Template with Full Audio -----------
def match_template(y, sr, avg_template):
    D_full = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db_full = librosa.amplitude_to_db(np.abs(D_full), ref=np.max)
    pca_full = PCA(n_components=1).fit_transform(S_db_full.T).flatten()
    pca_full = (pca_full - np.mean(pca_full)) / np.std(pca_full)
    
    avg_template = (avg_template - np.mean(avg_template)) / np.std(avg_template)
    corr = np.correlate(pca_full, avg_template, mode='valid')
    norm_corr = (corr - np.min(corr)) / (np.max(corr) - np.min(corr))
    times = librosa.frames_to_time(np.arange(len(norm_corr)), sr=sr, hop_length=512)
    return norm_corr, times


# ----------- STEP 5: Threshold Correlation to Find Blocks -----------
def threshold_correlation(corr, times, threshold=0.9, min_gap_sec=5.0):
    binary = (corr > threshold).astype(int)
    on_indices = np.where(binary == 1)[0]
    if len(on_indices) == 0:
        return []

    intervals = []
    start, prev = on_indices[0], on_indices[0]
    for idx in on_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            intervals.append((start, prev))
            start, prev = idx, idx
    intervals.append((start, prev))

    merged = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if times[start] - times[cur_end] < min_gap_sec:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    return [(times[s], times[e]) for s, e in merged]

