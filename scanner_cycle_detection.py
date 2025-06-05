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

# ----------- STEP 2: Detect High-Energy Blocks -----------
def detect_scanner_periods(y, sr, 
                           scan_min_duration=60.0, 
                           window_size=15.0, 
                           step_size=5.0,
                           energy_percentile=75,
                           acf_threshold=0.3,
                           merge_gap_sec=10.0):
    """
    Detect candidate scanner periods by checking for high energy and periodicity.

    Returns: List of (start_time, end_time) tuples.
    """
    hop_len = 512
    frame_len = 2048

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)

    win_len = int(window_size * sr / hop_len)
    step_len = int(step_size * sr / hop_len)
    threshold = np.percentile(rms, energy_percentile)

    candidate_blocks = []

    for i in range(0, len(rms) - win_len, step_len):
        window_rms = rms[i:i+win_len]
        mean_energy = np.mean(window_rms)
        if mean_energy < threshold:
            continue

        # Check for periodicity via autocorrelation
        acf = librosa.autocorrelate(window_rms)
        acf /= acf[0]  # Normalize
        # Suppress first peak at lag=0 and look for second peak
        peak_lag = np.argmax(acf[1:]) + 1
        peak_val = acf[peak_lag]

        if peak_val > acf_threshold:
            start = times[i]
            end = times[i + win_len]
            candidate_blocks.append((start, end))

    # --- Merge nearby or overlapping blocks ---
    merged = []
    for start, end in sorted(candidate_blocks):
        if not merged:
            merged.append([start, end])
        else:
            last_start, last_end = merged[-1]
            if start <= last_end + merge_gap_sec:
                merged[-1][1] = max(end, last_end)
            else:
                merged.append([start, end])

    # Filter out blocks that are too short
    final_blocks = [(s, e) for s, e in merged if (e - s) >= scan_min_duration]
    print(f"Detected {len(final_blocks)} scanner periods.")

    return final_blocks

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

