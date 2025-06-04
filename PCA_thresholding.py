import numpy as np
import librosa
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------- PARAMETERS --------
file_path = r"C:\Users\rcdav\Downloads\Hyperfine 1.m4a"  # CHANGE TO YOUR FILE PATH
template_duration = 8.0  # seconds
num_templates = 5
search_start = 240       # seconds
search_end = 300         # seconds
threshold = 0.5

# STFT params
n_fft = 2048
hop_length = 512

# -------- LOAD AUDIO --------
y, sr = librosa.load(file_path, sr=None)
print(f"Audio loaded, duration: {len(y)/sr:.2f} seconds, sample rate: {sr}")

# -------- FIND PEAKS TO SELECT TEMPLATE WINDOWS --------
start_sample_search = int(search_start * sr)
end_sample_search = int(search_end * sr)
search_segment = y[start_sample_search:end_sample_search]

# Smoothed energy to find peaks
energy = np.abs(search_segment)
window_smooth = int(0.1 * sr)  # 100ms smoothing
energy_smooth = np.convolve(energy, np.ones(window_smooth)/window_smooth, mode='same')

# Find peaks at least 8 seconds apart
peaks, _ = find_peaks(energy_smooth, distance=int(sr * template_duration))
peaks = peaks[:num_templates]

plt.figure(figsize=(12,4))
plt.plot(energy_smooth, label='Smoothed Energy')
plt.plot(peaks, energy_smooth[peaks], 'x', label='Detected Peaks')
plt.title("Detected Energy Peaks in Search Range")
plt.legend()
plt.show()

# -------- GENERATE PCA TEMPLATES --------
pca_templates = []

for i, peak_idx in enumerate(peaks):
    peak_sample = start_sample_search + peak_idx
    start_sample = peak_sample
    end_sample = start_sample + int(template_duration * sr)
    if end_sample > len(y):
        end_sample = len(y)
        start_sample = end_sample - int(template_duration * sr)

    segment = y[start_sample:end_sample]

    D = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(S_db.T)
    pca_vector = pca_result.flatten()

    # Normalize
    pca_vector -= np.mean(pca_vector)
    pca_vector /= np.std(pca_vector)

    # Shift so PCA vector starts at max value
    max_idx = np.argmax(pca_vector)
    shift_amount = pca_vector[max_idx]
    pca_vector_shifted = pca_vector - shift_amount

    pca_templates.append(pca_vector_shifted)

# -------- AVERAGE PCA TEMPLATES --------
max_len = max(len(vec) for vec in pca_templates)
padded_templates = [np.pad(vec, (0, max_len - len(vec)), mode='constant') for vec in pca_templates]
avg_pca_template = np.mean(padded_templates, axis=0)

plt.figure(figsize=(12,4))
plt.plot(avg_pca_template, color='darkgreen')
plt.title("Averaged PCA Template")
plt.xlabel("Frame Index")
plt.ylabel("Amplitude (shifted)")
plt.grid(True)
plt.show()

# -------- PCA ON FULL AUDIO --------
D_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
S_db_full = librosa.amplitude_to_db(np.abs(D_full), ref=np.max)

pca_full = PCA(n_components=1)
pca_full_result = pca_full.fit_transform(S_db_full.T)
pca_full_vector = pca_full_result.flatten()

# Normalize full PCA vector
pca_full_vector -= np.mean(pca_full_vector)
pca_full_vector /= np.std(pca_full_vector)

# Normalize avg template (safety)
avg_pca_template -= np.mean(avg_pca_template)
avg_pca_template /= np.std(avg_pca_template)

# -------- CROSS-CORRELATION --------
corr = np.correlate(pca_full_vector, avg_pca_template, mode='valid')

# Normalize correlation to 0-1
corr_norm = (corr - np.min(corr)) / (np.max(corr) - np.min(corr))

# Time axis for correlation results
times_corr = librosa.frames_to_time(np.arange(len(corr_norm)), sr=sr, hop_length=hop_length)

# -------- THRESHOLDING AND BLOCK DETECTION --------
above_thresh = corr_norm > threshold
rising_edges = np.where(np.diff(above_thresh.astype(int)) == 1)[0] + 1
falling_edges = np.where(np.diff(above_thresh.astype(int)) == -1)[0] + 1

if above_thresh[0]:
    rising_edges = np.insert(rising_edges, 0, 0)
if above_thresh[-1]:
    falling_edges = np.append(falling_edges, len(corr_norm) - 1)

print("Detected similarity blocks (start frame idx, end frame idx):")
for start, end in zip(rising_edges, falling_edges):
    print(f"{start}, {end}")

# -------- PLOT SIMILARITY WITH BLOCKS --------
plt.figure(figsize=(14,5))
plt.plot(times_corr, corr_norm, color='red', label='Similarity')
plt.axhline(y=threshold, color='gray', linestyle='--', label=f'Threshold = {threshold}')

for start, end in zip(rising_edges, falling_edges):
    plt.axvspan(times_corr[start], times_corr[end], color='green', alpha=0.3)

plt.title("Similarity between Averaged PCA Template and Full Audio with Threshold Blocks")
plt.xlabel("Time (seconds)")
plt.ylabel("Similarity (0 to 1)")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
