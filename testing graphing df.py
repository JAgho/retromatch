import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa

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


# --- PLOTTING PULSE DATA ---
dtype = np.dtype([('Pulse', '<u4'),('b', '<u4'),('c', '<u4')])

bin = np.fromfile('binary_stream.bin', dtype = dtype)

initial_pulse_t = bin[0][1]
bin = bin.reshape(-1, 1)

bin['b'] = bin['b'] - initial_pulse_t

flat_bin = bin.reshape(-1)
df = pd.DataFrame(flat_bin)

ax = df.plot(x = 'b', y = 'Pulse', kind = 'line', color = 'red')
ax.set_ylabel("Signal")

# --- PLOTTING AUDIO DATA ---
audio_timings = pd.read_csv('scanner_block_1_segments_segments_clean.csv')
    
for index, row in audio_timings.iterrows():
    start_sec = row['start_time']
    end_sec = row['end_time']
    
    y_audio, sr = load_audio('Hyperfine_1.m4a', start_time=start_sec, end_time=end_sec)

    t_audio = np.linspace(start_sec * 1000, end_sec * 1000, len(y_audio))
    ax.plot(t_audio, y_audio, label='Audio')

# --- FORMATTING GRAPH ---
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
plt.legend()
plt.xlabel("Time/ms")
plt.title("Synchronised Cardiac Cycle and MRI Readout")
plt.show()