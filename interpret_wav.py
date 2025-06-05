import numpy as np
import librosa
import matplotlib.pyplot as plt
import pywt

y,sr = librosa.load('Hyperfine_1.m4a', sr=None)
spect = librosa.stft(y,win_length=256)
plt.imshow(np.log(np.abs(spect[0:200, 100000:103000])),aspect='auto')
plt.show()

#print(y.size)
# q = y[13000000:13001000]
# samples = np.geomspace(1, 1024, num=1000)
# coeffs,freqs = pywt.cwt(q, samples, 'gaus1')
# print(np.shape(coeffs))
# print(freqs.size)
# plt.imshow(coeffs,aspect='auto')
# plt.show()