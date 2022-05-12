import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import time as t

start_time = t.time()
data, samplerate = sf.read("../Case/songs/xc26789.flac")
freq, time, Sxx = signal.spectrogram(data, samplerate, scaling='spectrum')
print("--- %s seconds ---" % (t.time() - start_time))

Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

print("--- %s seconds ---" % (t.time() - start_time))

#print(Sxx)
model = NMF(n_components=10,init='random', random_state=0,max_iter=200)
#print(model)
W = model.fit_transform(Sxx)
print(W)
print(model.components_)
print("--- %s seconds ---" % (t.time() - start_time))



plt.show()

