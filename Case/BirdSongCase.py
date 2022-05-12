import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import time as t

start_time = t.time()
data, samplerate = sf.read("../Case/songs/xc160879.flac")
freq, time, Sxx = signal.spectrogram(data, samplerate, scaling='spectrum')
print("--- %s seconds ---" % (t.time() - start_time))
print(Sxx.shape)

#print(Sxx)
model = NMF(n_components=10,init='random', random_state=0,max_iter=1)
#print(model)
W = model.fit_transform(Sxx)
#print(W)
print(W.shape)
print(model.components_.shape)
H = model.components_
#print(model.components_)
print("--- %s seconds ---" % (t.time() - start_time))

fig1, ax1 = plt.subplots(1,1)
fig2, ax2 = plt.subplots(1,1)

ax1.matshow(Sxx)

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

print("--- %s seconds ---" % (t.time() - start_time))

estimatedSpectrogram = np.dot(W,H)
print(estimatedSpectrogram.shape)

ax2.matshow(estimatedSpectrogram)

plt.show()

