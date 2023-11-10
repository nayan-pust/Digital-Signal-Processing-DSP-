import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

fs = 100
t = np.arange(0, 1.01, 1 / fs)
s = np.sin(2 * np.pi * t * 5) + np.sin(2 * np.pi * t * 15) + np.sin(2 * np.pi * t * 30)
plt.subplot(3, 1, 1)
plt.plot(t, s)

wp1 = 10 / 50
wp2 = 20 / 50
ws1 = 5 / 50
ws2 = 25 / 50
wp = np.array([wp1, wp2])
ws = np.array([ws1, ws2])
rp = 0.1
rs = 40
[n, w] = sig.ellipord(wp, ws, rp, rs)
[b, a] = sig.ellip(n, rp, rs, w, btype='bandpass')
w, h = sig.freqz(b, a, 512, fs=fs)
plt.subplot(3, 1, 2)
plt.plot(w, np.abs(h))

sf = sig.lfilter(b, a, s)
plt.subplot(3, 1, 3)
plt.plot(t, sf)
plt.show()
