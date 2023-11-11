# Digital-Signal-Processing-DSP-

# Z transform-------
```ruby
import control
import matplotlib.pyplot as plt

# Define a continuous-time transfer function
f = control.tf([1, 1], [1, 1, 1])
print("Continuous-time transfer function:", f)

# Convert the continuous-time transfer function to discrete-time with a sampling time of 1 second
z = control.c2d(f, 1)
print("Discrete-time transfer function:", z)

t, h = control.impulse_response(f)
plt.stem(t, h)
plt.show()

# Plot poles and zeros of the discrete-time transfer function
control.pzmap(z, True, 'Poles and Zeros')
plt.title('Poles and Zeros')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.grid(True)
plt.show()

# Calculate and print the poles of the discrete-time transfer function
poles = control.pole(z)
print("Poles:", poles)
zero = control.zeros(z)
print("Zeroes:", zero)
```
# FFT-------
```ruby
import numpy as np
import matplotlib.pyplot as plt

fs = 128
N = 256
T = 1 / fs
k = np.arange(N)
time = k * T
f = 0.25 + 2 * np.sin(2 * np.pi * 5 * k * T) + 1 * np.sin(2 * np.pi * 12.5 * k * T) + 1.5 * np.sin(
    2 * np.pi * 20 * k * T) + 0.5 * np.sin(2 * np.pi * 35 * k * T)

# Plot original signal
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(time, f)
axs[0].set_title('Signal sampled at 128Hz')

# Calculate FFT and plot frequency components
F = np.fft.fft(f)
magF = np.abs(np.hstack((F[0] / N, F[1:N // 2] / (N / 2))))
hertz = k[0:N // 2] * (1 / (N * T))
axs[1].stem(hertz, magF)
axs[1].set_title('Frequency Components')
plt.tight_layout()
plt.show()
```

# DFT-----
```ruby
import numpy as np
import matplotlib.pyplot as plt

# MATLAB-style plot formatting
plt.style.use('ggplot')

# Define variables
n = np.arange(-1, 4)
x = np.arange(1, 6)
N = len(n)
k = np.arange(len(n))

# Calculate Fourier transform
X = np.sum(x * np.exp(-2j * np.pi * np.outer(n, k) / N), axis=1)
magX = np.abs(X)
angX = np.angle(X)
realX = np.real(X)
imagX = np.imag(X)

# Plot Fourier transform components
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs[0, 0].plot(k, magX)
axs[0, 0].grid(True)
axs[0, 0].set_xlabel('Frequency in pi units')
axs[0, 0].set_title('Magnitude part')

axs[0, 1].plot(k, angX)
axs[0, 1].grid(True)
axs[0, 1].set_xlabel('Frequency in pi units')
axs[0, 1].set_title('Angle part')

axs[1, 0].plot(k, realX)
axs[1, 0].grid(True)
axs[1, 0].set_xlabel('Frequency in pi units')
axs[1, 0].set_title('Real part')

axs[1, 1].plot(k, imagX)
axs[1, 1].grid(True)
axs[1, 1].set_xlabel('Frequency in pi units')
axs[1, 1].set_title('Imaginary part')

plt.tight_layout()
plt.show()
```
# LowPass Filter FIR
```ruby
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

fs = 8000  # sampling rate
N = 50  # order of filer
fc = 1200  # cutoff frequency
b = sig.firwin(N + 1, fc, fs=fs, window='hamming', pass_zero='lowpass')
w, h_freq = sig.freqz(b, fs=fs)
z, p, k = sig.tf2zpk(b, 1)

plt.subplot(3, 1, 1)
plt.plot(w, np.abs(h_freq))  # magnitude
plt.xlabel('frequency(Hz)')
plt.ylabel('Magnitude')

plt.subplot(3, 1, 2)
plt.plot(w, np.unwrap(np.angle(h_freq)))  # phase
plt.xlabel('frequency(Hz)')
plt.ylabel('Phase(angel)')

plt.subplot(3, 1, 3)
plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b')
plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
plt.show()
```

