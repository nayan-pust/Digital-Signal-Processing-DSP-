import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import lcapy as lc
from lcapy.discretetime import n

x = 1 / 16 ** n
Z = x.ZT()
print(Z)

x = Z.IZT()
print(x)

num = [1, 0, 0, 1]
den = [1, 0, 2, 0, 1]
[z, p, k] = sig.tf2zpk(num, den)
plt.subplot(1, 1, 1)
plt.scatter(np.real(z), np.imag(z), edgecolors='b', marker='o')
plt.scatter(np.real(p), np.imag(p), color='b', marker='x')
plt.show()
