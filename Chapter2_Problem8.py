from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

t1 = np.linspace(0, 1, 500)
t2 = np.linspace(0, 1, 1000)
x = signal.sawtooth(2 * np.pi * 5 * t1)
w = np.random.randn(1000)
corr = signal.correlate(x, w, mode='full')

plt.subplot(3, 1, 1)
plt.plot(t1, x)
plt.subplot(3, 1, 2)
plt.plot(t2, w)
plt.subplot(3, 1, 3)
plt.plot(corr)

plt.show()
