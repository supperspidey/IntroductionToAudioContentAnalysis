from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1., 2., 3., 4.])
x2 = np.array([1.])

# Convolving with an impulse is the same as correlating with it.
conv = signal.convolve(x1, x2, mode='full')
corr = signal.correlate(x1, x2, mode='full')
print conv, conv
