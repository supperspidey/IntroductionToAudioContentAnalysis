from scipy.fftpack import fft, ifft
from scipy import signal
import numpy as np

x = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
h = np.array([1., 2., 3., 2., 1., 5., 1., 1.])

# Perform convolution between x and h
y_a = signal.convolve(x, h, mode='full')

# Perform FFT onto x and h, element-wise multiply the FFTs, then transform
# the result back to the time domain.
X = fft(x)
H = fft(h)
Y_b = np.multiply(X, H)
y_b = np.float64(ifft(Y_b))

# Why is the results between running convolve and FFT convolution are different?
# It's because FFT assumes periodic time signal. Therefore, when we perform
# FFT multiplication and then inverse transform, the result is circular
# convolution.
print y_a, y_b

# The only way to not have circular convolution is to calculate the length of
# of convolution (N + M - 1), pad extra zeros to the end of each sequence such
# that new sequences will have N+M-1 length, then perform FFT multiplication as
# we normally do.
N = len(x) + len(h) - 1
x = np.append(x, np.zeros(N - len(x)))
h = np.append(h, np.zeros(N - len(h)))
X = fft(x)
H = fft(h)
Y_c = np.multiply(X, H)
y_c = np.float64(ifft(Y_c))

print y_a, y_c
