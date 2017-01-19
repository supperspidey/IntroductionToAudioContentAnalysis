import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

#   Define a general digital filter
class Filter:
    __metaclass__ = ABCMeta

    def __init__(self, b, a):
        self.a = a
        self.b = b

    @abstractmethod
    def impulse_response(self): pass

    @abstractmethod
    def frequency_response(self): pass

#   Implement a moving average filter
class MovingAverageFilter(Filter):
    def __init__(self, num_taps):
        b = 1./num_taps * np.ones(num_taps)
        super(MovingAverageFilter, self).__init__(b, 1.)

    def impulse_response(self):
        return self.b

    def frequency_response(self):
        w = np.linspace(0, np.pi, 100)
        H = np.zeros(len(w), dtype=complex)
        for i in range(0, len(self.b)):
            H += self.b[i] * np.exp(-1j*w*i)
        return w, H

#   Implement a single-pole filter with coefficient alpha
class SinglePoleFilter(Filter):
    def __init__(self, alpha):
        self.alpha = alpha
        b = 1. - alpha
        a = np.array([1., -alpha])
        super(SinglePoleFilter, self).__init__(b, a)

    def impulse_response(self):
        #   Single-pole filters are IIR, meaning there is feedback. Therefore,
        #   I only compute the first 10 values of the impulse response.
        h = np.zeros(10)

        #   Note: Here I convolve the impulse response with an impulse to get
        #   the impulse response out.
        for n in range(0, len(h)):
            if n == 0:
                h[n] = (1 - self.alpha) * 1.
            else:
                h[n] = self.alpha * h[n-1]
        return h

    def frequency_response(self):
        w = np.linspace(0, np.pi, 100)
        H = self.b / (self.a[0] + self.a[1]*np.exp(-1j*w))
        return w, H

maf = MovingAverageFilter(30)
h_maf = maf.impulse_response()
plt.subplot(2, 2, 1)
plt.stem(h_maf)
plt.title('Impulse Response - Moving Average')

w, H_maf = maf.frequency_response()
plt.subplot(2, 2, 3)
Hmag_maf = np.absolute(H_maf)
plt.plot(w, Hmag_maf)
plt.xlim([w.min(), w.max()])
plt.xlabel('Frequency (Rad)')
plt.ylabel('Gain')
plt.title('Magnitude Response - Moving Average')

spf = SinglePoleFilter(0.4)
h_spf = spf.impulse_response()
plt.subplot(2, 2, 2)
plt.stem(h_spf)
plt.title('Impulse Response - Single-Pole')

w, H_spf = spf.frequency_response()
plt.subplot(2, 2, 4)
Hmag_spf = np.absolute(H_spf)
plt.plot(w, Hmag_spf)
plt.xlim([w.min(), w.max()])
plt.xlabel('Frequency (Rad)')
plt.ylabel('Gain')
plt.title('Magnitude Response - Single-Pole')

plt.tight_layout()
plt.show()
