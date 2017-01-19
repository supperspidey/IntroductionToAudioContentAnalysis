import numpy as np
from abc import ABCMeta, abstractmethod
from scipy import signal
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
        w = np.linspace(0, np.pi - 0.01, 512)
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
        w = np.linspace(0, np.pi, 512)
        H = self.b / (self.a[0] + self.a[1]*np.exp(-1j*w))
        return w, H

plt.subplot(2, 1, 1)
for n in [2, 10, 50]:
    maf = MovingAverageFilter(n)
    w, H_maf = maf.frequency_response()
    Hmag_maf = np.absolute(H_maf)
    plt.plot(w, 20*np.log10(Hmag_maf))
plt.xlim([0, np.pi])
plt.legend(['J = 2', 'J = 10', 'J = 50'])
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Gain (dB)')
plt.title('Magnitude Response - Moving Average')

plt.subplot(2, 1, 2)
for alpha in [0.5, 0.9, 0.999]:
    spf = SinglePoleFilter(alpha)
    w, H_spf = spf.frequency_response()
    Hmag_spf = np.absolute(H_spf)
    plt.plot(w, 20*np.log10(Hmag_spf))
plt.xlim([0, np.pi])
plt.legend(['alpha = 0.5', 'alpha = 0.9', 'alpha = 0.999'])
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Gain (dB)')
plt.title('Magnitude Response - Single-Pole')

plt.tight_layout()
plt.show()
