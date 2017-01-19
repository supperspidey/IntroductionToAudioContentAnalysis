import scipy.signal as signal
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt

#   Implement an auto-correlation function
def auto_correlate(x):
    x_flip = x[::-1]
    r = np.zeros(len(x) * 2 - 1)
    for n in range(0, len(x)):
        r[n:n+len(x)] = r[n:n+len(x)] + x_flip * x[n]
    lags = np.linspace(-np.ceil(len(r)/2), np.ceil(len(r)/2), len(r))
    return lags, r

#   Implement a function that finds the highest local max that is not
#   absolute max.
#   half_acf:   Half the auto-correlation result, starting from lag 0
def highest_local_max(half_acf):
    highest_max = 0
    idx_max = 0
    for n in range(1, len(half_acf)-1):
        if half_acf[n] > half_acf[n-1] and half_acf[n] > half_acf[n+1]:
            if idx_max == 0:
                highest_max = half_acf[n]
                idx_max = n
            else:
                if half_acf[n] > highest_max:
                    highest_max = half_acf[n]
                    idx_max = n
    return idx_max, highest_max

def num_blocks(nX, B, H, Fs):
    start = 0
    end = 0
    n = 0
    while start < nX and end < nX:
        n += 1
        end = start + B
        start += H

    t = np.zeros(n)
    for i in range(0, n):
        t[i] = i * H * 1./Fs

    return n, t

Fs, X = wavfile.read('sax_example.wav')
hop_length = 512
block_size = 4096
n_blocks, timestamps = num_blocks(len(X), block_size, hop_length, Fs)
block_max = np.zeros(n_blocks)

start = 0
end = 0
n = 0
while start < len(X) and end < len(X):
    end = start + block_size
    if end > len(X):
        end = len(X)
    x = np.float64(X[start:end])
    _, r = auto_correlate(x)
    zero_lag_idx = np.int64(np.ceil(len(r)/2))
    idx_max, highest_max = highest_local_max(r[zero_lag_idx:])
    block_max[n] = highest_max
    n += 1
    start += hop_length

fig, ax1 = plt.subplots()
t = np.linspace(0, 1./Fs*(len(X)-1), len(X))
ax1.plot(t, X, 'b')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Magnitude')
ax1.tick_params('x', colors='b')

ax2 = ax1.twinx()
ax2.plot(timestamps, block_max, 'r')
ax2.set_ylabel('Local Max', color='r')
ax2.tick_params('m', colors='r')

fig.tight_layout()
plt.show()
