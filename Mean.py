import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

def num_blocks(X, B, H):
    count = 0
    start = 0
    while start + B < len(X):
        count += 1
        start += H
    return count

def arithmetic_mean(X, B, H, Fs):
    nblocks = num_blocks(X, B, H)
    means = np.zeros(nblocks)
    timestamps = np.zeros(nblocks)
    for n in range(0, nblocks):
        timestamps[n] = n * H * 1./Fs

    start = 0
    end = 0
    block_idx = 0
    while start + B < len(X):
        end = start + B
        the_sum = 0
        for n in range(start, end):
            the_sum += X[n]
        means[block_idx] = np.float64(the_sum) / (end - start)
        block_idx += 1
        start += H

    return timestamps, means

def geometric_mean(X, B, H, Fs):
    return 0

def harmonic_mean(X, B, H, Fs):
    return 0

Fs, X = wavfile.read('sax_example.wav')
hop_length = 512
block_size = 4096
timestamps, means = arithmetic_mean(X, block_size, hop_length, Fs)

fig, ax1 = plt.subplots()
t = np.linspace(0, 1./Fs*(len(X)-1), len(X))
ax1.plot(t, X, 'b')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Magnitude', color='b')
ax1.tick_params('x', colors='b')

ax2 = ax1.twinx()
ax2.plot(timestamps, means, 'r')
ax2.set_ylabel('Mean', color='r')
ax2.tick_params('m', colors='r')

fig.tight_layout()
plt.show()
