import scipy.signal
import numpy as np
from numpy.fft import fft

#   N: Frequency resolution; must be greater than block length M
#   M: Block length
#   H: Hop length
#   Fs: Sampling frequency
#   data: Audio data
def spectrogram(N, M, H, Fs, data):
    start = 0
    end = 0
    previous_end = 0
    window = np.hanning(M)
    STFT = np.zeros((N, len(data)))

    while end < data.size:
        # Extract each block of data
        previous_end = end
        end = start + M
        if end > data.size:
            end = data.size
        samples = data[start:end]

        # Zero-pad the block to fill up buffer of size N
        if N - samples.size > 0:
            samples = np.float64(np.pad(
                samples,
                (0, N-samples.size),
                'constant',
                constant_values=0.0
            ))

        # Window the data
        samples[0:M] = np.multiply(samples[0:M], window)

        # Transform the buffer
        FFT = np.square(np.absolute(fft(samples)))
        if start == 0:
            for i in range(start, end):
                STFT[:,i] = FFT[::-1]
        else:
            for i in range(start, previous_end):
                STFT[:,i] = np.add(STFT[:,i], FFT[::-1])
            for i in range(previous_end, end):
                STFT[:,i] = FFT[::-1]
        start = start + H

    return STFT
