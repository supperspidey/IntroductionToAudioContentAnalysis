import scipy.io.wavfile
import scipy.signal
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

# Fs, frames = scipy.io.wavfile.read('Chopin_Nocturne.wav')
# print "Sampling Rate: " + str(Fs) + " Hz"
# print "Number of Frames: " + str(frames.shape[0])
# frames = np.float64(frames)

Fs = 1000
t = np.linspace(0, 20, 20000)
f0 = 0
f1 = 1000
t1 = 20
frames = scipy.signal.chirp(t, f0, t1, f1, method='linear')

#   N: Frequency resolution; must be greater than block length M
#   M: Block length
#   H: Hop length
#   Fs: Sampling frequency
#   data: Audio data
#   maxFreq: Maximum frequency to look at
#   minTime: Start timestamp of the data
#   maxTime: End timestamp of the data
def spectrogram(N, M, H, Fs, data, maxFreq, minTime, maxTime):
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

    # Plot the STFT
    extent = [minTime, maxTime, 0, maxFreq]
    nfreq_bins = np.int(np.ceil(maxFreq / (np.float64(Fs)/N)))
    plt.imshow(STFT[N-nfreq_bins:N,0:len(data)], extent=extent, aspect='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

spectrogram(
    N=512,
    M=500,
    H=250,
    Fs=Fs,
    data=frames,
    maxFreq=1000,
    minTime=0.,
    maxTime=20000./Fs
)
