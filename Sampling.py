import numpy as np
from scipy import signal
from spectrogram import spectrogram
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

D   = 5         #   Duration in seconds
f0  = 1000      #   Start frequency in Hz
f1  = 20000     #   End frequency in Hz

#   Generate a sinesweep at 48 kHz
Fs1  = 48000  #   Sampling rate in Hz
A   = 0.5
t   = np.linspace(0, D, D/(1./Fs1))
x1  = A * signal.chirp(t, f0, D, f1, method='linear')

#   Compute the spectrogram, and plot the result
Sxx1 = spectrogram(512, 200, 100, Fs1, x1)

plt.subplot(2, 1, 1)
extent = [0, D, 0, Fs1]
plt.imshow(Sxx1, extent=extent, aspect='auto')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Sampling Rate = 48 kHz')

#   Generate the same sinesweep, but at only the fourth of the previous Fs
Fs2     = 12000
t       = np.linspace(0, D, D/(1./Fs2))
x2      = A * signal.chirp(t, f0, D, f1, method='linear')
Sxx2    = spectrogram(512, 200, 100, Fs2, x2)

#   The plot will show the effect of aliasing -- aliases at multiples of
#   the sampling frequency overlap with each other.
plt.subplot(2, 1, 2)
extent = [0, D, 0, Fs2]
plt.imshow(Sxx2, extent=extent, aspect='auto')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Sampling Rate = 12 kHz')

plt.tight_layout()
plt.show()

#   Write both results to wav files for perceptual testing
wavfile.write('sampling_chirp_from_1kHz_to_20kHz_at_48kHz.wav', Fs1, x1)
wavfile.write('sampling_chirp_from_1kHz_to_20kHz_at_12kHz.wav', Fs2, x2)
