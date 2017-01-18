import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

def quantize(x, num_bits):
    num_steps = 2**num_bits
    step_size = 2. / num_steps
    steps = np.linspace(-1, 1 - step_size, num_steps)
    y = np.zeros(len(x))
    for n in range(0, len(y)):
        diff = np.abs(x[n] - steps)
        y[n] = steps[np.argmin(diff)]
    return y

#   Load the sample file
Fs, x = wavfile.read('sax_example.wav')
print "Sampling Rate: " + str(Fs) + " Hz"
print "Number of Frames: " + str(x.shape[0])

#   Normalize the signal
x_max   = np.float64(np.abs(x).max())
x       = x / x_max

#   Plot the normalized version
plt.subplot(3, 1, 1)
t = np.linspace(0, 1./Fs * (len(x)-1), len(x))
plt.plot(t, x)
plt.xlabel('Time (s)')

#   Quantize the signal using 4 bits
x_4bits = quantize(x, num_bits=4)
plt.subplot(3, 1, 2)
plt.plot(t, x_4bits)
plt.xlabel('Time (s)')

#   Quantize the signal using 8 bits
x_8bits = quantize(x, num_bits=8)
plt.subplot(3, 1, 3)
plt.plot(t, x_8bits)
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

#   Write to files for perceptual testing
wavfile.write('quantization_4bits.wav', Fs, x_4bits)
wavfile.write('quantization_8bits.wav', Fs, x_8bits)
