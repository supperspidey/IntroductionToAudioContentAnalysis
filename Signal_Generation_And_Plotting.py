import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

num_elements = 200

#   Generate a sinusoidal with 4 periods and amplitude of 0.707
f           = 50
D    = 1./f * 4
A           = 0.707
t           = np.linspace(0, D, num_elements)
s           = A * np.sin(2 * np.pi * f * t)

#   Plot the sinusoidal
plt.subplot(3, 1, 1)
plt.plot(t, s)
plt.axis([0, D, -A, A])
plt.xlabel('t (s)')
plt.ylabel('s(t)')

#   Generate white noise with uniform distribution and amplitude of 0.707
w = np.random.uniform(-A, A, num_elements)

#   Plot the noise
plt.subplot(3, 1, 2)
plt.plot(t, w)
plt.axis([0, D, -A, A])
plt.xlabel('t (s)')
plt.ylabel('w(t)')

#   Add the noise with the sinusoidal
x = s + w

#   Plot the result
plt.subplot(3, 1, 3)
plt.plot(t, x)
plt.axis([0, D, -2*A, 2*A])
plt.xlabel('t (s)')
plt.ylabel('x(t) = s(t) + w(t)')

plt.show()

#   Write the result to file
Fs = 1 / (D/num_elements)
wavfile.write('sine_and_noise.wav', Fs, x)
