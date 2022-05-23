"""
Take some IQ samples and write it to a WAV file.

I will consider this a success once it can be played back by SDRSharp
"""

import wave
from rtlsdr import RtlSdr
import tenutils
import numpy as np
import matplotlib.pyplot as plt

# Close previous instances of the sdr object
try: 
    sdr.close()
except NameError:
   pass

sdr = RtlSdr()
# Radio sampling frequency. 2^21 ~= about 2Msps. 
# Supposedly the chips on these aren't good for pushing more than 2.2 MHz
fsps = 2 ** 21

seconds = 2

N = fsps * seconds

# Nyquist frequency is one half of sampling frequency
nyquist = fsps / 2.0
# Center frequency
fc = 94.9e6 # KUOW Seattle

sdr.gain = 10
sdr.sample_rate = fsps
sdr.center_freq = fc - 3e5

samples = sdr.read_samples(N)

# This cell creates a spectrogram of the received data over time

# How many samples to use for each fft. Prefer powers of 2 for speed in fft
tenutils.complex_samples_to_spectrogram(samples)
plt.show()

# We now write to wav file.
# WAV is encoded as each frame being sample_width * nchannels wide. Channel
# 1 sample is followed immediately by channel 2 sample.
# I have yet to figure out endianness. Presumably just use sys.byte_order
interleaved_samples = np.empty(2 * len(samples), dtype=np.int16)

interleaved_samples[::2] = np.int16(np.real(samples[:]) * np.iinfo(np.int16).max)
interleaved_samples[1::2] = np.int16(np.imag(samples[:]) * np.iinfo(np.int16).max)

wav_out = wave.open("output/test.wav", mode='wb')

# First channel for I, second for Q by convention
wav_out.setnchannels(2)
# Set samples to be two bytes wide
wav_out.setsampwidth(2)
wav_out.setframerate(fsps)

wav_out.writeframesraw(interleaved_samples.tobytes())
