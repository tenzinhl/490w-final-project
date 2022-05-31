"""
Correct Direct Current offset from RTLSDR
"""

import wave
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tenutils
import recordings

def correct_dc(samples):
    """
    Takes an array of complex samples and returns a corrected version of the samples 
    such that the DC offset is 0
    """
    return samples - np.average(samples)

recording = recordings.TEST_1_RECORDING
MAX_FRAMES_TO_READ = 1_000_000

complex_samples = tenutils.read_wav_to_complex_samples(recording.filepath, MAX_FRAMES_TO_READ)

# Verify it looks correct
tenutils.complex_samples_to_fourier(complex_samples, fc=1e6)
plt.title("Fourier Transform before DC correction")

corrected_samples = correct_dc(complex_samples)
tenutils.complex_samples_to_fourier(corrected_samples)
plt.title("Fourier Transform after DC correction")
plt.show()
