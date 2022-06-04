# Python script for sandboxing how to decode SDRSharp IQ recordings
# These recordings are created by selecting the "Baseband" box when
# recording

import wave
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from tenutils import *
import recordings

# Select settings here: ----------------

# Choose a recording to play here
recording = recordings.KUOW_BAD_IQ_RECORDING

# Choose the max number of frames to use
MAX_FRAMES_TO_READ = 1_000_000

# End of settings ----------------------

filename = recording.filepath
fc = recording.fc
foffset = recording.foffset

wavinput = wave.open(filename, mode='rb')

num_samples = min(wavinput.getnframes(), MAX_FRAMES_TO_READ)
fs = wavinput.getframerate()
nyquist = fs / 2
sampwidth = wavinput.getsampwidth()
# I'm guessing there will be two channels which are going to be in-phase and quadrature?
num_channels = wavinput.getnchannels()

# Sample statistics, used initially when trying to decode WAV
# print(f"Num samples: {num_samples}")
# print(f"fs: {fs}")
# print(f"Each sample is {sampwidth} bytes wide")

# Read all of the samples. Samples have been checked as 2 bytes wide, so use int16 (need to confirm
# if it should be signed or unsigned: ANSWER: should be int16, verified by looking at graph)
sample_bytes = wavinput.readframes(num_samples)

combined_samples = np.frombuffer(sample_bytes, dtype=np.int16)

# Slice into two halves and see if it looks appropriate
isamples = combined_samples[::2]
qsamples = combined_samples[1::2]

# Form the array of complex samples
compsamples = isamples + 1j * qsamples

# Remove the dc component of the samples
compsamples = correct_dc(compsamples)

# At this point there should be an equal number of i and q samples
assert len(isamples) == len(qsamples)

# Check what the samples look like.
START_SAMPLE = 20000
GRAPH_WIDTH = 500

# used to call the equivalent of graphsamples() here, moved to bottom for
# readability

# Results: definitely seems correct to use int16 and that each frame has two samples
# one I and one Q. (This may be dependent on recording settings though)

# First step: bandpass the signal

# Cutoff freq of filter
fcutoff = recording.fcutoff

# Plot the fft of the signal with the bandpass filter to use
spectrum = np.fft.fftshift(np.fft.fft(compsamples))
freqs = np.linspace(fc - nyquist, fc + nyquist, num_samples)
mask = bandpassmask(num_samples, fs, fcutoff, foffset=foffset)
assert len(mask) == num_samples

# Scale filter when graphing so it's more clear
spectrum_max = np.max(np.abs(spectrum))

# Graph
plt.figure()
plt.title("Baseband spectrum before bandpass mask")
plt.plot(freqs/1e6, np.abs(spectrum))
plt.plot(freqs/1e6, mask * spectrum_max, 'r')
plt.show(block=False)

# Perform the bandpass on the spectrum then show the graph
filtered_spectrum = spectrum * mask
plt.figure()
plt.title("Filtered spectrum")
plt.plot(freqs/1e6, np.abs(filtered_spectrum))
plt.show(block=False)

filtered_signal = np.fft.ifft(np.fft.fftshift(filtered_spectrum))

signalfigure, signalaxs = plt.subplots(2)
signalaxs[0].set_title("Real component of raw signal (time domain)")
signalaxs[0].plot(np.real(compsamples[0:1000]))
signalaxs[1].set_title("Real component of filtered signal (Time domain)")
signalaxs[1].plot(np.real(filtered_signal[0:1000]))
signalfigure.show()

# Next step: decode FM
dtheta = fmdemod(filtered_signal)

fig = plt.figure()
plt.title("Cleaned dtheta")
plt.plot(dtheta[START_SAMPLE:START_SAMPLE+GRAPH_WIDTH])
plt.show(block=False)

# Play fm modulated audio

# Audio samples per second to play
faudiosps = 48000
audio = downsample(dtheta, fs, faudiosps)

sd.play(audio, faudiosps)

# Require user input before exiting
print("Press enter to exit...")
_ = input()


def graphsamples():
    """
    Graphs a window of the raw samples from the wav file. Initially used
    for figuring out how to decode the recordings.
    """
    fig, ax = plt.subplots(3)
    ax[0].set_title("Combined Samples")
    ax[0].plot(combined_samples[START_SAMPLE:START_SAMPLE+GRAPH_WIDTH])

    ax[1].set_title("isamples")
    ax[1].plot(isamples[START_SAMPLE:START_SAMPLE+GRAPH_WIDTH])

    ax[2].set_title("qsamples")
    ax[2].plot(qsamples[START_SAMPLE:START_SAMPLE+GRAPH_WIDTH])

    plt.show()
