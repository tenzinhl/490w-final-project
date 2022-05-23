# Play audio file recorded by SDRSharp

import wave
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

input = wave.open("recordings/kuow/SDRSharp_20220513_042744Z_94900000Hz_AF.wav", mode='rb')
num_samples = input.getnframes()
fs = input.getframerate()

print("fs = ", fs)

# Get audio as a python bytes object
audio = input.readframes(num_samples)

# Convert bytes to np array
audio_np = np.frombuffer(audio, dtype=np.int16)
# Note each frame in audio_np contains two values (I think it's for stereo?). This
# means each value in audio_np is duplicated.

print(f"length of audio_np: {len(audio_np)}")

# plt.figure()
# # Print some of the audio to see what it looks like
# plt.plot(audio_np[0:10000])
# plt.show()

# Play the audio
sd.play(audio_np[:45000:2], fs, blocking=True)
