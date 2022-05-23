# Import all necessary modules
import time
import numpy as  np
import matplotlib.pyplot as plt
import sounddevice as sd
from copy import copy
np.set_printoptions(precision=2)

# Set up the SDR
# Need to do this before you can adjust sampling frequency
# Drivers must be installed for this to work
# If this gives an error (eg because another program is controlling
# the SDR), close the other program and restart this kernel
from rtlsdr import RtlSdr 

sdr = RtlSdr() # Create a new sdr object (by keeping this in 
               # the block above can be used to close sdr without
               # creating a new sdr instance, which you might want to
               # do when switching to a new program
               # If this fails, try 
               # (1) running close block above
               # (2) closing other programs that may be using SDR
               # (3) restart this kernel

sdr.close()

# Set up constants.

# Radio sampling frequency. 2^21 ~= about 2Msps. 
# Supposedly the chips on these aren't good for pushing more than 2.2 MHz
fsps = 2 ** 21

# audio sampling frequency (for output)
faudiosps = 48000 

# Center frequency
fc = 94.9e6 # KUOW Seattle

# time step size between samples
dt = 1.0/fsps

# Nyquist frequency is one half of sampling frequency
nyquist = fsps /2.0

sdr.sample_rate = fsps 
sdr.center_freq = fc
sdr.gain = 22.9 # Arbitrary middling gain, selected from sdr.valid_gains_db

print("We are using ")
print("Gain (0==auto)  : ", sdr.gain)
print("Sample Rate     : ", sdr.sample_rate)
print("Center frequency: ", sdr.center_freq)
