"""
tenutils - Tenzin's utility functions for his CSE 490W project

Intended to be useful even as a general purpose simple DSP library
"""

from cv2 import normalize
import numpy as np
from math import fmod
import math

# TODO: many of the type hints which ask for np.array's long-term should be switched
# to use np.typing.ArrayLike. Given that I currently don't enable type checking and
# primarily use np.arrays, this isn't currently a priority.


def normangle02(angle: float) -> float:
    """Normalizes an angle in radians to the range [0, 2pi)"""
    pi_2 = np.pi * 2
    # Double fmod required to handle negative numbers
    return fmod(fmod(angle, pi_2) + pi_2, pi_2)


def normangle(angle: float) -> float:
    """Normalizes an angle in radians to the range [-pi, pi]"""
    retval: float = normangle02(angle)
    if retval > np.pi:
        retval -= 2 * np.pi

    return retval


def bandpassmask(N: int, fsps: int, fcutoff: float, foffset: float = 0):
    """
    Return a bandpassmask in the frequency domain for a signal. Basically
    a numpy array of 0's and 1's in the right spot

    :param N: the number of elements in the bandpass mask. Should match the length
        of the fourier transform you plan on applying this mask to.
    :param fsps: the sampling frequency
    :param fcutoff: the cutoff frequency. Will be applied in both directions 
        from the offset frequency
    :param foffset: the center of your bandpass window
    :return: the bandpass mask
    """
    # Credit to Joshua Smith, although the original code was slightly bugged. And was not
    # guaranteed to return a length N vector. That bug has been fixed below
    if fcutoff <= 0 or fsps <= 0:
        raise ValueError("fcutoff and fsps must be strictly positive")

    fnyq = fsps/2.0

    if math.fabs(foffset) + fcutoff > fnyq:
        raise ValueError(
            "Bandpass window must be contained within nyquist frequencies")

    if fcutoff >= fnyq:
        return np.ones(N)

    # Get proportion of cutoff relative to nyquist
    fcuttoff_nyq = fcutoff / fnyq

    mask_width: int = math.floor(fcuttoff_nyq * N)
    # Entire window will be 2 * fnyq wide, hence weird factor of 2. Implicitly canceled
    # when calculating width as fcutoff extends in both directions.
    offset_idx: int = math.floor(foffset / (2 * fnyq) * N)
    mask_start_idx = ((N - mask_width) // 2) + offset_idx

    ones = np.ones(mask_width)
    mask = np.zeros(N)

    mask[mask_start_idx: mask_start_idx + mask_width] += ones

    return mask


def clean_dtheta(theta: float, maxdelta: float = np.pi) -> float:
    """
    Pre: maxdelta >= 0
    Normalizes an angle and filters it to 0 if its magnitude > maxdelta
    """
    if maxdelta < 0:
        raise ValueError("maxdelta must be greater than 0!")

    normval = normangle(theta)
    if math.fabs(normval) > maxdelta:
        normval = 0

    return normval


def fmdemod(samples: np.array, clean=True) -> np.array:
    """
    Get the FM demodulated array of samples given an array of complex samples.

    :param samples: the array of complex samples to demodulate
    :param clean: whether or not to perform cleaning on the demodulated signal.
        If false, most signals will have large jumps where the angle crosses the
        0-line and jumps from 0 to 2pi.
    :return: a numpy array with length `len(samples) - 1` that has been FM demodulated
    """
    if (len(samples) < 2):
        raise ValueError("samples must have at least 2 values")

    # First calculate theta at each point
    thetas = np.arctan2(np.imag(samples), np.real(samples))

    # Take discrete derivative using neat little convolution trick
    dtheta = np.convolve(thetas, [1, -1], mode='same')
    # We slice off the first element as that doesn't make sense as a derivative
    dtheta = dtheta[1:]

    if clean:
        dtheta = [clean_dtheta(x) for x in dtheta]

    return dtheta


def downsample(samples: np.array, fsamples: int, ftarget: int, averaging: bool = True) -> np.array:
    """
    Downsamples an array of samples taken at frequency fsamples to a smaller
    array equivalent to taking the samples at frequency ftarget. 

    :param samples: the samples to downsample
    :fsamples: the frequency the samples were taken at
    :ftarget: the target frequency of the resulting samples
    :averaging: whether or not to average windows of the samples when downsampling
    """
    # TODO: currently this method suffers from rounding errors. By isntead doing the rounding
    # at each step instead of once at the beginning on the downsampling factor, we could 
    # converge towards the ideal resulting sampling frequency on long (not even necessarily)
    # that long) sample windows.
    
    # Down-sampling factor
    dsfactor = round(fsamples / ftarget)

    if not averaging:
        # Sample in the middle of what would otherwise be an averaging window
        return samples[dsfactor // 2::dsfactor]
    else:
        result = []
        for x in range(len(samples) // dsfactor):
            result.append(np.average(samples[x * dsfactor : (x + 1) * dsfactor]))
        return result

