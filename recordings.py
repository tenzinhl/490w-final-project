"""
Defines a clean interface for storing important details about WAV files necessary for playback
and proper display.
"""

from collections import namedtuple

"""
A Recording. Currently serves as a data packing structure. Wraps information necessary for
manipulation of IQ baseband recordings

filepaths are relative to the root of repository or absolute, these will need to be changed for
other users of the repository as I do not commit recordings.
"""
Recording = namedtuple('Recording', ['filepath', 'fc', 'foffset', 'fcutoff'])

TEST_1_RECORDING = Recording(
    filepath='output/test_1.wav',
    fc=0,
    foffset=300e3,
    fcutoff=100_000)

KUOW_BAD_IQ_RECORDING = Recording(
    filepath='recordings/kuow/SDRSharp_20220513_042737Z_95140881Hz_IQ.wav',
    fc=95140881,
    foffset=-240e3,
    fcutoff=100_000)
