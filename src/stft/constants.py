"""
File:
    stft/constants.py

Description:
    This file stores helpful constants value.
"""


class StftParameters(float):

    n_fft = 512                     # '31.25 Hz' band
    win_length = 512                # '32 ms' window
    hop_length = 256                # '16 ms' hop
    power = None
    normalized = False
    center = True
    batch_size = 64