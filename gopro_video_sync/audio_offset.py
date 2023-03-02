from os.path import join
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
from moviepy.editor import VideoFileClip
from scipy.io import wavfile

from .cross_correlation import get_offset


def get_wav_data(video: str, sample_rate=8000) -> Tuple[int, np.ndarray]:
    """
    Given the path to a video file, returns a tuple of (rate, data) where rate
    is the sample rate and data is a numpy array with WAV audio data.
    """
    clip = VideoFileClip(video)
    with TemporaryDirectory() as temp_folder:
        wav_filepath = join(temp_folder, "audio.wav")
        clip.audio.write_audiofile(
            wav_filepath, codec="pcm_s16le", logger=None, fps=sample_rate
        )
        return wavfile.read(wav_filepath)


def get_audio_offset(
    wav_1: Tuple[int, np.ndarray], wav_2: Tuple[int, np.ndarray]
) -> float:
    """
    Given the sample rate and data from two WAV streams, determines the offset
    of `wav_2` from `wav_1` using cross-correlation.

    Returns the offset in seconds that would have to be added to the beginning
    of `wav_2` for it to line up with `wav_1` (or vice versa for a negative
    value).
    """

    sample_rate_1, data_1 = wav_1
    sample_rate_2, data_2 = wav_2
    if sample_rate_1 != sample_rate_2:
        raise ValueError("Audio sample rates do not match.")

    # Get rid of extra channel if stereo
    if len(data_1.shape) > 1:
        data_1 = data_1[:, 0]
    if len(data_2.shape) > 1:
        data_2 = data_2[:, 0]

    return get_offset(data_1, data_2, sample_rate_1)
