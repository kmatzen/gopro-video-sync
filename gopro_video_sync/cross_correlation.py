import numpy as np
from scipy.signal import correlate


def get_offset(data_1: np.ndarray, data_2: np.ndarray, sample_rate: int) -> float:
    """
    Given 2 numpy arrays and their sample rate, determines the offset of
    `data_2` from `data_1` using cross-correlation. For this to work correctly,
    the signals must share distinct features, the offset must be small relative
    to the total duration, and the streams must cover a similar time frame.

    Returns the offset of `data_2` from `data_1` in seconds (positive
    means `data_2` is behind, negative means it's ahead).
    """

    # Pad with median so that we get clearer results than padding with 0
    median_padding = np.full(data_2.size, np.median(data_1))
    data_1_padded = np.concatenate((median_padding, data_1, median_padding))

    corr = correlate(data_1_padded, data_2, mode="valid")

    shift = -1 * (np.argmax(corr) - data_2.size)
    offset = shift / sample_rate

    return offset
