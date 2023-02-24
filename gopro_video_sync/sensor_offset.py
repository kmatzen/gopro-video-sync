import numpy as np
from scipy.signal import correlate

from .gopro_accel_gyro import Sensor3AxisStream


def regularize_stream_timescale(
    stream1: Sensor3AxisStream, stream2: Sensor3AxisStream, same_length: bool = False
) -> Sensor3AxisStream:
    """
    Given 2 sensor streams, regularizes the timescale of `stream2` to match
    `stream1`, duplicating/skipping samples as necessary. If `same_length` is
    `True`, `stream2` is trimmed/extended to match the length of `stream`,
    padding the end with the last value present in `stream2` if necessary.

    Returns the regularized `stream2` as a Sensor3AxisStream object.
    """

    new_data = []
    i = 0
    length = (
        stream1.sample_count
        if same_length
        else int(stream1.sample_count / stream1.duration * stream2.duration)
    )
    for i in range(length):
        index = round(
            ((stream1.duration / stream1.sample_count) * i / stream2.duration)
            * stream2.sample_count
        )
        if index > stream2.sample_count - 1:
            index = stream2.sample_count - 1
        new_data.append(stream2.data[index])

    return Sensor3AxisStream(
        stream2.key,
        stream1.duration,
        stream1.sample_count,
        stream2.name,
        stream2.units,
        new_data,
    )


def get_stream_offset(stream1: Sensor3AxisStream, stream2: Sensor3AxisStream) -> float:
    """
    Given 2 sensor streams, determines the offset of `stream2` from `stream1`
    using cross-correlation. For this to work correctly, the streams must share
    distinct features, the offset must be small relative to the total duration,
    and the clips must cover a similar time frame.

    Returns the offset of `stream2` from `stream1` in milliseconds (positive
    means `stream2` is behind, negative means it's ahead).
    """

    array1 = np.array(stream1.data)
    array2 = np.array(regularize_stream_timescale(stream1, stream2).data)

    magnitudes1 = (array1 * array1).sum(axis=1)
    magnitudes2 = (array2 * array2).sum(axis=1)

    # Pad with median so that we get clearer results than padding with 0
    median_padding = np.full(magnitudes1.size, np.median(magnitudes1))
    magnitudes1_padded = np.concatenate((median_padding, magnitudes1, median_padding))

    corr = correlate(magnitudes1_padded, magnitudes2, mode="valid")

    shift = -1 * (np.argmax(corr) - (len(magnitudes1) - 1))
    offset = (stream1.duration / stream1.sample_count) * shift

    return offset
