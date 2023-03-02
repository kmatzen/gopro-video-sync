import numpy as np

from .cross_correlation import get_offset
from .gopro_data import Sensor3AxisStream


def regularize_stream_timescale(
    stream_1: Sensor3AxisStream, stream_2: Sensor3AxisStream, same_length: bool = False
) -> Sensor3AxisStream:
    """
    Given 2 sensor streams, regularizes the timescale of `stream_2` to match
    `stream_1`, duplicating/skipping samples as necessary. If `same_length` is
    `True`, `stream_2` is trimmed/extended to match the length of `stream_1`,
    padding the end with the last value present in `stream_2` if necessary.

    Returns the regularized `stream_2` as a Sensor3AxisStream object.
    """

    new_data = []
    i = 0
    length = (
        stream_1.sample_count
        if same_length
        else int(stream_1.sample_count / stream_1.duration * stream_2.duration)
    )
    for i in range(length):
        index = round(
            ((stream_1.duration / stream_1.sample_count) * i / stream_2.duration)
            * stream_2.sample_count
        )
        if index > stream_2.sample_count - 1:
            index = stream_2.sample_count - 1
        new_data.append(stream_2.data[index])

    return Sensor3AxisStream(
        stream_2.key,
        stream_1.duration,
        stream_1.sample_count,
        stream_2.name,
        stream_2.units,
        new_data,
    )


def get_sensor_offset(
    stream_1: Sensor3AxisStream, stream_2: Sensor3AxisStream
) -> float:
    """
    Given 2 sensor streams, determines the offset of `stream_2` from `stream_1`
    using cross-correlation.

    Returns the offset in seconds that would have to be added to the beginning
    of `stream_2` for it to line up with `stream_1` (or vice versa for a
    negative value).
    """

    array1 = np.array(stream_1.data)
    array2 = np.array(regularize_stream_timescale(stream_1, stream_2).data)

    magnitudes_1 = (array1 * array1).sum(axis=1)
    magnitudes_2 = (array2 * array2).sum(axis=1)

    return get_offset(
        magnitudes_1, magnitudes_2, stream_1.sample_count / stream_1.duration * 1000
    )
