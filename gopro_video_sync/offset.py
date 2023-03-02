from enum import Enum
from typing import Optional, Tuple

from .audio_offset import get_audio_offset, get_wav_data
from .gopro_data import get_gopro_accel_gyro
from .sensor_offset import get_sensor_offset


class OffsetSource(Enum):
    UNKNOWN = 0
    GYROSCOPE = 1
    ACCELEROMETER = 2
    AUDIO = 3


def gopro_offset(
    video_1: str, video_2: str, agree_threshold=0.03
) -> Tuple[Optional[float], OffsetSource, OffsetSource]:
    """
    Given the paths to two GoPro video files, compares accelerometer, gyroscope,
    and audio data as necessary to obtain an offset in seconds from `video_1`
    to `video_2` (positive means `video_2` is behind `video_1` and vice versa).

    First obtains offsets for the accel and gyro, checking if they are within
    `agree_threshold` seconds of each other. If they are, gyro offset is used,
    as it is usually more accurate. If they don't agree, audio offset is
    compared to both of them, and if one agrees, then the measurement from that
    sensor is used. In each case, a sensor will be the "primary source" and the
    agreeing source will be the "secondary source".

    Returns (offset, primary source, secondary source) where offset is the
    offset in seconds and the primary and secondary sources are OffsetSource
    enums. If no sources agree, then the offset will be None and both
    sources will be `OffsetSource.UNKNOWN`.
    """

    accel_1, gyro_1 = get_gopro_accel_gyro(video_1)
    accel_2, gyro_2 = get_gopro_accel_gyro(video_2)

    accel_offset = get_sensor_offset(accel_1, accel_2)
    gyro_offset = get_sensor_offset(gyro_1, gyro_2)

    if abs(accel_offset - gyro_offset) <= agree_threshold:
        return (gyro_offset, OffsetSource.GYROSCOPE, OffsetSource.ACCELEROMETER)

    wav_1 = get_wav_data(video_1)
    wav_2 = get_wav_data(video_2)

    audio_offset = get_audio_offset(wav_1, wav_2)

    if abs(audio_offset - gyro_offset) <= agree_threshold:
        return (gyro_offset, OffsetSource.GYROSCOPE, OffsetSource.AUDIO)

    if abs(audio_offset - accel_offset) <= agree_threshold:
        return (accel_offset, OffsetSource.ACCELEROMETER, OffsetSource.AUDIO)

    return (None, OffsetSource.UNKNOWN, OffsetSource.UNKNOWN)
