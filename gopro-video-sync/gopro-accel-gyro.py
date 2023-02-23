from dataclasses import dataclass
from math import ceil
from os.path import getsize
from typing import Dict, List, BinaryIO, Tuple


@dataclass()
class Box:
    """Holds information about an MP4 box."""

    key: str
    offset: int
    size: int


def get_boxes(f: BinaryIO, offset: int, size: int, box_path: List[str]) -> List[Box]:
    """
    Given a binary stream representing part of an MP4 file, a start offset, the
    size of the starting box, and a "path" of box keys leading to the desired
    boxes, returns a list of Box objects representing the final boxes. If the
    specified boxes do not exist, returns an empty list.

    An example `box_path` might be `["moov", "udta", "HMMT"]`.

    To search a whole MP4 file, pass in the following:
      - `f`: stream obtained from `open()` using `"rb"`
      - `offset`: `0`
      - `size`: size of file in bytes
      - `box_path`: desired path starting from root level
    """

    # Loops through top-level boxes, recursing when we find a box matching the
    # first level of the path
    boxes = []
    end = offset + size
    f.seek(offset)
    while offset < end:
        header = f.read(8)
        size = int.from_bytes(header[0:4])
        key = header[4:8].decode("ascii")
        if key == box_path[0]:
            if len(box_path) == 1:
                boxes += [Box(key, offset, size)]
            else:
                boxes += get_boxes(f, offset + 8, size - 8, box_path[1:])
        offset += size
        f.seek(offset)
    return boxes


@dataclass()
class Sample:
    """Holds information about a sample."""

    time_delta: int
    duration: int
    offset: int
    size: int


def get_samples(f: BinaryIO, stbl: Box) -> List[Sample]:
    """
    Given a binary stream representing part of an MP4 file and the location of
    the "stbl" box, returns an array of Sample objects which describe where to
    find each sample along with some of their properties.
    """

    # Get duration of each sample from "stts" atom (https://developer.apple.com/library/archive/documentation/QuickTime/QTFF/QTFFChap2/qtff2.html#//apple_ref/doc/uid/TP40000939-CH204-BBCGFJII)
    sample_durations = []
    stts = get_boxes(f, stbl.offset, stbl.size, box_path=["stbl", "stts"])[0]
    f.seek(stts.offset + 12)
    num_entries = int.from_bytes(f.read(4))
    for _ in range(num_entries):
        sample_count = int.from_bytes(f.read(4))
        sample_duration = int.from_bytes(f.read(4))
        sample_durations += [sample_duration] * sample_count

    # Get size of each sample from "stsz" atom (https://developer.apple.com/library/archive/documentation/QuickTime/QTFF/QTFFChap2/qtff2.html#//apple_ref/doc/uid/TP40000939-CH204-BBCBBCGB)
    sample_sizes = []
    stsz = get_boxes(f, stbl.offset, stbl.size, box_path=["stbl", "stsz"])[0]
    f.seek(stsz.offset + 12)
    sample_size = int.from_bytes(f.read(4))
    num_entries = int.from_bytes(f.read(4))
    if sample_size != 0:
        sample_sizes = [sample_size] * num_entries
    else:
        for _ in range(num_entries):
            sample_sizes.append(int.from_bytes(f.read(4)))

    # Get offset of each sample from "stco" atom (https://developer.apple.com/library/archive/documentation/QuickTime/QTFF/QTFFChap2/qtff2.html#//apple_ref/doc/uid/TP40000939-CH204-BBCHAEEA)
    sample_offsets = []
    stco = get_boxes(f, stbl.offset, stbl.size, box_path=["stbl", "stco"])[0]
    f.seek(stco.offset + 12)
    num_entries = int.from_bytes(f.read(4))
    for _ in range(num_entries):
        sample_offsets.append(int.from_bytes(f.read(4)))

    # Compile array of Sample objects while calculating cumulative time delta
    samples = []
    time_delta = 0
    for i in range(len(sample_durations)):
        samples.append(
            Sample(
                time_delta,
                sample_durations[i],
                sample_offsets[i],
                sample_sizes[i],
            )
        )
        time_delta += sample_durations[i]

    return samples


@dataclass()
class GPMFBox:
    """Holds information about a GPMF box."""

    key: str
    offset: int
    size: int
    type: str
    struct_size: int
    repeat: int


def get_gpmf_boxes(
    f: BinaryIO, offset: int, size: int, box_path: List[str]
) -> List[GPMFBox]:
    """
    Given a binary stream representing part of a GPMF payload, a start offset,
    the size of the starting box, and a "path" of box keys leading to the
    desired boxes, returns a list of GPMFBox objects representing the final
    boxes. If the specified boxes do not exist, returns an empty list.

    An example `box_path` might be `["DEVC", "STRM", "ACCL"]`.

    See: https://github.com/gopro/gpmf-parser#definitions
    """

    # Loops through top-level boxes, recursing when we find a box matching the
    # first level of the path
    boxes = []
    end = offset + size
    f.seek(offset)
    while offset < end:
        header = f.read(8)
        key = header[0:4].decode("ascii")
        if int.from_bytes(header[4:5]) == 0:
            type = None
        else:
            type = header[4:5].decode("ascii")
        struct_size = int.from_bytes(header[5:6])
        repeat = int.from_bytes(header[6:8])
        # 32-bit aligned (https://github.com/gopro/gpmf-parser#alignment-and-storage)
        size = ceil((struct_size * repeat + 8) / 4) * 4
        if key == box_path[0]:
            if len(box_path) == 1:
                boxes += [GPMFBox(key, offset, size, type, struct_size, repeat)]
            elif type is None:
                boxes += get_gpmf_boxes(f, offset + 8, size - 8, box_path[1:])
        offset += size
        f.seek(offset)
    return boxes


@dataclass()
class Sensor3AxisStream:
    """Holds information about a stream of 3-axis sensor samples from a GoPro."""

    key: str
    total_duration: int
    total_sample_count: int
    name: str
    units: str

    data: List[Tuple[float, float, float]]


def get_3axis_sensor_data(
    f: BinaryIO, samples: List[Sample], sensor_key: str
) -> Sensor3AxisStream:
    """
    Given a binary stream representing part of an MP4 file, a list of Sample
    objects, and the FourCC of the desired GPMF stream (either "ACCL" or
    "GYRO"), compiles the sensor data and other metadata across payloads.

    Returns a Sensor3AxisStream object.

    See: https://github.com/gopro/gpmf-parser#property-hierarchy
    """

    data = []
    total_duration = 0
    total_sample_count = 0
    name = ""
    units = ""
    for sample in samples:
        # Finds the top-level "strm" box for the sensor data stream
        strm = None
        strm_boxes = get_gpmf_boxes(f, sample.offset, sample.size, ["DEVC", "STRM"])
        for box in strm_boxes:
            if get_gpmf_boxes(f, box.offset, box.size, ["STRM", sensor_key]):
                strm = box
                break
        if strm is None:
            raise ValueError(f"GPMF payload does not contain {sensor_key} data.")

        # Gets the scale factor for the data
        scal = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "SCAL"])[0]
        f.seek(scal.offset + 8)
        scale_divisor = int.from_bytes(f.read(2), signed=True)

        # Extracts the actual samples and scales them
        sensor_box = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", sensor_key])[0]
        f.seek(sensor_box.offset + 6)
        num_samples = int.from_bytes(f.read(2))
        for _ in range(num_samples):
            data.append(
                tuple(
                    [
                        int.from_bytes(f.read(2), signed=True) / scale_divisor
                        for _ in range(3)
                    ]
                )
            )

        # Get the stream name
        if name == "":
            stnm = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "STNM"])[0]
            f.seek(stnm.offset + 5)
            name_length = int.from_bytes(f.read(1)) * int.from_bytes(f.read(2))
            name = f.read(name_length).decode("ascii")

        # Get units (https://github.com/gopro/gpmf-parser#standard-units-for-physical-properties-supported-by-siun)
        if units == "":
            siun = get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "SIUN"])[0]
            f.seek(siun.offset + 5)
            units_length = int.from_bytes(f.read(1)) * int.from_bytes(f.read(2))
            # Special characters (https://github.com/gopro/gpmf-parser#special-ascii-characters)
            for _ in range(units_length):
                unit_char = f.read(1)
                unit_char_value = int.from_bytes(unit_char)
                if unit_char_value == 0xB0:
                    units += "°"
                elif unit_char_value == 0xB2:
                    units += "²"
                elif unit_char_value == 0xB3:
                    units += "³"
                elif unit_char_value == 0xB5:
                    units += "µ"
                else:
                    units += unit_char.decode("ascii")

        total_duration += sample.duration
        total_sample_count += num_samples

    return Sensor3AxisStream(
        sensor_key, total_duration, total_sample_count, name, units, data
    )


def get_gopro_accel_gyro(video: str) -> Tuple[Sensor3AxisStream, Sensor3AxisStream]:
    """
    (mention that the format of data varies depending on camera model)
    """

    with open(video, "rb") as f:
        # Verify that the video contains a GPMF track and get the sample table (https://github.com/gopro/gpmf-parser#mp4-implementation)
        size = getsize(video)
        stbl = None
        minf_boxes = get_boxes(f, 0, size, ["moov", "trak", "mdia", "minf"])
        for box in minf_boxes:
            if get_boxes(f, box.offset, box.size, ["minf", "gmhd", "gpmd"]):
                stbl = get_boxes(f, box.offset, box.size, ["minf", "stbl"])[0]
                break
        if stbl is None:
            raise ValueError("Video does not contain GPMF data.")

        samples = get_samples(f, stbl)

        return (
            get_3axis_sensor_data(f, samples, "ACCL"),
            get_3axis_sensor_data(f, samples, "GYRO"),
        )


print(get_gopro_accel_gyro("d:\\E\\Movies\\3D\\2023-02-21\\L\\GOPR1584.MP4"))
