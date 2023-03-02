from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="gopro-video-sync",
    packages=["gopro_video_sync"],
    version="0.0.1",
    author="Ethan Voth",
    author_email="ethanvoth7@gmail.com",
    url="https://github.com/evoth/gopro-video-sync",
    description="A tool to determine the offset between videos from two jointly mounted GoPros based on a combination of audio, accelerometer, and gyroscope data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "moviepy",
        "numpy",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
