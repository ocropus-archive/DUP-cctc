import os
from setuptools import setup, find_packages
import build

setup(
    name="cctc",
    version="0.1",
    packages=find_packages(),
    ext_package="cctc",
    cffi_modules=["build.py:ffi"],
)
