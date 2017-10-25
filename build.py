import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['cctc.cc']
headers = ['cctc.h']
defines = []

ffi = create_extension(
    'cctc',
    # package=True,
    # with_cuda=with_cuda
    headers=["cctc.h"],
    sources=["cctc.cc"],
    define_macros=defines,
    relative_to=__file__,
    language="c++",
    extra_compile_args=["--std=c++11", "-fsanitize=leak"],
)

if __name__ == '__main__':
    ffi.build()
