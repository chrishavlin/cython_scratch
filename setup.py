import numpy as np
from setuptools import Extension, setup

include_path = [np.get_include()]

extensions = [
    Extension(
        "cython_scratch.np_exps",
        ["cython_scratch/np_exps.pyx"],
        include_dirs=include_path,
    ),
]

setup(ext_modules=extensions)