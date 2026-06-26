"""Build the Cython extension modules for multiband_bls."""

from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

COMPILE_ARGS = ["-O3", "-ffast-math"]

extensions = [
    Extension(
        "multiband_bls._sbls",
        ["src/multiband_bls/_sbls.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=COMPILE_ARGS,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "multiband_bls._msbls",
        ["src/multiband_bls/_msbls.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=COMPILE_ARGS,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "multiband_bls._eebls",
        ["src/multiband_bls/_eebls.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=COMPILE_ARGS,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "multiband_bls._meebls",
        ["src/multiband_bls/_meebls.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=COMPILE_ARGS,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
