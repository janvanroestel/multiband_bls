"""Build the Cython extension modules for multiband_bls."""

from __future__ import annotations

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# -fno-finite-math-only keeps NaN handling defined (the multiband cores store NaN
# for unmeasured bands) while the rest of -ffast-math still applies. MSVC has no
# such flags: plain /O2 there.
COMPILE_ARGS = ["/O2"] if sys.platform == "win32" else [
    "-O3", "-ffast-math", "-fno-finite-math-only"]

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
    Extension(
        "multiband_bls._fast",
        ["src/multiband_bls/_fast.pyx"],
        include_dirs=[np.get_include(), "src/multiband_bls"],
        depends=["src/multiband_bls/_fast_kernels.h"],
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
