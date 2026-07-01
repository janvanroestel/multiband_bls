"""Sparse BLS and multiband Sparse BLS for sparse, multi-band light curves.

Implements the Sparse Box-fitting Least Squares algorithm (Panahi & Zucker
2021) and a multiband extension that shares the period/epoch/duration grid
across photometric bands while fitting an independent depth per band.

Quick start::

    from multiband_bls import multiband_eebls, build_frequency_grid

    freqs = build_frequency_grid(t, p_min=0.1, p_max=1.0)
    result = multiband_eebls(bands, freqs)   # bands: {label: (t, y, dy)}
    print(result.best_period, result.best_power)

All entry points return :class:`BLSResult`.  ``power`` is
``Δχ²/χ²_flat ∈ [0, 1]`` (fraction of variance explained).

GPU acceleration (requires CuPy) is available via :func:`eebls_gpu`,
:func:`multiband_eebls_gpu`, and their ``_fast`` variants.  Check
:func:`gpu_available` before use.

When the Cython extension is not installed all four search functions fall back
to pure-Python reference implementations automatically (with a warning).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .periodogram import BLSResult

try:
    __version__ = version("multiband_bls")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0"
from .reference import (
    build_frequency_grid,
    coadd_bands,
    eebls_reference,
    multiband_eebls_reference,
    multiband_sparse_bls_reference,
    sparse_bls_reference,
)

try:  # compiled cores are optional until the extension is built
    from .api import (
        eebls,
        multiband_eebls,
        multiband_sparse_bls,
        sparse_bls,
    )
except ImportError:  # pragma: no cover
    import warnings

    warnings.warn(
        "Compiled Cython cores not found; build with `pip install -e .` to use "
        "`sparse_bls`/`multiband_sparse_bls`/`eebls`/`multiband_eebls`. "
        "Falling back to pure-Python reference implementations (~100× slower).",
        stacklevel=2,
    )
    sparse_bls = sparse_bls_reference  # type: ignore[assignment]
    multiband_sparse_bls = multiband_sparse_bls_reference  # type: ignore[assignment]
    eebls = eebls_reference  # type: ignore[assignment]
    multiband_eebls = multiband_eebls_reference  # type: ignore[assignment]

# GPU (CuPy) backends -- import lazily; cupy itself is only needed at call time.
# When the module can't be imported at all (missing compiled cores), the GPU
# names are bound to stubs that raise a clear ImportError when called, rather
# than None (which would fail later with a confusing `NoneType is not callable`).
try:
    from .gpu import (  # noqa: E402
        eebls_gpu,
        eebls_gpu_fast,
        gpu_available,
        multiband_eebls_gpu,
        multiband_eebls_gpu_fast,
    )
except ImportError:  # pragma: no cover - needs the compiled cores

    def _gpu_unavailable(*args: object, **kwargs: object) -> BLSResult:
        raise ImportError(
            "GPU backend unavailable (compiled cores not importable). "
            "Build the package (`pip install -e .`) and install CuPy "
            "(`pip install cupy-cuda12x`) to use the GPU entry points."
        )

    eebls_gpu = multiband_eebls_gpu = _gpu_unavailable  # type: ignore[assignment]
    eebls_gpu_fast = multiband_eebls_gpu_fast = _gpu_unavailable  # type: ignore[assignment]

    def gpu_available() -> bool:  # type: ignore[misc]
        return False

__all__ = [
    "__version__",
    "BLSResult",
    "sparse_bls",
    "multiband_sparse_bls",
    "eebls",
    "multiband_eebls",
    "eebls_gpu",
    "eebls_gpu_fast",
    "multiband_eebls_gpu",
    "multiband_eebls_gpu_fast",
    "gpu_available",
    "eebls_reference",
    "sparse_bls_reference",
    "multiband_sparse_bls_reference",
    "multiband_eebls_reference",
    "build_frequency_grid",
    "coadd_bands",
]
