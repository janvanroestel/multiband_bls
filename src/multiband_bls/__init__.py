"""Sparse BLS and multiband Sparse BLS for sparse, multi-band light curves.

Implements the Sparse Box-fitting Least Squares algorithm (Panahi & Zucker
2021) and a multiband extension that shares the period/epoch/duration grid
across photometric bands while fitting an independent depth per band.

Quick start::

    from multiband_bls import multiband_eebls, build_frequency_grid

    freqs = build_frequency_grid(t, p_min=0.1, p_max=1.0)
    result = multiband_eebls(bands, freqs)   # bands: {label: (t, y, dy)}
    print(result.best_period, result.best_power)

All entry points return :class:`SBLSResult`.  ``power`` is
``Δχ²/χ²_flat ∈ [0, 1]`` (fraction of variance explained).

GPU acceleration (requires CuPy) is available via :func:`eebls_gpu`,
:func:`multiband_eebls_gpu`, and their ``_fast`` variants.  Check
:func:`gpu_available` before use.

When the Cython extension is not installed all four search functions fall back
to pure-Python reference implementations automatically (with a warning).
"""

from __future__ import annotations

from .periodogram import SBLSResult
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
# When unavailable, GPU names are set to None. Always call gpu_available() before
# using eebls_gpu / multiband_eebls_gpu or their _fast variants.
try:
    from .gpu import (  # noqa: E402
        eebls_gpu,
        eebls_gpu_fast,
        gpu_available,
        multiband_eebls_gpu,
        multiband_eebls_gpu_fast,
    )
except ImportError:  # pragma: no cover - needs the compiled cores
    eebls_gpu = multiband_eebls_gpu = None  # type: ignore[assignment]
    eebls_gpu_fast = multiband_eebls_gpu_fast = None  # type: ignore[assignment]

    def gpu_available() -> bool:  # type: ignore[misc]
        return False

__all__ = [
    "SBLSResult",
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
