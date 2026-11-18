"""GPU (CuPy ``RawKernel``) backends for the binned BLS methods.

One CUDA thread-block per trial frequency: the block bins the (small, shared)
light curve for its own frequency into shared-memory bins, runs the box search,
and writes one periodogram value. This mirrors the compiled binned cores
(:mod:`multiband_bls._eebls`, :mod:`multiband_bls._meebls`) and returns the same
``power = Delta chi^2 / chi^2_flat``.

The periodogram (the expensive part) is computed on the GPU; the best period's
``t0``/duration/depth are recomputed once on the CPU. Fallback is the caller's
responsibility -- :func:`gpu_available` reports whether a usable device exists.

CuPy is imported lazily so the package still imports without it / without a GPU.
"""

from __future__ import annotations

__all__ = [
    "gpu_available",
    "eebls_gpu",
    "multiband_eebls_gpu",
    "eebls_gpu_fast",
    "multiband_eebls_gpu_fast",
]

import logging
from collections.abc import Mapping
from importlib import resources
from typing import Any

import numpy as np
import numpy.typing as npt

from ._common import (  # noqa: F401  (_grid_params/_log_widths: used by tests and backends)
    _check_coarse,
    _check_dlogq,
    _grid_params,
    _log_widths,
)
from ._prep import auto_nbins
from ._validate import _check_sign, _validate_search_params
from .api import _search
from .engine import CapabilityError, RunOptions, SegmentSpec
from .result import BLSResult

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]

_THREADS = 256  # threads per block. Hard-coded into the kernels: the exact
# kernels size their static reduction as `red[256]`, and the fast kernels assume
# exactly 256/32 = 8 warps (`warp_max[8]`). Changing this requires editing those
# reduction sizes in the kernel source and recompiling.
# The CUDA multiband kernels are specialised per band count (-DNB), so the band loops
# unroll and the per-band arrays live in registers. The arrays are still declared with
# a fixed size [8]; change _MAX_BANDS together with every such declaration (bw, bn, s,
# r, cnt in the exact kernel; bw in the fast kernel). Each band count compiles once.
_MAX_BANDS: int = 8
# The fast kernels count in-transit points in float32 bins and gate them against
# ``count - 0.5f``; both are exact only below 2**23 (float32 spacing <= 0.5), so
# larger light curves are rejected.
_MAX_FAST_POINTS: int = 2**23
# The kernels take point and frequency counts as C int; larger inputs would wrap.
_MAX_INT32: int = int(np.iinfo(np.int32).max)
# The fast kernels hold median-centred times and frequencies in float32; keep
# both far below float32's ~3.4e38 so the casts and fold products stay finite.
_MAX_FAST_FLOAT32: float = 1e30
# The float-float fold's absolute phase error grows ~ 7e-15 per cycle of
# max|t_centred| * max(f) (measured: <= 4e-8 up to 1e6 cycles, 7e-7 at 1e8, 7e-5 at
# 1e10); beyond 1e8 cycles it is no longer small against a 1/nbins bin.
_MAX_FAST_CYCLES: float = 1e8
# Measured phase error of the float-float fold (RTX 3050 Ti, 2e5 random times vs an
# 80-bit reference, independent of f at fixed cycles): 3-8e-8 up to 1e7 cycles, 4-8e-7 at
# 1e8, 6-7e-6 at 1e9, 5-6e-5 at 1e10, i.e. about max(1e-7, 1e-14 * cycles). Even at the
# 1e8-cycle limit that is far below one bin for any nbins the kernels' shared memory
# allows, so no precision warning is issued.


def _check_int32_sizes(n: int, nf: int, name: str) -> None:
    """Raise unless the point and frequency counts fit the kernels' 32-bit ints."""
    if n > _MAX_INT32 or nf > _MAX_INT32:
        raise CapabilityError(
            f"{name}: {n} points and {nf} frequencies; both must be at most "
            f"{_MAX_INT32} (32-bit kernel indices). Split the frequency grid."
        )


def _check_fast_float32(t_centered: Array, frequencies: Array, name: str) -> None:
    """Raise unless centred times and frequencies fit the fast kernels' float32."""
    t_max = float(np.max(np.abs(t_centered)))
    f_max = float(np.max(frequencies))
    if t_max > _MAX_FAST_FLOAT32 or f_max > _MAX_FAST_FLOAT32:
        raise CapabilityError(
            f"{name}: median-centred |t| ({t_max:.3g}) and frequencies "
            f"({f_max:.3g}) must stay below {_MAX_FAST_FLOAT32:.0e} for the "
            "float32 kernel; use the exact GPU entry point."
        )
    if t_max * f_max > _MAX_FAST_CYCLES:
        raise CapabilityError(
            f"{name}: median-centred max|t| * max(f) = {t_max * f_max:.3g} cycles "
            f"exceeds {_MAX_FAST_CYCLES:.0e}, beyond which the float32 phase fold "
            "loses accuracy; use the exact GPU entry point."
        )


# --------------------------------------------------------------------------- #
# CUDA kernels                                                                 #
# --------------------------------------------------------------------------- #
def _source(name: str) -> str:
    """Text of the CUDA source ``kernels/<name>`` (package data, shipped in wheels/sdists)."""
    return resources.files("multiband_bls.kernels").joinpath(name).read_text(encoding="utf-8")


_SINGLE_SRC = _source("single.cu")
_MULTI_SRC = _source("multi.cu")
_FAST_HELPERS = _source("fast_helpers.cu")
_FAST_BOX_SINGLE_SRC = _FAST_HELPERS + _source("fast_single.cu")
_FAST_BOX_MULTI_SRC = _FAST_HELPERS + _source("fast_multi.cu")

_single_kernel: Any = None
_fast_single_kernel: Any = None
# Multiband kernels are specialised on the band count (-DNB) so the per-band loops
# unroll; one compiled kernel per (n_bands, with_k), built on first use.
_multi_kernels: dict[int, Any] = {}
_fast_multi_kernels: dict[tuple[int, bool, int], Any] = {}
_smem_per_sm: int = 102400  # updated at first _kernels() call from device attributes
_max_smem: int = 65536  # updated at first _kernels() call from device attributes


def gpu_available() -> bool:
    """True if CuPy is importable and a CUDA device is usable."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _opt_in_smem(kernel: Any) -> Any:
    """Raise ``kernel``'s dynamic shared-memory limit to the device maximum.

    The requested dynamic size must leave room for the kernel's own static shared
    memory (the exact kernels carry a static ``red[256]`` reduction buffer), so the
    static size is subtracted from the opt-in ceiling.
    """
    kernel.max_dynamic_shared_size_bytes = _max_smem - kernel.attributes["shared_size_bytes"]
    return kernel


def _kernels() -> None:
    global _single_kernel, _fast_single_kernel, _max_smem, _smem_per_sm
    if _single_kernel is None:
        import cupy as cp

        attrs = cp.cuda.Device().attributes
        _max_smem = attrs["MaxSharedMemoryPerBlockOptin"]
        _smem_per_sm = attrs["MaxSharedMemoryPerMultiprocessor"]
        _single_kernel = _opt_in_smem(cp.RawKernel(_SINGLE_SRC, "eebls_kernel"))
        _fast_single_kernel = _opt_in_smem(
            cp.RawKernel(_FAST_BOX_SINGLE_SRC, "eebls_fast_kernel")
        )
    # the multiband kernels are built per band count by the *_multi_kernel getters


def _multi_kernel_for(n_bands: int) -> Any:
    """Exact multiband kernel specialised for ``n_bands`` bands (cached)."""
    _kernels()
    k = _multi_kernels.get(n_bands)
    if k is None:
        import cupy as cp

        k = _opt_in_smem(cp.RawKernel(_MULTI_SRC, "meebls_kernel",
                                      options=(f"-DNB={n_bands}",)))
        _multi_kernels[n_bands] = k
    return k


def _min_blocks_hint(shared: int) -> int:
    """Minimum resident blocks per SM for ``__launch_bounds__``: what shared memory
    allows (about 1 KB per block is reserved), clamped to [1, 4]."""
    return max(1, min(4, _smem_per_sm // (shared + 1024)))


def _fast_multi_kernel_for(n_bands: int, with_k: bool, shared: int) -> Any:
    """Fast multiband kernel specialised for ``n_bands`` bands (cached).

    ``with_k`` selects the variant that also writes ``n_bands_used``; ``shared`` is the
    dynamic shared memory per block, which sets the launch-bounds block hint.
    """
    _kernels()
    minb = _min_blocks_hint(shared)
    k = _fast_multi_kernels.get((n_bands, with_k, minb))
    if k is None:
        import cupy as cp

        opts = (f"-DNB={n_bands}", f"-DMINB={minb}") + (("-DWITH_K",) if with_k else ())
        k = _opt_in_smem(cp.RawKernel(_FAST_BOX_MULTI_SRC, "meebls_fast_kernel", options=opts))
        _fast_multi_kernels[(n_bands, with_k, minb)] = k
    return k


def _split_time(t: Array) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Split times into (hi, lo) float32 pairs with ``t ~= hi + lo``.

    Feeds the float-float phase fold in the fast kernels. ``t`` must already
    be median-reduced so the products ``t * freq`` stay well inside the range
    where the two-float representation holds full double-like precision.
    """
    t64 = np.asarray(t, dtype=np.float64)
    t_hi = t64.astype(np.float32)
    t_lo = (t64 - t_hi.astype(np.float64)).astype(np.float32)
    return t_hi, t_lo


def eebls_gpu(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    sign: str = "both",
) -> BLSResult:
    """GPU binned BLS (single band). Mirrors :func:`multiband_bls.eebls`.

    Parameters
    ----------
    t :
        Observation times (days), 1-D float array.
    y :
        Magnitudes or fluxes, 1-D float array, same length as ``t``.
    dy :
        1-sigma uncertainties, 1-D float array, same length as ``t``.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required.
    sign :
        ``"both"`` (default), ``"positive"`` or ``"negative"``: keep only windows whose
        weighted mean of ``y`` is above / below the out-of-window baseline (per band in
        the multiband searches). See :func:`multiband_bls.eebls`.

    """
    import cupy  # noqa: F401  (fail before any validation if CuPy is missing)

    sg = _check_sign(sign, "eebls_gpu")
    freqs_np, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    return _search("binned", False, (t, y, dy), freqs_np,
                   SegmentSpec(float(q_min), float(q_max), nbins),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign), backend="gpu")


def multiband_eebls_gpu(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    rescale_errors: bool = False,
    sign: str = "both",
) -> BLSResult:
    """GPU binned multiband BLS. Mirrors :func:`multiband_bls.multiband_eebls`.

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch, and duration.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required across all bands.
    rescale_errors :
        As in :func:`multiband_bls.multiband_sparse_bls`.
    sign :
        ``"both"`` (default), ``"positive"`` or ``"negative"``: keep only windows whose
        weighted mean of ``y`` is above / below the out-of-window baseline (per band in
        the multiband searches). See :func:`multiband_bls.eebls`.

    Notes
    -----
    Fills :attr:`BLSResult.n_bands_used` like the CPU core. Where several windows
    tie exactly for the best statistic, the GPU reduction may keep a different one
    than the CPU's first-found window, so ``k`` can then differ.
    """
    import cupy  # noqa: F401  (fail before any validation if CuPy is missing)

    sg = _check_sign(sign, "multiband_eebls_gpu")
    freqs_np, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    return _search("binned", True, bands, freqs_np,
                   SegmentSpec(float(q_min), float(q_max), nbins),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign),
                   rescale_errors=rescale_errors, backend="gpu")


def eebls_gpu_fast(
    t: Array,
    y: Array,
    dy: Array,
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    dlogq: float = 0.3,
    coarse_factor: int = 1,
    n_refine: int = 20,
    sign: str = "both",
) -> BLSResult:
    """Fast GPU binned BLS using prefix-sum bins and log-spaced transit widths.

    Instead of evaluating all integer widths from ``kmi`` to ``kma``, only
    evaluates a geometrically-spaced subset, reducing window evaluations from
    O(kma - kmi) to O(log(kma/kmi) / log(1 + dlogq)). The phase bins are
    turned into cumulative sums so each window is an O(1) lookup, and the
    phase fold runs in float-float (two-float32) arithmetic, which is several
    times faster than float64 on consumer GPUs, with an absolute phase error of
    <= ~4e-8 for up to 1e6 cycles (median-centred ``max|t| * max(f)``); inputs beyond
    1e8 cycles are rejected.

    Parameters
    ----------
    t :
        Observation times (days), 1-D float array.
    y :
        Magnitudes or fluxes, 1-D float array, same length as ``t``.
    dy :
        1-sigma uncertainties, 1-D float array, same length as ``t``.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required.
    dlogq :
        Logarithmic spacing between trial widths. Smaller = more widths = higher
        fidelity but slower. ``dlogq=0`` recovers the full linear search.
        Typical range: 0.1 – 0.5.
    coarse_factor, n_refine :
        Opt-in coarse-to-fine search (default ``coarse_factor=1``: the full grid).
        Only every ``coarse_factor``-th frequency is searched, then the full grid
        around the ``n_refine`` highest separated coarse peaks; see
        :func:`multiband_bls.eebls_fast`. Can miss narrow or weak peaks, so validate
        recovery against the full grid first.
    sign :
        ``"both"`` (default), ``"positive"`` or ``"negative"``: keep only windows whose
        weighted mean of ``y`` is above / below the out-of-window baseline (per band in
        the multiband searches). See :func:`multiband_bls.eebls`.

    Notes
    -----
    Introduces a small (~1–2%) loss in peak power because the exact optimal
    integer width may not fall in the log-spaced set. For detection this is
    negligible; for precise parameter estimation use :func:`eebls_gpu` or
    :func:`eebls` at the recovered period.

    ``min_points`` is an exact global in-transit point count (same semantics
    as :func:`multiband_bls.eebls` and :func:`eebls_gpu`). Bins and sums are
    float32, so low-power (sub-peak) periodogram values are less accurate than
    the exact kernels'. ``best_t0``/``best_duration``/``best_depth`` are
    recomputed exactly on the CPU at the best frequency, while ``power`` and
    ``best_power`` stay the fast kernel's values; the two need not describe
    exactly the same box.
    Light curves with ``n >= 2**23`` points are rejected (float32 counts).
    """
    import cupy  # noqa: F401  (fail before any validation if CuPy is missing)

    sg = _check_sign(sign, "eebls_gpu_fast")
    freqs_np, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    dlogq = _check_dlogq(dlogq, "eebls_gpu_fast")
    coarse_factor, n_refine = _check_coarse(coarse_factor, n_refine, "eebls_gpu_fast")
    return _search("fast", False, (t, y, dy), freqs_np,
                   SegmentSpec(float(q_min), float(q_max), nbins),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign, dlogq=dlogq,
                              coarse_factor=coarse_factor, n_refine=n_refine),
                   backend="gpu")


def multiband_eebls_gpu_fast(
    bands: Mapping[str, tuple[Array, Array, Array]],
    frequencies: Array,
    nbins: int | None = None,
    q_min: float = 0.01,
    q_max: float = 0.10,
    min_points: int = 3,
    dlogq: float = 0.3,
    rescale_errors: bool = False,
    compute_n_bands_used: bool = True,
    coarse_factor: int = 1,
    n_refine: int = 20,
    sign: str = "both",
) -> BLSResult:
    """Fast GPU binned multiband BLS using prefix-sum bins and log-spaced widths.

    Same design as :func:`eebls_gpu_fast`: cumulative-sum phase bins (O(1)
    window sums), a geometrically-spaced width subset, and a float-float
    phase fold.

    Parameters
    ----------
    bands :
        Mapping ``label -> (t, y, dy)``. Each band keeps its own baseline and
        depth; all bands share the trial period, epoch, and duration.
    frequencies :
        Trial frequencies (1/day).
    nbins :
        Number of phase bins. Chosen automatically as ``clip(5 / q_min, 50, 500)``
        if ``None``.
    q_min :
        Minimum transit duration as a fraction of the period.
    q_max :
        Maximum transit duration as a fraction of the period.
    min_points :
        Minimum number of in-transit points required across all bands.
    dlogq :
        Logarithmic width spacing. Smaller = more widths = higher fidelity.
        Typical range: 0.1 – 0.5.
    rescale_errors :
        As in :func:`multiband_bls.multiband_sparse_bls`.
    compute_n_bands_used :
        Fill :attr:`BLSResult.n_bands_used` (default). Pass False to leave it
        ``None`` and skip the count, which costs up to ~15% kernel time for sparse
        light curves with few bands (measured: 3 bands, n = 1k–3k points) and
        ~1–5% otherwise.
    coarse_factor, n_refine :
        Opt-in coarse-to-fine search, as in :func:`eebls_gpu_fast`. With
        ``coarse_factor > 1`` :attr:`BLSResult.n_bands_used` is ``None`` (it is not
        defined at the interpolated frequencies), unless the grid is too short
        (``< 4 * coarse_factor`` frequencies) for any frequency to be skipped.
    sign :
        ``"both"`` (default), ``"positive"`` or ``"negative"``: keep only windows whose
        weighted mean of ``y`` is above / below the out-of-window baseline (per band in
        the multiband searches). See :func:`multiband_bls.eebls`.

    Notes
    -----
    Introduces a small (~1–2%) loss in peak power because the exact optimal
    integer width may not fall in the log-spaced set.

    ``min_points`` is an exact global in-transit point count across all bands,
    and a band contributes only with >= 1 point in the window (same semantics
    as :func:`multiband_bls.multiband_eebls` and :func:`multiband_eebls_gpu`).
    Bins and sums are float32, so low-power (sub-peak) periodogram values are
    less accurate than the exact kernels'. The best period's parameters are
    recomputed exactly on the CPU, while ``power``/``best_power`` stay the fast
    kernel's values (the two need not describe exactly the same box). Light
    curves with ``n >= 2**23`` points in total are rejected (float32 counts).

    ``n_bands_used`` counts the bands of the window this kernel
    selects. It matches :func:`multiband_bls.multiband_eebls` when both search
    the same windows, but this function bins median-centred times (so its phase
    bins start elsewhere) and evaluates only log-spaced widths, so its best window,
    and hence ``k``, can differ at some frequencies.
    """
    import cupy  # noqa: F401  (fail before any validation if CuPy is missing)

    sg = _check_sign(sign, "multiband_eebls_gpu_fast")
    freqs_np, min_points, nbins = _validate_search_params(
        frequencies, q_max, min_points, q_min=q_min, nbins=nbins
    )
    if nbins is None:
        nbins = auto_nbins(q_min)
    dlogq = _check_dlogq(dlogq, "multiband_eebls_gpu_fast")
    coarse_factor, n_refine = _check_coarse(coarse_factor, n_refine,
                                            "multiband_eebls_gpu_fast")
    return _search("fast", True, bands, freqs_np,
                   SegmentSpec(float(q_min), float(q_max), nbins),
                   RunOptions(min_points=min_points, sign=sg, sign_name=sign, dlogq=dlogq,
                              coarse_factor=coarse_factor, n_refine=n_refine,
                              compute_n_bands_used=compute_n_bands_used),
                   rescale_errors=rescale_errors, backend="gpu")


